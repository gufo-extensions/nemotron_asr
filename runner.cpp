#include "runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <gufo/ext/audio.hpp>
#include <gufo/ext/options.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "language.hpp"
#include "model.hpp"
#include "resampler.hpp"

namespace gufo::models::nemotron_asr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kRate = 16000;
// Uploads longer than this are refused; the streaming encoder itself has no
// length limit.
constexpr std::uint64_t kMaxSeconds = 4 * 3600;
// Live sessions update every 0.56 s (lookahead 6) unless --live-lookahead
// says otherwise; NVIDIA's processor defaults to 0.32 s (3), at about 0.4
// points more word errors on FLEURS.
constexpr std::size_t kLiveLookahead = 6;
// Encoder frames (80 ms each) per pass of live sessions and of uploads that
// stream their text: about 4.5 s of audio.
constexpr std::size_t kShortPassFrames = 56;
// Samples of an upload fed to the stream at a time (30 s), so that the
// features never hold the whole recording.
constexpr std::size_t kUploadBlock = 30 * kRate;

/// The lookahead of live sessions: --live-lookahead when given (one the
/// checkpoint supports), else 6 when the checkpoint supports it, else its
/// default.
std::size_t LiveLookahead(const Config& config,
                          const ext::ParsedOptions& options) {
  auto supported = config.encoder.lookaheads;
  const auto has = [&](std::size_t lookahead) {
    return std::find(supported.begin(), supported.end(), lookahead) !=
           supported.end();
  };
  if (const auto* chosen =
          ext::FindOption<std::int64_t>(options, "--live-lookahead")) {
    if (*chosen >= 0 && has(static_cast<std::size_t>(*chosen)))
      return static_cast<std::size_t>(*chosen);
    std::sort(supported.begin(), supported.end());
    std::string message = "--live-lookahead must be one of ";
    for (std::size_t i = 0; i < supported.size(); ++i)
      message += (i == 0 ? "" : ", ") + std::to_string(supported[i]);
    throw std::invalid_argument(message + " (the model's lookaheads), not " +
                                std::to_string(*chosen));
  }
  return has(kLiveLookahead) ? kLiveLookahead
                             : config.encoder.default_lookahead;
}

double Since(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

std::string Name(const TextDecoder& text, const Config& config,
                 std::size_t prompt) {
  if (text.language().has_value())
    return LanguageName(*text.language());
  for (const auto& [key, index] : config.prompt_dictionary)
    if (index == prompt && key != "auto" && key.find('-') != std::string::npos)
      return LanguageName(key);
  return {};
}

/// Live transcription of one utterance at any input rate. Its passes take
/// turns with other streams at the model's gate.
class Live : public ext::LiveTranscription {
public:
  Live(const Model& model, std::size_t lookahead, std::uint32_t rate)
      : resampler_(rate, kRate),
        stream_(model, lookahead, model.config().default_prompt,
                kShortPassFrames) {}

  bool Push(std::span<const float> samples, const Text& on_text,
            std::string* error) override {
    return Run(error, on_text, [&] {
      std::vector<float> resampled;
      resampler_.Push(samples, &resampled);
      stream_.Push(resampled);
    });
  }

  bool Finish(const Text& on_text, std::string* error) override {
    return Run(error, on_text, [&] {
      std::vector<float> resampled;
      resampler_.Finish(&resampled);
      stream_.Push(resampled);
      stream_.Finish();
    });
  }

  const std::string& text() const override { return text_; }

private:
  template<typename Work>
  bool Run(std::string* error, const Text& on_text, Work&& work) {
    try {
      work();
      std::string now = stream_.text().text();
      if (now.size() > text_.size() && now.starts_with(text_)) {
        const std::string delta = now.substr(text_.size());
        text_ = std::move(now);
        if (on_text && !on_text(delta))
          return false;
      }
      return true;
    } catch (const std::exception& failure) {
      if (error != nullptr)
        *error = failure.what();
      return false;
    }
  }

  Resampler resampler_;
  Stream stream_;
  std::string text_;
};

class Runner : public ext::AsrRunner {
public:
  Runner(std::unique_ptr<Model> model, std::size_t live_lookahead)
      : model_(std::move(model)), live_lookahead_(live_lookahead) {}

  bool Transcribe(const ext::TranscriptionRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::TranscriptionResult* result,
                  std::string* error) override {
    if (result == nullptr)
      return Fail(error, "the transcription result must not be null");
    *result = {};
    const auto total = Clock::now();
    try {
      const Config& config = model_->config();
      std::string failure;
      const auto prompt = ResolveLanguage(config, request.language, &failure);
      if (!prompt.has_value())
        return Fail(error, failure);
      if (request.wav.empty() == request.pcm16k.empty())
        return Fail(error, "Nemotron ASR needs exactly one audio source");

      const auto decode = Clock::now();
      std::vector<float> samples;
      std::uint32_t rate = kRate;
      if (!request.wav.empty()) {
        core::AudioLimits limits;
        limits.formats = core::AudioFormats::kAll;
        limits.max_seconds = kMaxSeconds;
        limits.downmix_mono = true;
        core::DecodedAudio audio = core::DecodeAudio(request.wav, limits, [&] {
          return is_cancelled && is_cancelled();
        });
        rate = audio.sample_rate;
        samples = std::move(audio.samples);
      } else {
        samples.assign(request.pcm16k.begin(), request.pcm16k.end());
      }
      for (const float sample : samples)
        if (!std::isfinite(sample))
          return Fail(error, "audio samples must be finite");
      if (rate != kRate)
        samples = core::Resample(samples, rate, kRate);
      result->timings.decode_audio_ms = Since(decode);

      // Whole uploads take the widest lookahead, the most accurate. Text
      // that streams arrives pass by pass, in short passes; otherwise full
      // passes, the rest split evenly at the end, are fastest.
      const bool streamed = static_cast<bool>(request.on_text);
      Stream stream(*model_, model_->max_lookahead(), *prompt,
                    streamed ? kShortPassFrames : kMaxPassFrames);
      stream.SetCancellation([&] { return is_cancelled && is_cancelled(); });
      if (streamed)
        stream.SetTextCallback(
            [&] { return request.on_text(stream.text().text()); });
      for (std::size_t at = 0; at < samples.size(); at += kUploadBlock) {
        const std::size_t n = std::min(kUploadBlock, samples.size() - at);
        stream.Push(std::span<const float>(samples.data() + at, n), streamed);
      }
      stream.Finish();

      const TextDecoder& text = stream.text();
      result->text = text.text();
      result->decoded = text.raw();
      result->language = Name(text, config, *prompt);
      result->generated_ids = text.ids();
      result->sample_rate = kRate;
      result->audio_samples = samples.size();
      result->mel_frames = stream.mel_frames();
      result->audio_tokens = stream.encoder_frames();
      result->chunks = stream.chunks();
      result->timings.feature_extraction_ms = stream.feature_ms();
      result->timings.audio_encoder_ms = stream.encoder_ms();
      result->timings.text_decoder_ms = stream.decoder_ms();
      result->timings.total_ms = Since(total);
      return true;
    } catch (const Cancelled&) {
      return Fail(error, "transcription cancelled");
    } catch (const std::exception& failure) {
      return Fail(error, failure.what());
    }
  }

  std::string ModelId() const override {
    return "nemotron-3.5-asr-streaming-0.6b";
  }

  std::unique_ptr<ext::LiveTranscription> CreateLiveTranscription(
      std::uint32_t rate, std::string* error) override {
    try {
      return std::make_unique<Live>(*model_, live_lookahead_, rate);
    } catch (const std::exception& failure) {
      if (error != nullptr)
        *error = failure.what();
      return nullptr;
    }
  }

  bool SupportsLiveTranscription() const override { return true; }

private:
  static bool Fail(std::string* error, const std::string& message) {
    if (error != nullptr)
      *error = message;
    return false;
  }

  std::unique_ptr<Model> model_;
  std::size_t live_lookahead_;
};

}  // namespace

std::unique_ptr<ext::AsrRunner> CreateNemotronAsrRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error) {
  ext::AsrRunnerOptions options;
  options.model_root = model_root;
  options.maximum_context_tokens = maximum_context_tokens;
  return CreateNemotronAsrRunnerWithOptions(options, error);
}

std::unique_ptr<ext::AsrRunner> CreateNemotronAsrRunnerWithOptions(
    const ext::AsrRunnerOptions& options, std::string* error) {
  if (!options.projector_path.empty()) {
    if (error != nullptr)
      *error = "Nemotron ASR does not take --mmproj";
    return nullptr;
  }
  // The option is checked against the configuration before the weights load.
  const auto config = LoadConfig(options.model_root, error);
  if (!config.has_value())
    return nullptr;
  std::size_t live = 0;
  try {
    live = LiveLookahead(*config, options.provider_options);
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return nullptr;
  }
  auto model = Model::Load(options.model_root, error);
  if (model == nullptr)
    return nullptr;
  return std::make_unique<Runner>(std::move(model), live);
}

}  // namespace gufo::models::nemotron_asr
