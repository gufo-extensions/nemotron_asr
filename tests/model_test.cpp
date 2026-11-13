// The GPU model (float16 matrices, float32 sums and activations) against
// tools/reference/golden.py (float32 transformers):
//   encoder        subsampling, encoder and projection outputs of the first
//                  three chunks, the prediction network's start state and
//                  the joint's logits;
//   transcription  token IDs of every reference run (offline transcription of
//                  two clips, the clip's language and auto, four
//                  lookaheads), the audio pushed in pieces;
//   runner         Gufo's AsrRunner: WAV and PCM uploads and live sessions.
#include "model.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "language.hpp"
#include "resampler.hpp"
#include "runner.hpp"
#include "safetensors.hpp"
#include "test_support.hpp"

namespace asr = gufo::models::nemotron_asr;
namespace fs = std::filesystem;
using nemotron_asr_test::Expect;

namespace {

const fs::path kFixtures = GUFO_NEMOTRON_ASR_FIXTURES;

const char* Language(const std::string& clip) {
  return clip == "english" ? "en-US" : "de-DE";
}

void TestEncoder(const asr::Model& model) {
  std::string error;
  const auto golden =
      asr::SafetensorsFile::Open(kFixtures / "golden.safetensors", &error);
  if (golden == nullptr)
    throw std::runtime_error(error);
  const std::size_t bins = model.config().encoder.mel_bins;
  for (const std::string clip : {"english", "german"}) {
    const auto prompt =
        asr::ResolveLanguage(model.config(), std::string(Language(clip)));
    const std::vector<float> features =
        nemotron_asr_test::Floats(*golden, clip + ".features");
    asr::Stream stream(model, 13, *prompt);
    // The prediction network's matrices are float16 (float32 sums).
    nemotron_asr_test::ExpectClose(
        clip + " decoder start", stream.DecoderOutput(),
        nemotron_asr_test::Floats(*golden, clip + ".decoder.start"), 5e-4);
    std::size_t at = 0;
    for (int chunk = 0; chunk < 3; ++chunk) {
      const std::size_t frames = chunk == 0 ? 105 : 112;
      asr::EncoderTrace trace;
      stream.EncodeFeatures(
          std::span<const float>(features.data() + at * bins, frames * bins),
          &trace, false);
      at += frames;
      const std::string name = clip + "." + std::to_string(chunk);
      nemotron_asr_test::ExpectClose(
          name + " subsampled", trace.subsampled,
          nemotron_asr_test::Floats(
              *golden, clip + ".subsampled." + std::to_string(chunk)),
          1e-4);
      nemotron_asr_test::ExpectClose(
          name + " encoder", trace.hidden,
          nemotron_asr_test::Floats(*golden,
                                    clip + ".encoder." + std::to_string(chunk)),
          2e-3);
      nemotron_asr_test::ExpectClose(
          name + " projected", trace.projected,
          nemotron_asr_test::Floats(
              *golden, clip + ".projected." + std::to_string(chunk)),
          2e-3);
      if (chunk == 0)
        nemotron_asr_test::ExpectClose(
            clip + " joint logits", stream.JointLogits(0),
            nemotron_asr_test::Floats(*golden, clip + ".joint.start"), 2e-3);
    }
  }
}

void TestTranscription(const asr::Model& model) {
  const auto golden = nemotron_asr_test::Golden(kFixtures);
  for (const auto& [clip, entry] : golden.find("clips")->members()) {
    const std::vector<float> audio =
        nemotron_asr_test::ReadWav(kFixtures / (clip + ".wav"));
    std::uint32_t seed = 1;
    for (const auto& run : entry.find("runs")->items()) {
      const auto lookahead =
          static_cast<std::size_t>(run.find("lookahead")->as_double());
      const auto prompt =
          static_cast<std::size_t>(run.find("prompt")->as_double());
      asr::Stream stream(model, lookahead, prompt);
      nemotron_asr_test::Splits splits(seed++);
      for (std::size_t at = 0; at < audio.size();) {
        const std::size_t n = std::min(audio.size() - at, splits.Next(9000));
        stream.Push(std::span<const float>(audio.data() + at, n));
        at += n;
      }
      stream.Finish();
      const auto expected = nemotron_asr_test::Ids(*run.find("ids"));
      const std::string label = clip + " " + run.find("language")->str() +
                                " lookahead " + std::to_string(lookahead);
      Expect(stream.chunks() ==
                 static_cast<std::size_t>(run.find("chunks")->as_double()),
             label + ": chunk count");
      Expect(
          stream.encoder_frames() ==
              static_cast<std::size_t>(run.find("encoder_frames")->as_double()),
          label + ": encoder frames");
      Expect(stream.text().ids() == expected, label + ": token IDs");
      Expect(stream.text().text() == run.find("text")->str(),
             label + ": \"" + stream.text().text() + "\"");
      std::cout << label << ": " << stream.text().text() << " (encoder "
                << stream.encoder_ms() << " ms, decoder " << stream.decoder_ms()
                << " ms)\n";
    }
  }
}

void TestRunner(const std::string& root) {
  std::string error;
  auto runner = asr::CreateNemotronAsrRunner(root, 0, &error);
  if (runner == nullptr)
    throw std::runtime_error(error);
  const auto golden = nemotron_asr_test::Golden(kFixtures);
  const auto expected = [&](const std::string& clip,
                            const std::string& language, int lookahead) {
    for (const auto& run :
         golden.find("clips")->find(clip)->find("runs")->items())
      if (run.find("language")->str() == language &&
          static_cast<int>(run.find("lookahead")->as_double()) == lookahead)
        return run.find("text")->str();
    throw std::runtime_error("no reference run");
  };

  const std::string wav = nemotron_asr_test::ReadFile(kFixtures / "german.wav");
  gufo::ext::TranscriptionResult result;
  std::string progress;
  gufo::ext::TranscriptionRequest request;
  request.wav = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(wav.data()), wav.size());
  request.language = std::string("de");
  request.on_text = [&](std::string_view text) {
    Expect(text.starts_with(progress), "progress only appends");
    progress = std::string(text);
    return true;
  };
  Expect(runner->Transcribe(request, {}, &result, &error),
         "WAV upload: " + error);
  Expect(result.text == expected("german", "de-DE", 13),
         "WAV upload text: " + result.text);
  Expect(result.language == "German", "language " + result.language);
  Expect(progress == result.text, "progress ends at the transcript");
  std::cout << "runner: " << result.text << " (" << result.timings.total_ms
            << " ms, " << result.chunks << " chunks)\n";

  const std::vector<float> english =
      nemotron_asr_test::ReadWav(kFixtures / "english.wav");
  gufo::ext::TranscriptionRequest pcm;
  pcm.pcm16k = english;
  Expect(runner->Transcribe(pcm, {}, &result, &error), "PCM upload: " + error);
  Expect(result.text == expected("english", "auto", 13),
         "PCM text: " + result.text);
  Expect(result.language == "English", "detected English: " + result.language);

  gufo::ext::TranscriptionRequest bad;
  bad.pcm16k = english;
  bad.language = std::string("xx");
  Expect(!runner->Transcribe(bad, {}, &result, &error),
         "unknown language fails");

  gufo::ext::TranscriptionRequest cancelled;
  cancelled.pcm16k = english;
  Expect(!runner->Transcribe(
             cancelled, [] { return true; }, &result, &error),
         "cancelled request fails");

  // Live at 16 kHz equals the reference's lookahead-6 auto run.
  Expect(runner->SupportsLiveTranscription(), "live transcription");
  for (const std::uint32_t rate : {16000U, 24000U}) {
    std::vector<float> audio = english;
    if (rate != 16000) {
      asr::Resampler up(16000, rate);
      std::vector<float> out;
      up.Push(english, &out);
      up.Finish(&out);
      audio = std::move(out);
    }
    auto live = runner->CreateLiveTranscription(rate, &error);
    Expect(live != nullptr, "live session: " + error);
    if (!live)
      continue;
    std::string deltas;
    const auto collect = [&](std::string_view delta) {
      deltas += delta;
      return true;
    };
    nemotron_asr_test::Splits splits(rate);
    for (std::size_t at = 0; at < audio.size();) {
      const std::size_t n = std::min(audio.size() - at, splits.Next(4800));
      Expect(live->Push(std::span<const float>(audio.data() + at, n), collect,
                        &error),
             "live push: " + error);
      at += n;
    }
    Expect(live->Finish(collect, &error), "live finish: " + error);
    Expect(deltas == live->text(), "deltas add up to the transcript");
    std::cout << "live " << rate << " Hz: " << live->text() << "\n";
    if (rate == 16000)
      Expect(live->text() == expected("english", "auto", 6), "live text");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* root = std::getenv("GUFO_NEMOTRON_ASR_MODEL");
  if (root == nullptr || *root == '\0') {
    std::cout << "GUFO_NEMOTRON_ASR_MODEL is not set; skipping\n";
    return nemotron_asr_test::kSkip;
  }
  const std::string mode = argc > 1 ? argv[1] : "encoder";
  try {
    if (mode == "runner") {
      TestRunner(root);
    } else {
      std::string error;
      const auto model = asr::Model::Load(root, &error);
      if (model == nullptr)
        throw std::runtime_error(error);
      if (mode == "encoder")
        TestEncoder(*model);
      else
        TestTranscription(*model);
    }
  } catch (const std::exception& failure) {
    std::cerr << "FAIL: " << failure.what() << "\n";
    return 1;
  }
  if (nemotron_asr_test::failures != 0) {
    std::cerr << nemotron_asr_test::failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
