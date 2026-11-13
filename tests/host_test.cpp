// Configuration, language prompts, transcript cleaning and the resampler;
// with "model", the probe and configuration of GUFO_NEMOTRON_ASR_MODEL.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include "config.hpp"
#include "language.hpp"
#include "probe.hpp"
#include "resampler.hpp"
#include "test_support.hpp"
#include "tokenizer.hpp"

namespace asr = gufo::models::nemotron_asr;
using nemotron_asr_test::Expect;
using nemotron_asr_test::failures;

namespace {

constexpr const char* kConfig = R"({
  "architectures": ["Nemotron3_5AsrForRNNT"],
  "blank_token_id": 13087, "decoder_hidden_size": 640, "durations": [],
  "encoder_config": {
    "attention_bias": false, "conv_kernel_size": 9, "convolution_bias": false,
    "default_num_lookahead_tokens": 3, "hidden_act": "silu", "hidden_size": 1024,
    "intermediate_size": 4096, "max_position_embeddings": 5000,
    "model_type": "nemotron_asr_streaming_encoder", "num_attention_heads": 8,
    "num_hidden_layers": 24, "num_key_value_heads": 8, "num_mel_bins": 128,
    "scale_input": false, "sliding_window": 57, "subsampling_conv_channels": 256,
    "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
    "subsampling_factor": 8, "supported_num_lookahead_tokens": [3, 0, 6, 13]
  },
  "hidden_act": "relu", "max_symbols_per_step": 10, "model_type": "nemotron3_5_asr",
  "num_decoder_layers": 2, "num_prompts": 128, "prompt_intermediate_size": 2048,
  "vocab_size": 13088, "default_prompt_id": 101
})";

constexpr const char* kProcessor = R"({
  "default_num_lookahead_tokens": 3,
  "feature_extractor": {"feature_size": 128, "hop_length": 160, "n_fft": 512,
    "preemphasis": 0.97, "sampling_rate": 16000, "win_length": 400},
  "num_prompts": 128,
  "prompt_dictionary": {"auto": 101, "de": 9, "de-DE": 9, "en": 0, "en-US": 0,
    "en-GB": 1, "ja-JA": 10, "ja-JP": 10, "zh-CN": 4, "zh-ZH": 4, "zh-TW": 5,
    "nb": 103, "nb-NO": 103, "no": 27, "no-NO": 27}
})";

void TestConfig() {
  std::string error;
  const auto config = asr::ParseConfig(kConfig, kProcessor, &error);
  Expect(config.has_value(), "the sample configuration parses: " + error);
  if (!config)
    return;
  Expect(asr::IsSupported(*config, &error),
         "the sample is supported: " + error);
  Expect(config->encoder.head_dim() == 128, "head dimension");
  Expect(config->encoder.left_context() == 56, "left context");
  Expect(config->encoder.default_lookahead == 3, "default lookahead");
  Expect(asr::FirstChunkFrames(*config, 13) == 105, "first chunk frames");
  Expect(asr::ChunkFrames(*config, 13) == 112, "chunk frames");
  Expect(asr::FirstChunkFrames(*config, 0) == 1, "first chunk, lookahead 0");
  Expect(!asr::ParseConfig("{}", kProcessor).has_value(), "empty config fails");
  Expect(!asr::ParseConfig(kConfig, "{", &error).has_value(), "bad JSON fails");

  const auto prompt = [&](const char* language) -> long {
    const auto p = asr::ResolveLanguage(*config, std::string(language));
    return p ? static_cast<long>(*p) : -1;
  };
  Expect(prompt("de") == 9, "de");
  Expect(prompt("DE_de") == 9, "DE_de");
  Expect(prompt("German") == 9, "German");
  Expect(prompt("german") == 9, "german");
  Expect(prompt("ja") == 10, "ja (shared prompt)");
  Expect(prompt("zh") == 4, "zh (its country's locale)");
  Expect(prompt("ZH-tw") == 5, "zh-TW");
  Expect(prompt("Norwegian") == 103, "Norwegian");
  Expect(prompt("auto") == 101, "auto");
  Expect(prompt("") == 101, "empty");
  Expect(prompt("xx") == -1, "unknown language");
  Expect(asr::ResolveLanguage(*config, std::nullopt) == 101u, "no language");
  Expect(asr::LanguageName("de-DE") == "German", "German name");
  Expect(asr::LanguageName("xx-YY") == "xx-YY", "unnamed locale");
  Expect(asr::TagLocale("<de-DE>") == std::optional<std::string>("de-DE"),
         "tag");
  Expect(!asr::TagLocale("<unk>").has_value(), "unk is no tag");
  Expect(!asr::TagLocale("<pad>").has_value(), "pad is no tag");
}

void TestText() {
  Expect(asr::CleanText(" Heute ist.  Wir ") == "Heute ist. Wir", "clean");
  Expect(asr::CleanText("") == "", "clean empty");
  Expect(asr::CleanText(" \n ") == "", "clean spaces");
  // Appending to the raw text only appends to its transcript.
  const std::string raw = " A b.  ";
  const std::string longer = raw + "C";
  Expect(asr::CleanText(longer).starts_with(asr::CleanText(raw)), "prefix");
}

std::vector<float> Resample(std::uint32_t from, std::uint32_t to,
                            const std::vector<float>& in, std::size_t split) {
  asr::Resampler resampler(from, to);
  std::vector<float> out;
  nemotron_asr_test::Splits splits(7);
  for (std::size_t at = 0; at < in.size();) {
    const std::size_t n =
        std::min(in.size() - at, split == 0 ? in.size() : splits.Next(split));
    resampler.Push(std::span<const float>(in.data() + at, n), &out);
    at += n;
  }
  resampler.Finish(&out);
  return out;
}

void TestResampler() {
  std::vector<float> tone(24000);
  for (std::size_t i = 0; i < tone.size(); ++i)
    tone[i] = 0.5F * static_cast<float>(std::sin(2.0 * std::numbers::pi *
                                                 1000.0 * i / 24000.0));
  const auto whole = Resample(24000, 16000, tone, 0);
  Expect(whole.size() == 16000, "24 kHz -> 16 kHz length");
  Expect(Resample(24000, 16000, tone, 777) == whole, "split invariance");
  Expect(Resample(24000, 16000, tone, 3) == whole, "tiny splits");
  // A 1 kHz tone keeps its amplitude and phase away from the edges.
  double error = 0.0;
  for (std::size_t i = 2000; i < 14000; ++i)
    error = std::max(error,
                     std::abs(whole[i] - 0.5 * std::sin(2.0 * std::numbers::pi *
                                                        1000.0 * i / 16000.0)));
  std::cout << "resampled tone max error " << error << "\n";
  Expect(error < 1e-3, "tone after resampling");
  const auto same = Resample(16000, 16000, tone, 100);
  Expect(same == tone, "passthrough");
  const auto odd =
      Resample(44100, 16000, std::vector<float>(44100, 0.25F), 1000);
  Expect(odd.size() == 16000, "44.1 kHz -> 16 kHz length");
  Expect(std::abs(odd[8000] - 0.25F) < 1e-3, "DC gain");
}

int TestModel() {
  const char* root = std::getenv("GUFO_NEMOTRON_ASR_MODEL");
  if (root == nullptr || *root == '\0') {
    std::cout << "GUFO_NEMOTRON_ASR_MODEL is not set; skipping\n";
    return nemotron_asr_test::kSkip;
  }
  std::string error;
  const auto config = asr::LoadConfig(root, &error);
  Expect(config.has_value(), "the model configuration loads: " + error);
  if (config) {
    Expect(asr::IsSupported(*config, &error),
           "the model is supported: " + error);
    Expect(config->encoder.default_lookahead == 3, "default lookahead 3");
    Expect(config->encoder.lookaheads.size() == 4, "four lookaheads");
    Expect(asr::ResolveLanguage(*config, std::string("de")) == 9u, "de prompt");
    Expect(asr::ResolveLanguage(*config, std::string("zh")) == 4u,
           "zh takes zh-ZH, which shares zh-CN's prompt");
  }
  Expect(asr::ProbeNemotronAsr(root), "probe accepts the model");
  Expect(asr::ResolveNemotronAsrArtifacts(root, &error),
         "artifacts resolve: " + error);
  const auto other = std::filesystem::temp_directory_path() /
                     ("nemotron_asr_probe_" + std::to_string(::getpid()));
  std::filesystem::create_directories(other);
  std::ofstream(other / "config.json") << R"({"model_type": "parakeet"})";
  std::ofstream(other / "processor_config.json") << "{}";
  Expect(!asr::ProbeNemotronAsr(other.string()), "probe rejects other models");
  std::filesystem::remove_all(other);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "samples";
  try {
    if (mode == "model") {
      if (TestModel() == nemotron_asr_test::kSkip)
        return nemotron_asr_test::kSkip;
    } else {
      TestConfig();
      TestText();
      TestResampler();
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
