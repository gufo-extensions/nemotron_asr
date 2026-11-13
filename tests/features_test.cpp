// The log-mel front end against tools/reference/golden.py: each fixture
// clip, pushed in pieces, then ended as a centered pass over the whole clip
// ends (zeros after the pre-emphasized signal, samples / 160 frames).
#include "features.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "safetensors.hpp"
#include "test_support.hpp"

namespace asr = gufo::models::nemotron_asr;
namespace fs = std::filesystem;
using nemotron_asr_test::Expect;

int main() {
  try {
    const fs::path fixtures = GUFO_NEMOTRON_ASR_FIXTURES;
    std::string error;
    const auto golden =
        asr::SafetensorsFile::Open(fixtures / "golden.safetensors", &error);
    Expect(golden != nullptr, "golden opens: " + error);
    if (golden == nullptr)
      return 1;
    asr::FeatureConfig config;
    constexpr std::size_t kBins = 128;
    for (const char* clip : {"english", "german"}) {
      std::uint32_t rate = 0;
      const std::vector<float> audio = nemotron_asr_test::ReadWav(
          fixtures / (std::string(clip) + ".wav"), &rate);
      Expect(rate == 16000, std::string(clip) + " is 16 kHz");
      const std::vector<float> expected =
          nemotron_asr_test::Floats(*golden, std::string(clip) + ".features");
      for (const std::size_t split :
           {std::size_t{0}, std::size_t{1}, std::size_t{997}}) {
        asr::FeatureStream stream(config, kBins);
        std::vector<float> frames;
        nemotron_asr_test::Splits splits(static_cast<std::uint32_t>(split + 3));
        for (std::size_t at = 0; at < audio.size();) {
          const std::size_t n =
              split == 0 ? audio.size() - at
                         : std::min(audio.size() - at,
                                    splits.Next(split == 1 ? 7 : split));
          stream.Push(std::span<const float>(audio.data() + at, n), &frames);
          at += n;
        }
        const std::size_t total = stream.FramesFor(audio.size());
        stream.Flush(total, &frames);
        Expect(stream.frames() == total, "frame count");
        nemotron_asr_test::ExpectClose(
            std::string(clip) + " features, split " + std::to_string(split),
            frames, expected, 1e-4);
      }
    }
  } catch (const std::exception& failure) {
    std::cerr << "FAIL: " << failure.what() << "\n";
    return 1;
  }
  if (nemotron_asr_test::failures != 0)
    return 1;
  std::cout << "ok\n";
  return 0;
}
