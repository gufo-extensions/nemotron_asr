// Token IDs to transcripts against the reference's decoding of every run in
// tests/fixtures/golden.json (needs GUFO_NEMOTRON_ASR_MODEL's tokenizer.json).
#include "tokenizer.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "test_support.hpp"

namespace asr = gufo::models::nemotron_asr;
namespace fs = std::filesystem;
using nemotron_asr_test::Expect;

int main() {
  const char* root = std::getenv("GUFO_NEMOTRON_ASR_MODEL");
  if (root == nullptr || *root == '\0') {
    std::cout << "GUFO_NEMOTRON_ASR_MODEL is not set; skipping\n";
    return nemotron_asr_test::kSkip;
  }
  try {
    std::string error;
    const auto tokenizer =
        asr::Tokenizer::Load(fs::path(root) / "tokenizer.json", &error);
    Expect(tokenizer != nullptr, "tokenizer loads: " + error);
    if (tokenizer == nullptr)
      return 1;
    Expect(tokenizer->size() == 13089, "13089 tokens");
    Expect(tokenizer->IsSpecial(13087), "<pad> (blank) is special");
    Expect(tokenizer->Piece(518) == "<de-DE>", "language tag piece");
    const auto golden = nemotron_asr_test::Golden(GUFO_NEMOTRON_ASR_FIXTURES);
    std::size_t runs = 0;
    for (const auto& [clip, entry] : golden.find("clips")->members()) {
      for (const auto& run : entry.find("runs")->items()) {
        asr::TextDecoder text(*tokenizer);
        for (const std::uint32_t id : nemotron_asr_test::Ids(*run.find("ids")))
          text.Push(id);
        const std::string expected = run.find("text")->str();
        Expect(text.text() == expected,
               clip + ": \"" + text.text() + "\" != \"" + expected + "\"");
        const std::string raw = run.find("raw")->str();
        const std::size_t tag = raw.rfind('<');
        if (tag != std::string::npos) {
          const std::string locale =
              raw.substr(tag + 1, raw.find('>', tag) - tag - 1);
          Expect(text.language() == std::optional<std::string>(locale),
                 clip + ": language tag " + locale);
        }
        ++runs;
      }
    }
    Expect(runs == 16, "16 reference runs");
  } catch (const std::exception& failure) {
    std::cerr << "FAIL: " << failure.what() << "\n";
    return 1;
  }
  if (nemotron_asr_test::failures != 0)
    return 1;
  std::cout << "ok\n";
  return 0;
}
