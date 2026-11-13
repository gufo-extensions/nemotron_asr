#ifndef GUFO_MODELS_NEMOTRON_ASR_TOKENIZER_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_TOKENIZER_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::nemotron_asr {

/// The decoding half of `tokenizer.json` (SentencePiece BPE with a Metaspace
/// decoder): token IDs to text. Transcription never encodes text.
class Tokenizer {
public:
  [[nodiscard]] static std::unique_ptr<Tokenizer> Load(
      const std::filesystem::path& path, std::string* error = nullptr);
  [[nodiscard]] static std::unique_ptr<Tokenizer> Parse(
      std::string_view json, std::string* error = nullptr);

  [[nodiscard]] std::size_t size() const { return pieces_.size(); }
  /// The token's text with "▁" as a space; empty for special tokens.
  [[nodiscard]] std::string_view Text(std::uint32_t id) const;
  [[nodiscard]] bool IsSpecial(std::uint32_t id) const;
  /// The token as stored ("<de-DE>", "▁Heute").
  [[nodiscard]] std::string_view Piece(std::uint32_t id) const;

private:
  Tokenizer() = default;

  std::vector<std::string> pieces_;
  std::vector<std::string> texts_;
  std::vector<bool> special_;
};

/// The transcript of a raw decoding: runs of whitespace become one space,
/// and none leads or trails. Appending to the raw text only appends to (or
/// completes the trailing space of) its transcript.
[[nodiscard]] std::string CleanText(std::string_view raw);

/// Accumulates emitted tokens into the transcript and remembers the last
/// language tag the model emitted.
class TextDecoder {
public:
  explicit TextDecoder(const Tokenizer& tokenizer) : tokenizer_(&tokenizer) {}

  void Push(std::uint32_t id);
  void Reset();

  [[nodiscard]] const std::vector<std::uint32_t>& ids() const { return ids_; }
  [[nodiscard]] const std::string& raw() const { return raw_; }
  [[nodiscard]] std::string text() const { return CleanText(raw_); }
  /// The locale of the last language tag ("de-DE"), if any.
  [[nodiscard]] const std::optional<std::string>& language() const {
    return language_;
  }

private:
  const Tokenizer* tokenizer_;
  std::vector<std::uint32_t> ids_;
  std::string raw_;
  std::optional<std::string> language_;
};

}  // namespace gufo::models::nemotron_asr

#endif  // GUFO_MODELS_NEMOTRON_ASR_TOKENIZER_HPP_
