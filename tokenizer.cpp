#include "tokenizer.hpp"

#include <cmath>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <stdexcept>

#include "language.hpp"

namespace gufo::models::nemotron_asr {
namespace {

namespace json = gufo::json;

constexpr std::string_view kMetaspace = "\xE2\x96\x81";  // U+2581

std::string Spaces(std::string_view piece) {
  std::string out;
  for (std::size_t at = 0; at < piece.size();) {
    if (piece.compare(at, kMetaspace.size(), kMetaspace) == 0) {
      out += ' ';
      at += kMetaspace.size();
    } else {
      out += piece[at++];
    }
  }
  return out;
}

bool Whitespace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
         c == '\f';
}

std::uint32_t Id(const json::Value& value) {
  if (!value.is_number() || value.as_double() < 0 ||
      value.as_double() != std::floor(value.as_double()) ||
      value.as_double() > 1e7)
    throw std::runtime_error("tokenizer.json has an invalid token ID");
  return static_cast<std::uint32_t>(value.as_double());
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::Load(const std::filesystem::path& path,
                                           std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error != nullptr)
      *error = "cannot read " + path.string();
    return nullptr;
  }
  const std::string text{std::istreambuf_iterator<char>(in),
                         std::istreambuf_iterator<char>()};
  return Parse(text, error);
}

std::unique_ptr<Tokenizer> Tokenizer::Parse(std::string_view text,
                                            std::string* error) {
  try {
    const json::Value root = json::parse(std::string(text));
    const json::Value* model = root.find("model");
    if (model == nullptr || !model->is_object())
      throw std::runtime_error("tokenizer.json lacks a model");
    const json::Value* type = model->find("type");
    if (type == nullptr || !type->is_string() || type->str() != "BPE")
      throw std::runtime_error("tokenizer.json is not a BPE model");
    const json::Value* decoder = root.find("decoder");
    const json::Value* decoder_type =
        decoder == nullptr ? nullptr : decoder->find("type");
    if (decoder_type == nullptr || !decoder_type->is_string() ||
        decoder_type->str() != "Metaspace")
      throw std::runtime_error("tokenizer.json lacks a Metaspace decoder");
    const json::Value* vocab = model->find("vocab");
    if (vocab == nullptr || !vocab->is_object())
      throw std::runtime_error("tokenizer.json lacks a vocabulary");

    auto tokenizer = std::unique_ptr<Tokenizer>(new Tokenizer());
    const auto put = [&](const std::string& piece, std::uint32_t id,
                         bool special) {
      if (id >= tokenizer->pieces_.size()) {
        tokenizer->pieces_.resize(id + 1);
        tokenizer->texts_.resize(id + 1);
        tokenizer->special_.resize(id + 1, true);
      }
      tokenizer->pieces_[id] = piece;
      tokenizer->texts_[id] = special ? std::string() : Spaces(piece);
      tokenizer->special_[id] = special;
    };
    for (const auto& [piece, id] : vocab->members())
      put(piece, Id(id), false);
    if (const json::Value* added = root.find("added_tokens");
        added != nullptr && added->is_array()) {
      for (const json::Value& token : added->items()) {
        const json::Value* content = token.find("content");
        const json::Value* id = token.find("id");
        if (content == nullptr || !content->is_string() || id == nullptr)
          throw std::runtime_error(
              "tokenizer.json has a malformed added token");
        const json::Value* special = token.find("special");
        put(content->str(), Id(*id),
            special != nullptr && special->as_bool(false));
      }
    }
    if (tokenizer->pieces_.empty())
      throw std::runtime_error("tokenizer.json has no tokens");
    return tokenizer;
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return nullptr;
  }
}

std::string_view Tokenizer::Text(std::uint32_t id) const {
  return id < texts_.size() ? std::string_view(texts_[id]) : std::string_view();
}

bool Tokenizer::IsSpecial(std::uint32_t id) const {
  return id >= special_.size() || special_[id];
}

std::string_view Tokenizer::Piece(std::uint32_t id) const {
  return id < pieces_.size() ? std::string_view(pieces_[id])
                             : std::string_view();
}

std::string CleanText(std::string_view raw) {
  std::string out;
  bool space = false;
  for (const char c : raw) {
    if (Whitespace(c)) {
      space = !out.empty();
      continue;
    }
    if (space)
      out += ' ';
    space = false;
    out += c;
  }
  return out;
}

void TextDecoder::Push(std::uint32_t id) {
  ids_.push_back(id);
  if (tokenizer_->IsSpecial(id)) {
    if (auto locale = TagLocale(tokenizer_->Piece(id)))
      language_ = std::move(locale);
    return;
  }
  raw_ += tokenizer_->Text(id);
}

void TextDecoder::Reset() {
  ids_.clear();
  raw_.clear();
  language_.reset();
}

}  // namespace gufo::models::nemotron_asr
