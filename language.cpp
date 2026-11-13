#include "language.hpp"

#include <array>
#include <cctype>
#include <utility>

namespace gufo::models::nemotron_asr {
namespace {

// English names of the languages NVIDIA lists for the model (transcription
// ready, broad coverage and adaptation ready).
constexpr std::array<std::pair<std::string_view, std::string_view>, 40> kNames =
    {{
        {"en", "English"},   {"es", "Spanish"},    {"fr", "French"},
        {"it", "Italian"},   {"pt", "Portuguese"}, {"nl", "Dutch"},
        {"de", "German"},    {"tr", "Turkish"},    {"ru", "Russian"},
        {"ar", "Arabic"},    {"hi", "Hindi"},      {"ja", "Japanese"},
        {"ko", "Korean"},    {"vi", "Vietnamese"}, {"uk", "Ukrainian"},
        {"pl", "Polish"},    {"sv", "Swedish"},    {"cs", "Czech"},
        {"nb", "Norwegian"}, {"da", "Danish"},     {"bg", "Bulgarian"},
        {"fi", "Finnish"},   {"hr", "Croatian"},   {"sk", "Slovak"},
        {"zh", "Chinese"},   {"hu", "Hungarian"},  {"ro", "Romanian"},
        {"et", "Estonian"},  {"el", "Greek"},      {"lt", "Lithuanian"},
        {"lv", "Latvian"},   {"mt", "Maltese"},    {"sl", "Slovenian"},
        {"he", "Hebrew"},    {"th", "Thai"},       {"nn", "Nynorsk"},
        {"no", "Norwegian"}, {"id", "Indonesian"}, {"ms", "Malay"},
        {"fa", "Persian"},
    }};

std::string Normalize(std::string_view text) {
  std::string out;
  for (const char c : text)
    out += c == '_'
               ? '-'
               : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string_view Base(std::string_view locale) {
  const std::size_t dash = locale.find('-');
  return dash == std::string_view::npos ? locale : locale.substr(0, dash);
}

std::optional<std::size_t> Lookup(const Config& config,
                                  std::string_view normalized) {
  for (const auto& [key, prompt] : config.prompt_dictionary)
    if (Normalize(key) == normalized)
      return prompt;
  if (normalized.find('-') != std::string_view::npos)
    return std::nullopt;
  // A bare code: its own country's locale ("zh" -> "zh-ZH"), else the one
  // prompt all its locales share.
  const std::string own =
      std::string(normalized) + "-" + std::string(normalized);
  for (const auto& [key, prompt] : config.prompt_dictionary)
    if (Normalize(key) == own)
      return prompt;
  std::optional<std::size_t> shared;
  for (const auto& [key, prompt] : config.prompt_dictionary) {
    const std::string lower = Normalize(key);
    if (Base(lower) != normalized)
      continue;
    if (shared.has_value() && *shared != prompt)
      return std::nullopt;
    shared = prompt;
  }
  return shared;
}

}  // namespace

std::optional<std::size_t> ResolveLanguage(
    const Config& config, const std::optional<std::string>& language,
    std::string* error) {
  if (!language.has_value())
    return config.default_prompt;
  const std::string normalized = Normalize(*language);
  if (normalized.empty())
    return config.default_prompt;
  if (auto prompt = Lookup(config, normalized))
    return prompt;
  for (const auto& [code, name] : kNames)
    if (Normalize(name) == normalized)
      if (auto prompt = Lookup(config, code))
        return prompt;
  if (error != nullptr)
    *error = "Nemotron ASR does not support language \"" + *language +
             "\"; use a code such as \"de\", a locale such as \"de-DE\", "
             "an English name or \"auto\"";
  return std::nullopt;
}

std::string LanguageName(std::string_view locale) {
  const std::string base(Base(Normalize(locale)));
  for (const auto& [code, name] : kNames)
    if (code == base)
      return std::string(name);
  return std::string(locale);
}

std::optional<std::string> TagLocale(std::string_view token) {
  if (token.size() < 4 || token.front() != '<' || token.back() != '>')
    return std::nullopt;
  const std::string_view inner = token.substr(1, token.size() - 2);
  const std::size_t dash = inner.find('-');
  if (dash == std::string_view::npos || dash == 0 || dash + 1 == inner.size())
    return std::nullopt;
  for (const char c : inner)
    if (!std::isalpha(static_cast<unsigned char>(c)) && c != '-')
      return std::nullopt;
  return std::string(inner);
}

}  // namespace gufo::models::nemotron_asr
