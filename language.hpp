#ifndef GUFO_MODELS_NEMOTRON_ASR_LANGUAGE_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_LANGUAGE_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "config.hpp"

namespace gufo::models::nemotron_asr {

/// The language prompt for a request's `language`: absent, empty or "auto"
/// detects the language; otherwise a key of the prompt dictionary ("de",
/// "de-DE"; case-insensitive, "_" for "-"), a bare code with its country's
/// locale ("zh" for "zh-ZH") or whose locales share one prompt ("ja" for
/// "ja-JP"), or an English language name ("German").
/// Null with a message for anything else.
[[nodiscard]] std::optional<std::size_t> ResolveLanguage(
    const Config& config, const std::optional<std::string>& language,
    std::string* error = nullptr);

/// The English name of a locale or code ("de-DE" -> "German"); the input
/// itself for languages without one here.
[[nodiscard]] std::string LanguageName(std::string_view locale);

/// The locale of a language tag token ("<de-DE>" -> "de-DE"), if it is one.
[[nodiscard]] std::optional<std::string> TagLocale(std::string_view token);

}  // namespace gufo::models::nemotron_asr

#endif  // GUFO_MODELS_NEMOTRON_ASR_LANGUAGE_HPP_
