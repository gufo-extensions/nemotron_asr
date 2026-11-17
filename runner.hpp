#ifndef GUFO_MODELS_NEMOTRON_ASR_RUNNER_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_RUNNER_HPP_

#include <cstddef>
#include <gufo/ext/asr.hpp>
#include <memory>
#include <string>

namespace gufo::models::nemotron_asr {

/// Gufo's ASR runner for Nemotron 3.5 ASR (`gufo serve asr`).
[[nodiscard]] std::unique_ptr<ext::AsrRunner> CreateNemotronAsrRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);

/// The same runner with the options it declares: `--live-lookahead` sets the
/// lookahead of live sessions.
[[nodiscard]] std::unique_ptr<ext::AsrRunner>
CreateNemotronAsrRunnerWithOptions(const ext::AsrRunnerOptions& options,
                                   std::string* error);

}  // namespace gufo::models::nemotron_asr

#endif  // GUFO_MODELS_NEMOTRON_ASR_RUNNER_HPP_
