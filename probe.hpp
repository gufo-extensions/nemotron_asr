#ifndef GUFO_MODELS_NEMOTRON_ASR_PROBE_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_PROBE_HPP_

#include <string>
#include <string_view>

namespace gufo::models::nemotron_asr {

/// `model_root` is a Hugging Face directory of Nemotron 3.5 ASR: config.json
/// with model_type "nemotron3_5_asr" (architecture Nemotron3_5AsrForRNNT)
/// and processor_config.json.
[[nodiscard]] bool ProbeNemotronAsr(std::string_view model_root);
/// Checks that the configuration is supported and the weight and tokenizer
/// files exist.
[[nodiscard]] bool ResolveNemotronAsrArtifacts(std::string_view model_root,
                                               std::string* error = nullptr);

}  // namespace gufo::models::nemotron_asr

#endif  // GUFO_MODELS_NEMOTRON_ASR_PROBE_HPP_
