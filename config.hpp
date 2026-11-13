#ifndef GUFO_MODELS_NEMOTRON_ASR_CONFIG_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_CONFIG_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::models::nemotron_asr {

/// The FastConformer encoder of `config.json`'s `encoder_config`.
struct EncoderConfig {
  std::size_t hidden = 0;
  std::size_t layers = 0;
  std::size_t heads = 0;
  std::size_t intermediate = 0;
  std::size_t conv_kernel = 0;
  std::size_t mel_bins = 0;
  std::size_t subsampling_channels = 0;
  std::size_t subsampling_kernel = 0;
  std::size_t subsampling_stride = 0;
  std::size_t subsampling_factor = 0;
  /// Left attention context plus one, in encoder frames.
  std::size_t sliding_window = 0;
  std::size_t max_positions = 0;
  /// Right attention contexts the checkpoint was trained with, in encoder
  /// frames; the first is the default.
  std::vector<std::size_t> lookaheads;
  std::size_t default_lookahead = 0;

  [[nodiscard]] std::size_t head_dim() const { return hidden / heads; }
  /// Attention frames left of a chunk.
  [[nodiscard]] std::size_t left_context() const { return sliding_window - 1; }
};

/// The log-mel front end of `processor_config.json`.
struct FeatureConfig {
  std::size_t sample_rate = 16000;
  std::size_t n_fft = 512;
  std::size_t window = 400;
  std::size_t hop = 160;
  double preemphasis = 0.97;
};

/// Nemotron 3.5 ASR (`Nemotron3_5AsrForRNNT`): the encoder, the language
/// prompt and the RNN-T prediction and joint networks.
struct Config {
  EncoderConfig encoder;
  FeatureConfig features;
  std::size_t decoder_hidden = 0;
  std::size_t decoder_layers = 0;
  std::size_t vocab = 0;
  std::size_t blank = 0;
  std::size_t prompts = 0;
  std::size_t prompt_hidden = 0;
  std::size_t default_prompt = 0;
  std::size_t max_symbols = 0;
  /// Language keys (locales, bare codes, "auto") and their prompt indices.
  std::vector<std::pair<std::string, std::size_t>> prompt_dictionary;
};

/// Parses `config.json` and `processor_config.json` texts.
[[nodiscard]] std::optional<Config> ParseConfig(std::string_view config_json,
                                                std::string_view processor_json,
                                                std::string* error = nullptr);
/// Reads both files from a model directory.
[[nodiscard]] std::optional<Config> LoadConfig(std::string_view model_root,
                                               std::string* error = nullptr);
/// Whether the kernels support these shapes.
[[nodiscard]] bool IsSupported(const Config& config,
                               std::string* error = nullptr);

/// Mel frames of the first streaming chunk and of every later one.
[[nodiscard]] std::size_t FirstChunkFrames(const Config& config,
                                           std::size_t lookahead);
[[nodiscard]] std::size_t ChunkFrames(const Config& config,
                                      std::size_t lookahead);

}  // namespace gufo::models::nemotron_asr

#endif  // GUFO_MODELS_NEMOTRON_ASR_CONFIG_HPP_
