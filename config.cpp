#include "config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <stdexcept>
#include <string>

namespace gufo::models::nemotron_asr {
namespace {

namespace json = gufo::json;

class ConfigError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Typed access to one JSON object; failures name the key.
class Fields {
public:
  Fields(const json::Value& object, std::string where)
      : object_(object), where_(std::move(where)) {
    if (!object_.is_object())
      throw ConfigError(where_ + " is not a JSON object");
  }

  [[nodiscard]] const json::Value* Find(const std::string& key) const {
    const json::Value* value = object_.find(key);
    return value == nullptr || value->is_null() ? nullptr : value;
  }

  [[nodiscard]] const json::Value& Get(const std::string& key) const {
    const json::Value* value = Find(key);
    if (value == nullptr)
      throw ConfigError(where_ + " lacks \"" + key + "\"");
    return *value;
  }

  [[nodiscard]] std::size_t Index(const json::Value& value,
                                  const std::string& what) const {
    if (!value.is_number() || value.as_double() < 0 ||
        value.as_double() != std::floor(value.as_double()))
      throw ConfigError(where_ + "." + what + " is not a natural number");
    return static_cast<std::size_t>(value.as_double());
  }

  [[nodiscard]] std::size_t Size(const std::string& key) const {
    const std::size_t size = Index(Get(key), key);
    if (size == 0)
      throw ConfigError(where_ + "." + key + " is not a positive integer");
    return size;
  }

  [[nodiscard]] std::size_t Natural(const std::string& key,
                                    std::size_t fallback) const {
    const json::Value* value = Find(key);
    return value == nullptr ? fallback : Index(*value, key);
  }

  [[nodiscard]] double Number(const std::string& key, double fallback) const {
    const json::Value* value = Find(key);
    if (value == nullptr)
      return fallback;
    if (!value->is_number() || !std::isfinite(value->as_double()))
      throw ConfigError(where_ + "." + key + " is not a number");
    return value->as_double();
  }

  [[nodiscard]] std::string String(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_string())
      throw ConfigError(where_ + "." + key + " is not a string");
    return value.str();
  }

  [[nodiscard]] bool Bool(const std::string& key, bool fallback) const {
    const json::Value* value = Find(key);
    if (value == nullptr)
      return fallback;
    if (!value->is_bool())
      throw ConfigError(where_ + "." + key + " is not a boolean");
    return value->as_bool();
  }

  [[nodiscard]] std::vector<std::size_t> Sizes(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_array() || value.items().empty())
      throw ConfigError(where_ + "." + key + " is not a non-empty array");
    std::vector<std::size_t> out;
    for (const json::Value& item : value.items())
      out.push_back(Index(item, key));
    return out;
  }

  [[nodiscard]] const json::Value& object() const { return object_; }
  [[nodiscard]] const std::string& where() const { return where_; }

private:
  const json::Value& object_;
  std::string where_;
};

json::Value ParseJson(std::string_view text, const char* what) {
  try {
    return json::parse(std::string(text));
  } catch (const std::exception& error) {
    throw ConfigError(std::string(what) +
                      " is not valid JSON: " + error.what());
  }
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw ConfigError("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void Fail(std::string* error, const std::string& message) {
  if (error != nullptr)
    *error = message;
}

}  // namespace

std::optional<Config> ParseConfig(std::string_view config_json,
                                  std::string_view processor_json,
                                  std::string* error) {
  try {
    const json::Value root = ParseJson(config_json, "config.json");
    const Fields fields(root, "config.json");
    if (fields.String("model_type") != "nemotron3_5_asr")
      throw ConfigError("config.json model_type is not nemotron3_5_asr");
    Config config;
    const Fields encoder(fields.Get("encoder_config"),
                         "config.json.encoder_config");
    EncoderConfig& e = config.encoder;
    e.hidden = encoder.Size("hidden_size");
    e.layers = encoder.Size("num_hidden_layers");
    e.heads = encoder.Size("num_attention_heads");
    if (encoder.Natural("num_key_value_heads", e.heads) != e.heads)
      throw ConfigError("grouped key/value heads are not supported");
    e.intermediate = encoder.Size("intermediate_size");
    e.conv_kernel = encoder.Size("conv_kernel_size");
    e.mel_bins = encoder.Size("num_mel_bins");
    e.subsampling_channels = encoder.Size("subsampling_conv_channels");
    e.subsampling_kernel = encoder.Size("subsampling_conv_kernel_size");
    e.subsampling_stride = encoder.Size("subsampling_conv_stride");
    e.subsampling_factor = encoder.Size("subsampling_factor");
    e.sliding_window = encoder.Size("sliding_window");
    e.max_positions = encoder.Size("max_position_embeddings");
    e.lookaheads = encoder.Sizes("supported_num_lookahead_tokens");
    e.default_lookahead =
        encoder.Natural("default_num_lookahead_tokens", e.lookaheads.front());
    if (encoder.Bool("attention_bias", false) ||
        encoder.Bool("convolution_bias", false))
      throw ConfigError("attention and convolution biases are not supported");
    if (encoder.Bool("scale_input", false))
      throw ConfigError("scaled encoder input is not supported");
    if (encoder.String("hidden_act") != "silu")
      throw ConfigError("the encoder activation is not silu");

    config.decoder_hidden = fields.Size("decoder_hidden_size");
    config.decoder_layers = fields.Size("num_decoder_layers");
    config.vocab = fields.Size("vocab_size");
    config.blank = fields.Natural("blank_token_id", config.vocab);
    config.prompts = fields.Size("num_prompts");
    config.prompt_hidden = fields.Size("prompt_intermediate_size");
    config.default_prompt = fields.Natural("default_prompt_id", 0);
    config.max_symbols = fields.Size("max_symbols_per_step");
    if (fields.String("hidden_act") != "relu")
      throw ConfigError("the joint activation is not relu");
    const json::Value* durations = fields.Find("durations");
    if (durations != nullptr &&
        (!durations->is_array() || !durations->items().empty()))
      throw ConfigError("TDT durations are not supported");

    const json::Value processor =
        ParseJson(processor_json, "processor_config.json");
    const Fields p(processor, "processor_config.json");
    const Fields features(p.Get("feature_extractor"),
                          "processor_config.json.feature_extractor");
    config.features.sample_rate = features.Size("sampling_rate");
    config.features.n_fft = features.Size("n_fft");
    config.features.window = features.Size("win_length");
    config.features.hop = features.Size("hop_length");
    // transformers' feature extractor defaults to 0.97.
    config.features.preemphasis = features.Number("preemphasis", 0.97);
    if (features.Size("feature_size") != e.mel_bins)
      throw ConfigError("the feature size differs from num_mel_bins");
    if (p.Find("default_num_lookahead_tokens") != nullptr)
      e.default_lookahead = p.Natural("default_num_lookahead_tokens", 0);
    const Fields dictionary(p.Get("prompt_dictionary"),
                            "processor_config.json.prompt_dictionary");
    for (const auto& [key, value] : dictionary.object().members())
      config.prompt_dictionary.emplace_back(key, dictionary.Index(value, key));
    return config;
  } catch (const std::exception& failure) {
    Fail(error, failure.what());
    return std::nullopt;
  }
}

std::optional<Config> LoadConfig(std::string_view model_root,
                                 std::string* error) {
  try {
    const std::filesystem::path root(model_root);
    return ParseConfig(ReadFile(root / "config.json"),
                       ReadFile(root / "processor_config.json"), error);
  } catch (const std::exception& failure) {
    Fail(error, failure.what());
    return std::nullopt;
  }
}

bool IsSupported(const Config& config, std::string* error) {
  const EncoderConfig& e = config.encoder;
  const auto fail = [error](const std::string& message) {
    Fail(error, "unsupported Nemotron ASR checkpoint: " + message);
    return false;
  };
  if (e.hidden % e.heads != 0 || e.head_dim() != 128)
    return fail("attention heads must have 128 dimensions");
  if (e.hidden % 256 != 0)
    return fail("the hidden size must be a multiple of 256");
  if (e.intermediate % 32 != 0)
    return fail("the feed-forward size must be a multiple of 32");
  if (e.subsampling_kernel != 3 || e.subsampling_stride != 2 ||
      e.subsampling_factor != 8)
    return fail("subsampling must be three stride-2 convolutions of size 3");
  if (e.subsampling_channels == 0 || e.subsampling_channels % 32 != 0 ||
      e.subsampling_channels > 1024)
    return fail("the subsampling channels must be a multiple of 32 up to 1024");
  if (e.conv_kernel % 2 != 1 || e.conv_kernel > 31)
    return fail("the convolution kernel must be odd and at most 31");
  if (e.sliding_window < 2)
    return fail("the attention window is too small");
  for (const std::size_t lookahead : e.lookaheads) {
    if (e.left_context() % (lookahead + 1) != 0)
      return fail("the left context must hold whole chunks");
    if (e.left_context() + lookahead + 1 > kMaxAttendedFrames)
      return fail("a query may attend to at most " +
                  std::to_string(kMaxAttendedFrames) + " frames");
  }
  if (std::find(e.lookaheads.begin(), e.lookaheads.end(),
                e.default_lookahead) == e.lookaheads.end())
    return fail("the default lookahead is not a supported one");
  if (config.features.sample_rate != 16000)
    return fail("the feature extractor must expect 16 kHz audio");
  if (config.features.n_fft & (config.features.n_fft - 1))
    return fail("n_fft must be a power of two");
  if (config.features.window > config.features.n_fft)
    return fail("the window is longer than n_fft");
  if (config.blank >= config.vocab)
    return fail("the blank token is outside the joint's outputs");
  if (config.vocab % 16 != 0)
    return fail("the joint's outputs must be a multiple of 16");
  if (config.prompt_hidden == 0 || config.prompt_hidden % 32 != 0)
    return fail("the prompt projector's width must be a multiple of 32");
  if (config.default_prompt >= config.prompts)
    return fail("the default prompt is out of range");
  for (const auto& [key, index] : config.prompt_dictionary)
    if (index >= config.prompts)
      return fail("prompt \"" + key + "\" is out of range");
  if (config.decoder_hidden % 128 != 0 || config.decoder_hidden > 1024)
    return fail("the decoder width must be a multiple of 128 up to 1024");
  if (config.max_symbols == 0)
    return fail("greedy decoding needs at least one symbol per frame");
  return true;
}

std::size_t FirstChunkFrames(const Config& config, std::size_t lookahead) {
  return 1 + config.encoder.subsampling_factor * lookahead;
}

std::size_t ChunkFrames(const Config& config, std::size_t lookahead) {
  return config.encoder.subsampling_factor * (lookahead + 1);
}

}  // namespace gufo::models::nemotron_asr
