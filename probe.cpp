#include "probe.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <string>
#include <system_error>

#include "config.hpp"

namespace gufo::models::nemotron_asr {
namespace {

bool HasArchitecture(const std::filesystem::path& root) {
  std::ifstream in(root / "config.json", std::ios::binary);
  if (!in)
    return false;
  const std::string text{std::istreambuf_iterator<char>(in),
                         std::istreambuf_iterator<char>()};
  try {
    const gufo::json::Value config = gufo::json::parse(text);
    const gufo::json::Value* architectures = config.find("architectures");
    if (architectures == nullptr || !architectures->is_array())
      return false;
    for (const gufo::json::Value& item : architectures->items())
      if (item.is_string() && item.str() == "Nemotron3_5AsrForRNNT")
        return true;
  } catch (const std::exception&) {
  }
  return false;
}

}  // namespace

bool ProbeNemotronAsr(std::string_view model_root) {
  std::error_code code;
  const std::filesystem::path root(model_root);
  if (!std::filesystem::is_directory(root, code) ||
      !std::filesystem::is_regular_file(root / "config.json", code) ||
      !std::filesystem::is_regular_file(root / "processor_config.json", code))
    return false;
  if (!HasArchitecture(root))
    return false;
  const auto config = LoadConfig(model_root);
  return config.has_value() && IsSupported(*config);
}

bool ResolveNemotronAsrArtifacts(std::string_view model_root,
                                 std::string* error) {
  const std::filesystem::path root(model_root);
  const auto config = LoadConfig(model_root, error);
  if (!config.has_value() || !IsSupported(*config, error))
    return false;
  constexpr std::array<const char*, 2> kFiles = {"model.safetensors",
                                                 "tokenizer.json"};
  for (const char* file : kFiles) {
    std::error_code code;
    if (!std::filesystem::is_regular_file(root / file, code)) {
      if (error != nullptr)
        *error = "Nemotron ASR model directory lacks " + std::string(file);
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::nemotron_asr
