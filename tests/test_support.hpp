// Helpers shared by the tests: check counting, fixture loading and WAV
// decoding.
#ifndef GUFO_MODELS_NEMOTRON_ASR_TESTS_TEST_SUPPORT_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_TESTS_TEST_SUPPORT_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/audio.hpp>
#include <gufo/ext/json.hpp>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "safetensors.hpp"

namespace nemotron_asr_test {

constexpr int kSkip = 77;
inline int failures = 0;

inline void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

inline std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// Mono float samples of a WAV fixture.
inline std::vector<float> ReadWav(const std::filesystem::path& path,
                                  std::uint32_t* rate = nullptr) {
  const std::string bytes = ReadFile(path);
  gufo::core::AudioLimits limits;
  limits.downmix_mono = true;
  const gufo::core::DecodedAudio audio = gufo::core::DecodeAudio(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()),
      limits);
  if (rate != nullptr)
    *rate = audio.sample_rate;
  return audio.samples;
}

inline std::vector<float> Floats(
    const gufo::models::nemotron_asr::SafetensorsFile& file,
    const std::string& name) {
  const auto* tensor = file.Find(name);
  if (tensor == nullptr ||
      tensor->dtype != gufo::models::nemotron_asr::DType::kF32)
    throw std::runtime_error("golden lacks " + name);
  std::vector<float> values(tensor->Elements());
  std::memcpy(values.data(), tensor->data, tensor->bytes);
  return values;
}

struct Difference {
  double max_abs = 0.0;
  double max_ref = 0.0;
  double mean_abs = 0.0;
};

inline Difference Compare(const std::vector<float>& actual,
                          const std::vector<float>& expected) {
  Difference d;
  if (actual.size() != expected.size()) {
    d.max_abs = INFINITY;
    return d;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double e = std::abs(static_cast<double>(actual[i]) - expected[i]);
    d.max_abs = std::max(d.max_abs, e);
    d.max_ref = std::max(d.max_ref, std::abs(static_cast<double>(expected[i])));
    d.mean_abs += e;
  }
  d.mean_abs /= std::max<std::size_t>(1, actual.size());
  return d;
}

/// Checks max |a - e| <= relative * max |e|.
inline void ExpectClose(const std::string& what,
                        const std::vector<float>& actual,
                        const std::vector<float>& expected, double relative) {
  const Difference d = Compare(actual, expected);
  // Libraries may leave std::cout in fixed notation: format here.
  std::ostringstream line;
  line << std::defaultfloat << std::setprecision(6) << what << ": max abs "
       << d.max_abs << " (of " << d.max_ref << "), mean abs " << d.mean_abs
       << "\n";
  std::cout << line.str();
  Expect(d.max_abs <= relative * d.max_ref,
         what + " differs from the reference");
}

inline gufo::json::Value Golden(const std::filesystem::path& fixtures) {
  return gufo::json::parse(ReadFile(fixtures / "golden.json"));
}

inline std::vector<std::uint32_t> Ids(const gufo::json::Value& array) {
  std::vector<std::uint32_t> ids;
  for (const auto& item : array.items())
    ids.push_back(static_cast<std::uint32_t>(item.as_double()));
  return ids;
}

/// Deterministic pseudo-random split sizes.
class Splits {
public:
  explicit Splits(std::uint32_t seed) : state_(seed) {}
  std::size_t Next(std::size_t limit) {
    state_ = state_ * 1664525U + 1013904223U;
    return 1 + (state_ >> 8) % limit;
  }

private:
  std::uint32_t state_;
};

}  // namespace nemotron_asr_test

#endif  // GUFO_MODELS_NEMOTRON_ASR_TESTS_TEST_SUPPORT_HPP_
