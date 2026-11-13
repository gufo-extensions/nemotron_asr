#ifndef GUFO_MODELS_NEMOTRON_ASR_RESAMPLER_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_RESAMPLER_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::nemotron_asr {

/// A streaming polyphase resampler for a rational rate change: a
/// Kaiser-windowed sinc low-pass (flat to 0.913 of the lower Nyquist
/// frequency, about 100 dB stop band), linear phase and no delay. Pushing the
/// samples in any split gives the same output; Finish ends the signal with
/// silence and emits round(n * to / from) samples in all.
class Resampler {
public:
  Resampler(std::uint32_t from, std::uint32_t to);

  void Push(std::span<const float> in, std::vector<float>* out);
  void Finish(std::vector<float>* out);

  [[nodiscard]] bool passthrough() const { return up_ == down_; }

private:
  void Emit(bool final, std::vector<float>* out);

  std::uint64_t up_ = 1;    // L
  std::uint64_t down_ = 1;  // M
  std::uint64_t from_ = 1;
  std::uint64_t to_ = 1;
  std::vector<float> taps_;  // N taps at the upsampled rate
  std::uint64_t center_ = 0;
  std::vector<float> input_;  // samples from absolute index base_ on
  std::uint64_t base_ = 0;
  std::uint64_t received_ = 0;
  std::uint64_t emitted_ = 0;
  bool finished_ = false;
};

}  // namespace gufo::models::nemotron_asr

#endif  // GUFO_MODELS_NEMOTRON_ASR_RESAMPLER_HPP_
