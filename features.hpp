#ifndef GUFO_MODELS_NEMOTRON_ASR_FEATURES_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_FEATURES_HPP_

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

#include "config.hpp"

namespace gufo::models::nemotron_asr {

/// The log-mel front end of the reference's feature extractor, computed
/// incrementally: pre-emphasis, a centered STFT with a symmetric Hann window
/// (zero padding at the start), the power spectrum, Slaney mel filters as
/// librosa builds them and log(x + 2^-24). Frames come out as soon as their
/// window is complete, equal to a centered pass over the whole signal.
class FeatureStream {
public:
  FeatureStream(const FeatureConfig& config, std::size_t mel_bins);

  /// Appends 16 kHz samples; completed frames are appended to `frames`
  /// ([frames][mel_bins]).
  void Push(std::span<const float> samples, std::vector<float>* frames);
  /// Ends the signal: zeros after the pre-emphasized samples (as a centered
  /// STFT pads) until `total` frames exist; the new frames are appended to
  /// `frames`.
  void Flush(std::size_t total, std::vector<float>* frames);
  void Reset();

  /// Samples pushed (without the silence of Flush).
  [[nodiscard]] std::size_t samples() const { return pushed_; }
  /// Frames produced so far.
  [[nodiscard]] std::size_t frames() const { return frames_; }
  /// The valid frames of a centered pass over `samples` samples.
  [[nodiscard]] std::size_t FramesFor(std::size_t samples) const {
    return samples / config_.hop;
  }
  [[nodiscard]] const std::vector<float>& filters() const { return filters_; }

private:
  void Append(float emphasized);
  void Emit(std::vector<float>* frames);
  void Frame(std::size_t index, float* out);

  FeatureConfig config_;
  std::size_t bins_;
  std::size_t spectrum_;
  std::vector<float> window_;   // n_fft, the Hann window centered
  std::vector<float> filters_;  // [bins][spectrum]
  // Each filter's nonzero columns: [filter_begin_[m], filter_end_[m]).
  std::vector<std::size_t> filter_begin_, filter_end_;
  std::vector<std::complex<double>> twiddles_;  // e^(-2 pi i k / n_fft)
  std::vector<std::size_t> reversal_;           // of n_fft / 2 points
  std::vector<std::complex<double>> fft_;
  std::vector<float> power_;
  // Pre-emphasized signal from absolute sample `base_` on.
  std::vector<float> signal_;
  std::size_t base_ = 0;
  std::size_t length_ = 0;  // emphasized samples so far, silence included
  std::size_t pushed_ = 0;
  std::size_t frames_ = 0;
  float previous_ = 0.0F;
  bool started_ = false;
};

}  // namespace gufo::models::nemotron_asr

#endif  // GUFO_MODELS_NEMOTRON_ASR_FEATURES_HPP_
