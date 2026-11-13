// The log-mel computation of transformers' NemotronAsrStreamingFeatureExtractor
// (Apache-2.0, see licenses/transformers.txt), whose mel filters are
// librosa's (ISC, see licenses/librosa.txt).
#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace gufo::models::nemotron_asr {
namespace {

constexpr double kMinLogHz = 1000.0;
constexpr double kHzPerMel = 200.0 / 3.0;
constexpr float kLogGuard = 5.9604644775390625e-08F;  // 2^-24

// librosa's Slaney mel scale.
double HzToMel(double hz) {
  const double min_log_mel = kMinLogHz / kHzPerMel;
  const double logstep = std::log(6.4) / 27.0;
  if (hz >= kMinLogHz)
    return min_log_mel + std::log(hz / kMinLogHz) / logstep;
  return hz / kHzPerMel;
}

double MelToHz(double mel) {
  const double min_log_mel = kMinLogHz / kHzPerMel;
  const double logstep = std::log(6.4) / 27.0;
  if (mel >= min_log_mel)
    return kMinLogHz * std::exp(logstep * (mel - min_log_mel));
  return kHzPerMel * mel;
}

/// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney")
/// as float32, row-major [bins][n_fft / 2 + 1].
std::vector<float> MelFilters(std::size_t rate, std::size_t n_fft,
                              std::size_t bins) {
  const std::size_t spectrum = n_fft / 2 + 1;
  std::vector<double> frequencies(spectrum);
  for (std::size_t k = 0; k < spectrum; ++k)
    frequencies[k] = static_cast<double>(k) * static_cast<double>(rate) /
                     static_cast<double>(n_fft);
  const double low = HzToMel(0.0);
  const double high = HzToMel(static_cast<double>(rate) / 2.0);
  std::vector<double> mel(bins + 2);
  for (std::size_t i = 0; i < mel.size(); ++i) {
    // numpy.linspace: start + i * step, the last point exact.
    const double step = (high - low) / static_cast<double>(bins + 1);
    const double point = i + 1 == mel.size() ? high : low + i * step;
    mel[i] = MelToHz(point);
  }
  std::vector<float> filters(bins * spectrum);
  for (std::size_t i = 0; i < bins; ++i) {
    const double lower_width = mel[i + 1] - mel[i];
    const double upper_width = mel[i + 2] - mel[i + 1];
    const double norm = 2.0 / (mel[i + 2] - mel[i]);
    for (std::size_t k = 0; k < spectrum; ++k) {
      const double lower = -(mel[i] - frequencies[k]) / lower_width;
      const double upper = (mel[i + 2] - frequencies[k]) / upper_width;
      const auto weight =
          static_cast<float>(std::max(0.0, std::min(lower, upper)));
      filters[i * spectrum + k] =
          static_cast<float>(static_cast<double>(weight) * norm);
    }
  }
  return filters;
}

}  // namespace

FeatureStream::FeatureStream(const FeatureConfig& config, std::size_t mel_bins)
    : config_(config), bins_(mel_bins), spectrum_(config.n_fft / 2 + 1) {
  const std::size_t n = config_.n_fft;
  if (n == 0 || (n & (n - 1)) != 0 || config_.window > n ||
      config_.window < 2 || config_.hop == 0)
    throw std::invalid_argument("unsupported STFT shape");
  window_.assign(n, 0.0F);
  const std::size_t offset = (n - config_.window) / 2;
  for (std::size_t i = 0; i < config_.window; ++i)
    window_[offset + i] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(i) /
                             static_cast<double>(config_.window - 1)));
  filters_ = MelFilters(config_.sample_rate, n, bins_);
  filter_begin_.assign(bins_, 0);
  filter_end_.assign(bins_, 0);
  for (std::size_t m = 0; m < bins_; ++m) {
    const float* filter = filters_.data() + m * spectrum_;
    std::size_t begin = 0;
    while (begin < spectrum_ && filter[begin] == 0.0F)
      ++begin;
    std::size_t end = spectrum_;
    while (end > begin && filter[end - 1] == 0.0F)
      --end;
    filter_begin_[m] = begin;
    filter_end_[m] = end;
  }
  twiddles_.resize(n / 2);
  for (std::size_t k = 0; k < n / 2; ++k)
    twiddles_[k] =
        std::polar(1.0, -2.0 * std::numbers::pi * static_cast<double>(k) /
                            static_cast<double>(n));
  // The real transform of n samples is a complex one of n / 2.
  const std::size_t half = n / 2;
  std::size_t bits = 0;
  while ((std::size_t{1} << bits) < half)
    ++bits;
  reversal_.resize(half);
  for (std::size_t i = 0; i < half; ++i) {
    std::size_t r = 0;
    for (std::size_t b = 0; b < bits; ++b)
      r |= ((i >> b) & 1U) << (bits - 1 - b);
    reversal_[i] = r;
  }
  fft_.resize(half);
  power_.resize(spectrum_);
}

void FeatureStream::Reset() {
  signal_.clear();
  base_ = 0;
  length_ = 0;
  pushed_ = 0;
  frames_ = 0;
  previous_ = 0.0F;
  started_ = false;
}

void FeatureStream::Append(float emphasized) {
  signal_.push_back(emphasized);
  ++length_;
}

void FeatureStream::Push(std::span<const float> samples,
                         std::vector<float>* frames) {
  const auto coefficient = static_cast<float>(config_.preemphasis);
  for (const float sample : samples) {
    Append(started_ ? sample - coefficient * previous_ : sample);
    previous_ = sample;
    started_ = true;
  }
  pushed_ += samples.size();
  Emit(frames);
}

void FeatureStream::Flush(std::size_t total, std::vector<float>* frames) {
  const std::size_t half = config_.n_fft / 2;
  if (frames_ >= total)
    return;
  // Zeros until frame `total - 1` is complete.
  const std::size_t needed = (total - 1) * config_.hop + half;
  while (length_ < needed)
    Append(0.0F);
  Emit(frames);
}

void FeatureStream::Emit(std::vector<float>* frames) {
  const std::size_t half = config_.n_fft / 2;
  while (length_ >= frames_ * config_.hop + half) {
    const std::size_t at = frames->size();
    frames->resize(at + bins_);
    Frame(frames_, frames->data() + at);
    ++frames_;
  }
  // Keep the samples later frames still read.
  const std::size_t first = frames_ * config_.hop;
  const std::size_t keep_from = first > half ? first - half : 0;
  if (keep_from > base_ + 4096) {
    signal_.erase(
        signal_.begin(),
        signal_.begin() + static_cast<std::ptrdiff_t>(keep_from - base_));
    base_ = keep_from;
  }
}

void FeatureStream::Frame(std::size_t index, float* out) {
  const std::size_t n = config_.n_fft;
  const std::size_t half = n / 2;
  // Window start in the centered signal: index * hop - n_fft / 2.
  const auto start = static_cast<std::ptrdiff_t>(index * config_.hop) -
                     static_cast<std::ptrdiff_t>(half);
  const auto windowed = [&](std::size_t i) {
    const std::ptrdiff_t at = start + static_cast<std::ptrdiff_t>(i);
    float sample = 0.0F;
    if (at >= 0 && static_cast<std::size_t>(at) < length_)
      sample = signal_[static_cast<std::size_t>(at) - base_];
    return static_cast<double>(sample * window_[i]);
  };
  // Even samples as real parts, odd ones as imaginary parts.
  for (std::size_t i = 0; i < half; ++i)
    fft_[reversal_[i]] = {windowed(2 * i), windowed(2 * i + 1)};
  for (std::size_t size = 2; size <= half; size *= 2) {
    const std::size_t stride = n / size;
    for (std::size_t block = 0; block < half; block += size)
      for (std::size_t k = 0; k < size / 2; ++k) {
        const std::complex<double> t =
            twiddles_[k * stride] * fft_[block + k + size / 2];
        fft_[block + k + size / 2] = fft_[block + k] - t;
        fft_[block + k] += t;
      }
  }
  // Split into the transforms of the even and odd samples and combine:
  // X[k] = E[k] + e^(-2 pi i k / n) O[k] for k = 0..n / 2.
  for (std::size_t k = 0; k <= half; ++k) {
    const std::complex<double> z = fft_[k % half];
    const std::complex<double> mirror = std::conj(fft_[(half - k) % half]);
    const std::complex<double> even = 0.5 * (z + mirror);
    const std::complex<double> odd =
        std::complex<double>(0.0, -0.5) * (z - mirror);
    const std::complex<double> turn =
        k < half ? twiddles_[k] : std::complex<double>(-1.0, 0.0);
    power_[k] = static_cast<float>(std::norm(even + turn * odd));
  }
  for (std::size_t m = 0; m < bins_; ++m) {
    // Zero filter weights add exact zeros: skip them.
    const float* filter = filters_.data() + m * spectrum_;
    float sum = 0.0F;
    for (std::size_t k = filter_begin_[m]; k < filter_end_[m]; ++k)
      sum += filter[k] * power_[k];
    out[m] = std::log(sum + kLogGuard);
  }
}

}  // namespace gufo::models::nemotron_asr
