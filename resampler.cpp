#include "resampler.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace gufo::models::nemotron_asr {
namespace {

double BesselI0(double x) {
  double sum = 1.0;
  double term = 1.0;
  for (int k = 1; k < 64; ++k) {
    term *= (x / (2.0 * k)) * (x / (2.0 * k));
    sum += term;
    if (term < sum * 1e-17)
      break;
  }
  return sum;
}

}  // namespace

Resampler::Resampler(std::uint32_t from, std::uint32_t to) {
  if (from == 0 || to == 0)
    throw std::invalid_argument("sample rates must be positive");
  if (from > 64U * to || to > 64U * from)
    throw std::invalid_argument("the rate change is too large");
  const std::uint64_t g = std::gcd(from, to);
  up_ = to / g;
  down_ = from / g;
  from_ = from;
  to_ = to;
  if (up_ == down_)
    return;
  // Cut-off between the pass band (0.913) and the lower Nyquist frequency.
  constexpr double kAttenuation = 100.0;
  const double nyquist = 0.5 * std::min(from, to);
  const double upsampled = static_cast<double>(from) * static_cast<double>(up_);
  const double transition = (1.0 - 0.913) * nyquist;
  const double cutoff = (0.913 + 1.0) / 2.0 * nyquist / upsampled;
  const double delta = 2.0 * std::numbers::pi * transition / upsampled;
  auto length = static_cast<std::uint64_t>(
      std::ceil((kAttenuation - 8.0) / (2.285 * delta)));
  length = std::max<std::uint64_t>(length, 3) | 1U;
  const double beta = 0.1102 * (kAttenuation - 8.7);
  const double norm = BesselI0(beta);
  center_ = (length - 1) / 2;
  taps_.resize(length);
  for (std::uint64_t n = 0; n < length; ++n) {
    const double x = static_cast<double>(n) - static_cast<double>(center_);
    const double arg = 2.0 * cutoff * x;
    const double sinc =
        x == 0.0 ? 1.0
                 : std::sin(std::numbers::pi * arg) / (std::numbers::pi * arg);
    const double ratio = x / static_cast<double>(center_);
    const double window =
        BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / norm;
    taps_[n] = static_cast<float>(static_cast<double>(up_) * 2.0 * cutoff *
                                  sinc * window);
  }
}

void Resampler::Push(std::span<const float> in, std::vector<float>* out) {
  if (finished_)
    throw std::logic_error("Push after Finish");
  if (passthrough()) {
    out->insert(out->end(), in.begin(), in.end());
    received_ += in.size();
    return;
  }
  input_.insert(input_.end(), in.begin(), in.end());
  received_ += in.size();
  Emit(false, out);
}

void Resampler::Finish(std::vector<float>* out) {
  if (finished_)
    return;
  finished_ = true;
  if (passthrough())
    return;
  Emit(true, out);
}

void Resampler::Emit(bool final, std::vector<float>* out) {
  const std::uint64_t length = taps_.size();
  // All output samples once the signal has ended: round(n * to / from).
  const std::uint64_t total = (received_ * to_ + from_ / 2) / from_;
  for (;;) {
    const std::uint64_t m = emitted_;
    if (final && m >= total)
      break;
    const std::uint64_t position = m * down_ + center_;  // upsampled index
    const std::uint64_t last = position / up_;           // newest input used
    if (!final && last >= received_)
      break;
    // Inputs j with 0 <= position - j * L < length.
    const std::uint64_t first =
        position + 1 > length ? (position + 1 - length + up_ - 1) / up_ : 0;
    double sum = 0.0;
    for (std::uint64_t j = first; j <= last; ++j) {
      if (j >= received_)
        break;
      sum += static_cast<double>(taps_[position - j * up_]) *
             static_cast<double>(input_[j - base_]);
    }
    out->push_back(static_cast<float>(sum));
    ++emitted_;
  }
  // Drop inputs no later output reads.
  const std::uint64_t next = emitted_ * down_ + center_;
  const std::uint64_t keep = next + 1 > length ? (next + 1 - length) / up_ : 0;
  if (keep > base_ + 4096) {
    input_.erase(input_.begin(),
                 input_.begin() + static_cast<std::ptrdiff_t>(keep - base_));
    base_ = keep;
  }
}

}  // namespace gufo::models::nemotron_asr
