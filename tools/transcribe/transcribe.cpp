// Development driver:
//   nemotron_asr_transcribe MODEL AUDIO [--language L] [--lookahead N]
//                           [--repeat N] [--push MS]
// prints the transcript, the detected language and the time per pass.
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <gufo/ext/audio.hpp>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "language.hpp"
#include "model.hpp"
#include "resampler.hpp"

namespace asr = gufo::models::nemotron_asr;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " MODEL AUDIO [--language L] [--lookahead N] [--repeat N] "
                 "[--push MS]\n";
    return 2;
  }
  std::optional<std::string> language;
  std::optional<std::size_t> lookahead;
  int repeat = 1;
  std::size_t push_ms = 0;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--language")
      language = argv[i + 1];
    else if (flag == "--lookahead")
      lookahead = std::stoul(argv[i + 1]);
    else if (flag == "--repeat")
      repeat = std::stoi(argv[i + 1]);
    else if (flag == "--push")
      push_ms = std::stoul(argv[i + 1]);
    else {
      std::cerr << "unknown option " << flag << "\n";
      return 2;
    }
  }
  try {
    using Clock = std::chrono::steady_clock;
    const auto load_begin = Clock::now();
    std::string error;
    const auto model = asr::Model::Load(argv[1], &error);
    if (model == nullptr)
      throw std::runtime_error(error);
    std::cout
        << "loaded in "
        << std::chrono::duration<double>(Clock::now() - load_begin).count()
        << " s\n";
    std::ifstream in(argv[2], std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(in),
                            std::istreambuf_iterator<char>()};
    gufo::core::AudioLimits limits;
    limits.formats = gufo::core::AudioFormats::kAll;
    limits.max_seconds = 4 * 3600;
    limits.downmix_mono = true;
    const auto audio = gufo::core::DecodeAudio(
        std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()),
        limits);
    std::vector<float> samples = audio.samples;
    if (audio.sample_rate != 16000) {
      asr::Resampler resampler(audio.sample_rate, 16000);
      std::vector<float> out;
      resampler.Push(samples, &out);
      resampler.Finish(&out);
      samples = std::move(out);
    }
    const auto prompt = asr::ResolveLanguage(model->config(), language, &error);
    if (!prompt)
      throw std::runtime_error(error);
    const std::size_t look = lookahead.value_or(model->max_lookahead());
    const double seconds = static_cast<double>(samples.size()) / 16000.0;
    for (int run = 0; run < repeat; ++run) {
      const auto begin = Clock::now();
      asr::Stream stream(*model, look, *prompt);
      if (push_ms == 0) {
        stream.Push(samples, false);
      } else {
        const std::size_t step = push_ms * 16;
        for (std::size_t at = 0; at < samples.size(); at += step)
          stream.Push(std::span<const float>(
              samples.data() + at, std::min(step, samples.size() - at)));
      }
      stream.Finish();
      const double elapsed =
          std::chrono::duration<double>(Clock::now() - begin).count();
      // hipBLASLt may leave std::cout in fixed notation.
      std::cout << std::defaultfloat << std::setprecision(6);
      std::cout << "run " << run << ": " << seconds << " s audio in " << elapsed
                << " s (RTF " << elapsed / seconds << "), encoder "
                << stream.encoder_ms() << " ms, decoder " << stream.decoder_ms()
                << " ms, features " << stream.feature_ms() << " ms, "
                << stream.chunks() << " chunks, " << stream.text().ids().size()
                << " tokens\n";
      if (run + 1 == repeat) {
        std::cout << "language: " << stream.text().language().value_or("(none)")
                  << "\n"
                  << stream.text().text() << "\n";
      }
    }
  } catch (const std::exception& failure) {
    std::cerr << "error: " << failure.what() << "\n";
    return 1;
  }
  return 0;
}
