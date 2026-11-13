#ifndef GUFO_MODELS_NEMOTRON_ASR_MODEL_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <gufo/ext/cancellable_gate.hpp>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "config.hpp"
#include "features.hpp"
#include "tokenizer.hpp"

namespace gufo::models::nemotron_asr {

struct Weights;
struct StreamDevice;

/// Encoder frames (80 ms each) of a pass at most.
inline constexpr std::size_t kMaxPassFrames = 224;

/// The loaded checkpoint: configuration, tokenizer and weights on the GPU
/// (the conformer layers' matrices in float16, the rest in float32).
/// Streams share it; each stream has its own HIP stream and state.
class Model {
public:
  [[nodiscard]] static std::unique_ptr<Model> Load(
      const std::string& root, std::string* error = nullptr);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] const Config& config() const { return config_; }
  [[nodiscard]] const Tokenizer& tokenizer() const { return *tokenizer_; }
  [[nodiscard]] const Weights& weights() const { return *weights_; }
  /// Largest supported lookahead: the positional tables cover its chunks.
  [[nodiscard]] std::size_t max_lookahead() const { return max_lookahead_; }

private:
  friend class Stream;
  Model() = default;

  /// A stream's device state for passes of up to `frames` encoder frames,
  /// reused from an earlier stream when one is idle; returned when the
  /// stream ends.
  [[nodiscard]] std::unique_ptr<StreamDevice> AcquireDevice(
      std::size_t frames) const;
  void ReleaseDevice(std::unique_ptr<StreamDevice> device) const;

  Config config_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Weights> weights_;
  std::size_t max_lookahead_ = 0;
  mutable std::mutex pool_mutex_;
  mutable std::vector<std::unique_ptr<StreamDevice>> pool_;
  // One pass computes at a time, in the order streams ask (FIFO).
  mutable core::CancellableGate gate_;
};

/// Encoder activations of one pass, copied to the host (for tests).
struct EncoderTrace {
  std::vector<float> subsampled;  // [frames][hidden]
  std::vector<float> hidden;      // [frames][hidden], the encoder output
  std::vector<float> projected;   // [frames][decoder_hidden]
};

/// One utterance in cache-aware streaming: 16 kHz samples in, tokens out.
/// The encoder runs on whole chunks (1 + 8 * lookahead mel frames, then
/// 8 * (lookahead + 1)), greedy RNN-T decoding follows each pass; Finish
/// encodes the rest as offline transcription ends an utterance (its last
/// chunk partial). Feeding the same samples in any split gives the same
/// tokens.
class Stream {
public:
  /// Passes hold up to `pass_frames` encoder frames (whole chunks, at least
  /// one): smaller passes report text sooner, take less device memory and
  /// let other streams compute in between; larger ones finish sooner.
  Stream(const Model& model, std::size_t lookahead, std::size_t prompt,
         std::size_t pass_frames = kMaxPassFrames);
  ~Stream();
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  /// Appends samples and transcribes every completed chunk. With `eager`
  /// false (audio that is complete already) only full passes are encoded
  /// while more than one is pending; Finish splits the rest evenly.
  void Push(std::span<const float> samples, bool eager = true);
  /// Ends the utterance.
  void Finish();

  /// Encodes (and unless `decode` is false, decodes) chunk-complete mel
  /// frames directly ([frames][bins]): the first call takes
  /// 1 + 8 * lookahead + k * 8 * (lookahead + 1) frames, later ones
  /// multiples of 8 * (lookahead + 1).
  void EncodeFeatures(std::span<const float> mel, EncoderTrace* trace = nullptr,
                      bool decode = true);

  /// Polled between passes, while waiting for the GPU and between decoding
  /// windows; true stops with an exception.
  void SetCancellation(std::function<bool()> cancelled) {
    cancelled_ = std::move(cancelled);
  }
  /// Called after each pass that emitted tokens, outside the model's gate.
  void SetTextCallback(std::function<bool()> on_tokens) {
    on_tokens_ = std::move(on_tokens);
  }

  [[nodiscard]] const TextDecoder& text() const { return text_; }
  [[nodiscard]] std::size_t encoder_frames() const { return frames_done_; }
  [[nodiscard]] std::size_t chunks() const { return chunks_; }
  [[nodiscard]] std::size_t mel_frames() const { return features_.frames(); }
  [[nodiscard]] std::size_t samples() const { return features_.samples(); }
  [[nodiscard]] double encoder_ms() const { return encoder_ms_; }
  [[nodiscard]] double decoder_ms() const { return decoder_ms_; }
  [[nodiscard]] double feature_ms() const { return feature_ms_; }

  /// The prediction network's output for the current state (for tests).
  [[nodiscard]] std::vector<float> DecoderOutput() const;
  /// The joint's logits for encoder frame `frame` of the last pass.
  [[nodiscard]] std::vector<float> JointLogits(std::size_t frame) const;

private:
  enum class Pending { kEager, kLazy, kFinal };
  void EncodeReady(Pending mode);
  void EncodePass(std::span<const float> mel, EncoderTrace* trace, bool decode,
                  bool final);
  /// Encodes whole chunks, or when `final` any frames as the end of the
  /// utterance.
  void Encode(const float* mel, std::size_t frames, EncoderTrace* trace,
              bool final);
  void Decode(std::size_t frames);
  void DecoderStep(std::uint32_t token);
  /// Packed arg maxes of the joint for `frames` frames of the last pass
  /// from `frame` on (host memory, valid until the next call).
  [[nodiscard]] const unsigned long long* Joint(std::size_t frame,
                                                std::size_t frames);
  void CheckCancelled() const;

  const Model& model_;
  const Config& config_;
  std::size_t lookahead_;
  std::size_t chunk_;        // encoder frames per chunk
  std::size_t first_mel_;    // mel frames of the first chunk
  std::size_t chunk_mel_;    // mel frames of later chunks
  std::size_t pass_chunks_;  // chunks per encoder pass at most
  std::size_t prompt_;
  FeatureStream features_;
  std::vector<float> pending_;   // mel frames not yet encoded
  std::vector<float> last_mel_;  // the stem's cached input row
  std::unique_ptr<StreamDevice> device_;
  TextDecoder text_;
  std::size_t frames_done_ = 0;
  std::size_t chunks_ = 0;
  std::size_t last_pass_frames_ = 0;
  bool finished_ = false;
  double encoder_ms_ = 0.0;
  double decoder_ms_ = 0.0;
  double feature_ms_ = 0.0;
  std::function<bool()> cancelled_;
  std::function<bool()> on_tokens_;
};

/// Thrown when the cancellation callback asks to stop.
class Cancelled : public std::exception {
public:
  [[nodiscard]] const char* what() const noexcept override {
    return "transcription cancelled";
  }
};

}  // namespace gufo::models::nemotron_asr

#endif  // GUFO_MODELS_NEMOTRON_ASR_MODEL_HPP_
