// The encoder, prompt projector and RNN-T transducer of transformers'
// Nemotron3_5AsrForRNNT and NemotronAsrStreamingEncoder (Apache-2.0, see
// licenses/transformers.txt), which port NVIDIA NeMo's cache-aware
// FastConformer and RNN-T greedy decoding (Apache-2.0, see licenses/NeMo.txt).
// Rewritten in C++ and HIP for Gufo: one chunked path for offline and live
// transcription, positional terms tabulated per layer at load.
#include "model.hpp"

#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "device.hpp"
#include "kernels/kernels.hpp"
#include "kernels/wmma.hpp"
#include "safetensors.hpp"

namespace gufo::models::nemotron_asr {

struct LayerWeights {
  const float* ff1_norm_w = nullptr;
  const float* ff1_norm_b = nullptr;
  const std::uint16_t* ff1_up = nullptr;
  const std::uint16_t* ff1_down = nullptr;
  const float* att_norm_w = nullptr;
  const float* att_norm_b = nullptr;
  const std::uint16_t* qkv = nullptr;  // [3 * hidden][hidden]
  const std::uint16_t* out = nullptr;
  const float* bias_u = nullptr;
  const float* bias_v = nullptr;
  const float* pos = nullptr;  // [pos_rows][hidden]
  const float* conv_norm_w = nullptr;
  const float* conv_norm_b = nullptr;
  const std::uint16_t* pointwise1 = nullptr;  // [2 * hidden][hidden]
  const float* depthwise = nullptr;           // [hidden][kernel]
  const float* conv_inner_norm_w = nullptr;
  const float* conv_inner_norm_b = nullptr;
  const std::uint16_t* pointwise2 = nullptr;
  const float* ff2_norm_w = nullptr;
  const float* ff2_norm_b = nullptr;
  const std::uint16_t* ff2_up = nullptr;
  const std::uint16_t* ff2_down = nullptr;
  const float* out_norm_w = nullptr;
  const float* out_norm_b = nullptr;
};

// The conformer layers' matrices and the joint's head are float16 tiles for
// the matrix cores, the prediction network's matrices float16 rows, all
// summed in float32; everything else stays float32.
struct Weights {
  std::vector<DeviceBuffer<float>> storage;
  std::vector<DeviceBuffer<std::uint16_t>> half_storage;
  const float* stem_w = nullptr;
  const float* stem_b = nullptr;
  const float* stage_dw_w[2] = {};
  const float* stage_dw_b[2] = {};
  const float* stage_pw_w[2] = {};
  const float* stage_pw_b[2] = {};
  const float* sub_linear_w = nullptr;  // columns as [frequency][channel]
  const float* sub_linear_b = nullptr;
  std::vector<LayerWeights> layers;
  const float* prompt_w1 = nullptr;    // [prompt_hidden][hidden + prompts]
  const float* prompt_bias = nullptr;  // [prompts][prompt_hidden], folded
  const float* prompt_w2 = nullptr;
  const float* prompt_b2 = nullptr;
  const float* proj_w = nullptr;
  const float* proj_b = nullptr;
  const float* embedding = nullptr;
  std::vector<const std::uint16_t*>
      lstm_w;                        // [4 * width][2 * width]: input | hidden
  std::vector<const float*> lstm_b;  // b_ih + b_hh
  const std::uint16_t* dec_proj_w = nullptr;
  const float* dec_proj_b = nullptr;
  const std::uint16_t* head_w = nullptr;
  const float* head_b = nullptr;
  std::size_t frequencies[4] = {};  // mel bins, then after each stage
  std::size_t pos_rows = 0;
  std::size_t pos_zero = 0;
};

namespace {

using Clock = std::chrono::steady_clock;

// Idle stream devices a model keeps for reuse.
constexpr std::size_t kPooledDevices = 4;

double Since(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

/// y[rows][out] = x[rows][in] (row stride ldx) * W^T, W [out][ldw] row-major.
void Linear(hipblasHandle_t blas, const float* x, std::size_t rows,
            std::size_t in, std::size_t ldx, const float* w, std::size_t ldw,
            std::size_t out, float* y, std::size_t ldy, float beta = 0.0F) {
  if (rows == 0)
    return;
  const float alpha = 1.0F;
  Check(hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(out),
                     static_cast<int>(rows), static_cast<int>(in), &alpha, w,
                     static_cast<int>(ldw), x, static_cast<int>(ldx), &beta, y,
                     static_cast<int>(ldy)),
        "hipblasSgemm");
}

class Loader {
public:
  Loader(const SafetensorsFile& file, Weights& weights)
      : file_(file), weights_(weights) {}

  const Tensor& Get(const std::string& name,
                    std::initializer_list<std::size_t> shape) {
    std::string error;
    const Tensor* tensor = file_.Expect(name, DType::kF32, shape, &error);
    if (tensor == nullptr)
      throw std::runtime_error(error.empty() ? "model lacks " + name : error);
    return *tensor;
  }

  const float* Upload(const float* data, std::size_t count) {
    DeviceBuffer<float> buffer(count);
    Check(hipMemcpy(buffer.get(), data, count * sizeof(float),
                    hipMemcpyHostToDevice),
          "hipMemcpy weights");
    const float* pointer = buffer.get();
    weights_.storage.push_back(std::move(buffer));
    return pointer;
  }

  const float* Upload(const std::string& name,
                      std::initializer_list<std::size_t> shape) {
    const Tensor& tensor = Get(name, shape);
    return Upload(reinterpret_cast<const float*>(tensor.data),
                  tensor.Elements());
  }

  /// Uploads float32 values as float16.
  const std::uint16_t* UploadHalf(const float* data, std::size_t count) {
    staging_.Reserve(count);
    Check(hipMemcpy(staging_.get(), data, count * sizeof(float),
                    hipMemcpyHostToDevice),
          "hipMemcpy weights");
    DeviceBuffer<std::uint16_t> buffer(count);
    kernels::ToHalf(staging_.get(), buffer.get(),
                    static_cast<std::int64_t>(count), nullptr);
    Check(hipDeviceSynchronize(), "float16 weights");
    const std::uint16_t* pointer = buffer.get();
    weights_.half_storage.push_back(std::move(buffer));
    return pointer;
  }

  const std::uint16_t* UploadHalf(const std::string& name,
                                  std::initializer_list<std::size_t> shape) {
    const Tensor& tensor = Get(name, shape);
    return UploadHalf(reinterpret_cast<const float*>(tensor.data),
                      tensor.Elements());
  }

  /// Uploads a float32 matrix [rows][in] as float16 tiles of the matrix-core
  /// products.
  const std::uint16_t* UploadTiles(const float* data, std::size_t rows,
                                   std::size_t in) {
    const std::size_t count = rows * in;
    staging_.Reserve(count);
    half_staging_.Reserve(count);
    Check(hipMemcpy(staging_.get(), data, count * sizeof(float),
                    hipMemcpyHostToDevice),
          "hipMemcpy weights");
    DeviceBuffer<std::uint16_t> buffer(count);
    kernels::ToHalf(staging_.get(), half_staging_.get(),
                    static_cast<std::int64_t>(count), nullptr);
    kernels::HalfToTiles(half_staging_.get(), rows, in, buffer.get(), nullptr);
    Check(hipDeviceSynchronize(), "float16 tiles");
    const std::uint16_t* pointer = buffer.get();
    weights_.half_storage.push_back(std::move(buffer));
    return pointer;
  }

  const std::uint16_t* UploadTiles(const std::string& name, std::size_t rows,
                                   std::size_t in) {
    const Tensor& tensor = Get(name, {rows, in});
    return UploadTiles(reinterpret_cast<const float*>(tensor.data), rows, in);
  }

  /// UploadTiles for a convolution weight [rows][in][1].
  const std::uint16_t* UploadTiles1x1(const std::string& name, std::size_t rows,
                                      std::size_t in) {
    const Tensor& tensor = Get(name, {rows, in, 1});
    return UploadTiles(reinterpret_cast<const float*>(tensor.data), rows, in);
  }

  static const float* Host(const Tensor& tensor) {
    return reinterpret_cast<const float*>(tensor.data);
  }

private:
  const SafetensorsFile& file_;
  Weights& weights_;
  DeviceBuffer<float> staging_;
  DeviceBuffer<std::uint16_t> half_staging_;
};

/// The sinusoids of transformers' relative positional encoding for
/// distances first..first + rows - 1, computed in float32 as PyTorch does.
std::vector<float> Sinusoids(std::size_t hidden, std::ptrdiff_t first,
                             std::size_t rows) {
  std::vector<float> table(rows * hidden);
  for (std::size_t r = 0; r < rows; ++r) {
    const auto distance =
        static_cast<float>(first + static_cast<std::ptrdiff_t>(r));
    for (std::size_t k = 0; k < hidden / 2; ++k) {
      const float exponent =
          static_cast<float>(2 * k) / static_cast<float>(hidden);
      const float inverse = 1.0F / std::pow(10000.0F, exponent);
      const float angle = distance * inverse;
      table[r * hidden + 2 * k] = std::sin(angle);
      table[r * hidden + 2 * k + 1] = std::cos(angle);
    }
  }
  return table;
}

void LoadWeights(const Config& config, const std::filesystem::path& root,
                 std::size_t max_lookahead, Weights& w) {
  std::string error;
  const auto file = SafetensorsFile::Open(root / "model.safetensors", &error);
  if (file == nullptr)
    throw std::runtime_error(error);
  file->Prefault();
  Loader load(*file, w);
  const EncoderConfig& e = config.encoder;
  const std::size_t h = e.hidden;
  const std::size_t c = e.subsampling_channels;

  w.frequencies[0] = e.mel_bins;
  for (int i = 1; i < 4; ++i)
    w.frequencies[i] = w.frequencies[i - 1] / 2 + 1;

  const std::string sub = "encoder.subsampling.";
  w.stem_w = load.Upload(sub + "conv_in.weight", {c, 1, 3, 3});
  w.stem_b = load.Upload(sub + "conv_in.bias", {c});
  for (int s = 0; s < 2; ++s) {
    const std::string stage = sub + "layers." + std::to_string(s) + ".";
    w.stage_dw_w[s] =
        load.Upload(stage + "depthwise_conv.weight", {c, 1, 3, 3});
    w.stage_dw_b[s] = load.Upload(stage + "depthwise_conv.bias", {c});
    w.stage_pw_w[s] =
        load.Upload(stage + "pointwise_conv.weight", {c, c, 1, 1});
    w.stage_pw_b[s] = load.Upload(stage + "pointwise_conv.bias", {c});
  }
  {
    // The reference flattens [channel][frequency]; ours is
    // [frequency][channel].
    const std::size_t f = w.frequencies[3];
    const Tensor& linear = load.Get(sub + "linear.weight", {h, c * f});
    const float* src = Loader::Host(linear);
    std::vector<float> permuted(h * c * f);
    for (std::size_t o = 0; o < h; ++o)
      for (std::size_t ch = 0; ch < c; ++ch)
        for (std::size_t q = 0; q < f; ++q)
          permuted[o * c * f + q * c + ch] = src[o * c * f + ch * f + q];
    w.sub_linear_w = load.Upload(permuted.data(), permuted.size());
    w.sub_linear_b = load.Upload(sub + "linear.bias", {h});
  }

  // Distances a query sees: from (chunk - 1) ahead to left + chunk - 1 back.
  const std::size_t left = e.left_context();
  w.pos_zero = max_lookahead;
  w.pos_rows = left + 2 * max_lookahead + 1;
  const std::vector<float> sinusoids =
      Sinusoids(h, -static_cast<std::ptrdiff_t>(max_lookahead), w.pos_rows);
  DeviceBuffer<float> sinusoid_table(sinusoids.size());
  Check(hipMemcpy(sinusoid_table.get(), sinusoids.data(),
                  sinusoids.size() * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy sinusoids");
  DeviceBuffer<float> relative(h * h);
  hipblasHandle_t blas = nullptr;
  Check(hipblasCreate(&blas), "hipblasCreate");
  struct BlasGuard {
    hipblasHandle_t handle;
    ~BlasGuard() { (void)hipblasDestroy(handle); }
  } guard{blas};

  for (std::size_t i = 0; i < e.layers; ++i) {
    const std::string p = "encoder.layers." + std::to_string(i) + ".";
    LayerWeights layer;
    layer.ff1_norm_w = load.Upload(p + "norm_feed_forward1.weight", {h});
    layer.ff1_norm_b = load.Upload(p + "norm_feed_forward1.bias", {h});
    layer.ff1_up =
        load.UploadTiles(p + "feed_forward1.linear1.weight", e.intermediate, h);
    layer.ff1_down =
        load.UploadTiles(p + "feed_forward1.linear2.weight", h, e.intermediate);
    layer.att_norm_w = load.Upload(p + "norm_self_att.weight", {h});
    layer.att_norm_b = load.Upload(p + "norm_self_att.bias", {h});
    {
      std::vector<float> qkv(3 * h * h);
      const char* names[3] = {"q_proj", "k_proj", "v_proj"};
      for (int j = 0; j < 3; ++j) {
        const Tensor& t =
            load.Get(p + "self_attn." + names[j] + ".weight", {h, h});
        std::memcpy(qkv.data() + j * h * h, t.data, h * h * sizeof(float));
      }
      layer.qkv = load.UploadTiles(qkv.data(), 3 * h, h);
    }
    layer.out = load.UploadTiles(p + "self_attn.o_proj.weight", h, h);
    layer.bias_u = load.Upload(p + "self_attn.bias_u", {e.heads, e.head_dim()});
    layer.bias_v = load.Upload(p + "self_attn.bias_v", {e.heads, e.head_dim()});
    {
      const Tensor& r =
          load.Get(p + "self_attn.relative_k_proj.weight", {h, h});
      Check(hipMemcpy(relative.get(), r.data, h * h * sizeof(float),
                      hipMemcpyHostToDevice),
            "hipMemcpy relative_k_proj");
      DeviceBuffer<float> table(w.pos_rows * h);
      Linear(blas, sinusoid_table.get(), w.pos_rows, h, h, relative.get(), h, h,
             table.get(), h);
      Check(hipDeviceSynchronize(), "positional tables");
      layer.pos = table.get();
      w.storage.push_back(std::move(table));
    }
    layer.conv_norm_w = load.Upload(p + "norm_conv.weight", {h});
    layer.conv_norm_b = load.Upload(p + "norm_conv.bias", {h});
    layer.pointwise1 =
        load.UploadTiles1x1(p + "conv.pointwise_conv1.weight", 2 * h, h);
    layer.depthwise =
        load.Upload(p + "conv.depthwise_conv.weight", {h, 1, e.conv_kernel});
    layer.conv_inner_norm_w = load.Upload(p + "conv.norm.weight", {h});
    layer.conv_inner_norm_b = load.Upload(p + "conv.norm.bias", {h});
    layer.pointwise2 =
        load.UploadTiles1x1(p + "conv.pointwise_conv2.weight", h, h);
    layer.ff2_norm_w = load.Upload(p + "norm_feed_forward2.weight", {h});
    layer.ff2_norm_b = load.Upload(p + "norm_feed_forward2.bias", {h});
    layer.ff2_up =
        load.UploadTiles(p + "feed_forward2.linear1.weight", e.intermediate, h);
    layer.ff2_down =
        load.UploadTiles(p + "feed_forward2.linear2.weight", h, e.intermediate);
    layer.out_norm_w = load.Upload(p + "norm_out.weight", {h});
    layer.out_norm_b = load.Upload(p + "norm_out.bias", {h});
    w.layers.push_back(layer);
  }

  const std::size_t ph = config.prompt_hidden;
  const std::size_t prompts = config.prompts;
  const std::size_t d = config.decoder_hidden;
  {
    const Tensor& w1 =
        load.Get("prompt_projector.linear_1.weight", {ph, h + prompts});
    const Tensor& b1 = load.Get("prompt_projector.linear_1.bias", {ph});
    w.prompt_w1 = load.Upload(Loader::Host(w1), w1.Elements());
    // The one-hot prompt selects one column of linear_1: fold it into the bias.
    std::vector<float> folded(prompts * ph);
    for (std::size_t q = 0; q < prompts; ++q)
      for (std::size_t r = 0; r < ph; ++r)
        folded[q * ph + r] =
            Loader::Host(b1)[r] + Loader::Host(w1)[r * (h + prompts) + h + q];
    w.prompt_bias = load.Upload(folded.data(), folded.size());
  }
  w.prompt_w2 = load.Upload("prompt_projector.linear_2.weight", {h, ph});
  w.prompt_b2 = load.Upload("prompt_projector.linear_2.bias", {h});
  w.proj_w = load.Upload("encoder_projector.weight", {d, h});
  w.proj_b = load.Upload("encoder_projector.bias", {d});

  w.embedding = load.Upload("decoder.embedding.weight", {config.vocab, d});
  for (std::size_t l = 0; l < config.decoder_layers; ++l) {
    const std::string s = std::to_string(l);
    const Tensor& wi = load.Get("decoder.lstm.weight_ih_l" + s, {4 * d, d});
    const Tensor& wh = load.Get("decoder.lstm.weight_hh_l" + s, {4 * d, d});
    const Tensor& bi = load.Get("decoder.lstm.bias_ih_l" + s, {4 * d});
    const Tensor& bh = load.Get("decoder.lstm.bias_hh_l" + s, {4 * d});
    std::vector<float> joined(4 * d * 2 * d);
    std::vector<float> bias(4 * d);
    for (std::size_t r = 0; r < 4 * d; ++r) {
      std::memcpy(joined.data() + r * 2 * d, Loader::Host(wi) + r * d,
                  d * sizeof(float));
      std::memcpy(joined.data() + r * 2 * d + d, Loader::Host(wh) + r * d,
                  d * sizeof(float));
      bias[r] = Loader::Host(bi)[r] + Loader::Host(bh)[r];
    }
    w.lstm_w.push_back(load.UploadHalf(joined.data(), joined.size()));
    w.lstm_b.push_back(load.Upload(bias.data(), bias.size()));
  }
  w.dec_proj_w = load.UploadHalf("decoder.decoder_projector.weight", {d, d});
  w.dec_proj_b = load.Upload("decoder.decoder_projector.bias", {d});
  w.head_w = load.UploadTiles("joint.head.weight", config.vocab, d);
  w.head_b = load.Upload("joint.head.bias", {config.vocab});
  file->ReleasePages();
}

/// y[rows][out] = scale * x[rows][in] W^T, plus y when `accumulate`, on the
/// matrix cores.
void Linear(const std::uint16_t* x, const std::uint16_t* tiles, float* y,
            std::size_t rows, std::size_t in, std::size_t out, bool accumulate,
            hipStream_t stream, float scale = 1.0F) {
  if (!kernels::WmmaLinear(x, tiles, y, rows, in, out, accumulate, scale,
                           stream))
    throw std::logic_error("unsupported matrix-core product shape");
}

}  // namespace

std::unique_ptr<Model> Model::Load(const std::string& root,
                                   std::string* error) {
  try {
    auto model = std::unique_ptr<Model>(new Model());
    std::string failure;
    auto config = LoadConfig(root, &failure);
    if (!config.has_value() || !IsSupported(*config, &failure))
      throw std::runtime_error(failure);
    model->config_ = std::move(*config);
    model->tokenizer_ = Tokenizer::Load(
        std::filesystem::path(root) / "tokenizer.json", &failure);
    if (model->tokenizer_ == nullptr)
      throw std::runtime_error(failure);
    if (model->tokenizer_->size() < model->config_.vocab - 1)
      throw std::runtime_error(
          "tokenizer.json has fewer tokens than the joint");
    model->max_lookahead_ =
        *std::max_element(model->config_.encoder.lookaheads.begin(),
                          model->config_.encoder.lookaheads.end());
    if (model->config_.encoder.left_context() + model->max_lookahead_ + 1 > 256)
      throw std::runtime_error("the attention window exceeds 256 frames");
    model->weights_ = std::make_unique<Weights>();
    LoadWeights(model->config_, root, model->max_lookahead_, *model->weights_);
    return model;
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = std::string("Nemotron ASR: ") + failure.what();
    return nullptr;
  }
}

// A stream's HIP stream, scratch buffers and caches, for passes of up to
// `frames` encoder frames (and 8 times as many mel frames). Streams of any
// lookahead reuse them.
struct StreamDevice {
  // The handles first: they outlive the buffers and are released even when
  // a buffer allocation fails in the constructor.
  struct Handles {
    hipStream_t stream = nullptr;
    hipblasHandle_t blas = nullptr;
    ~Handles() {
      if (blas != nullptr)
        (void)hipblasDestroy(blas);
      if (stream != nullptr)
        (void)hipStreamDestroy(stream);
    }
  } handles;
  hipStream_t& stream = handles.stream;
  hipblasHandle_t& blas = handles.blas;
  std::size_t frames = 0;
  DeviceBuffer<float> mel, s1, s1_cache, s2_in, s2, s2_cache, s3_in, s3;
  DeviceBuffer<float> x, wide, mid, projected;
  DeviceBuffer<std::uint16_t> norm_half, wide_half, attention_half;
  // [layers][left][hidden] and [layers][kernel - 1][hidden].
  DeviceBuffer<float> key_cache, value_cache, conv_cache;
  DeviceBuffer<float> h, c, gates, decoder_out, logits;
  DeviceBuffer<std::uint16_t> joint_input;
  DeviceBuffer<unsigned long long> best;
  unsigned long long* best_host = nullptr;

  StreamDevice(const Config& config, const Weights& w, std::size_t capacity)
      : frames(capacity) {
    const EncoderConfig& e = config.encoder;
    Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
          "hipStreamCreate");
    Check(hipblasCreate(&blas), "hipblasCreate");
    Check(hipblasSetStream(blas, stream), "hipblasSetStream");
    const std::size_t h_size = e.hidden;
    const std::size_t ch = e.subsampling_channels;
    const std::size_t max_mel = frames * e.subsampling_factor;
    const std::size_t t1 = max_mel / 2 + 2;
    const std::size_t t2 = t1 / 2 + 2;
    const std::size_t t3 = t2 / 2 + 2;
    mel.Reserve((max_mel + 2) * e.mel_bins);
    s1.Reserve((t1 + 2) * w.frequencies[1] * ch);
    s1_cache.Reserve(w.frequencies[1] * ch);
    s2_in.Reserve(t2 * w.frequencies[2] * ch);
    s2.Reserve((t2 + 2) * w.frequencies[2] * ch);
    s2_cache.Reserve(w.frequencies[2] * ch);
    s3_in.Reserve(t3 * w.frequencies[3] * ch);
    s3.Reserve(t3 * w.frequencies[3] * ch);
    const std::size_t rows = std::max(frames, t3);
    const std::size_t widest =
        std::max({e.intermediate, 3 * h_size, config.prompt_hidden});
    x.Reserve(rows * h_size);
    wide.Reserve(rows * widest);
    mid.Reserve(rows * h_size);
    norm_half.Reserve(rows * h_size);
    wide_half.Reserve(rows * e.intermediate);
    attention_half.Reserve(rows * h_size);
    projected.Reserve(rows * config.decoder_hidden);
    key_cache.Reserve(e.layers * e.left_context() * h_size);
    value_cache.Reserve(e.layers * e.left_context() * h_size);
    conv_cache.Reserve(e.layers * (e.conv_kernel - 1) * h_size);
    const std::size_t width = config.decoder_hidden;
    h.Reserve(config.decoder_layers * width);
    c.Reserve(config.decoder_layers * width);
    gates.Reserve(4 * width);
    decoder_out.Reserve(width);
    logits.Reserve(kernels::kMaxJointFrames * config.vocab);
    joint_input.Reserve(kernels::kMaxJointFrames * width);
    best.Reserve(kernels::kMaxJointFrames);
    Check(hipHostMalloc(reinterpret_cast<void**>(&best_host),
                        kernels::kMaxJointFrames * sizeof(unsigned long long),
                        hipHostMallocDefault),
          "hipHostMalloc");
  }

  ~StreamDevice() {
    if (best_host != nullptr)
      (void)hipHostFree(best_host);
  }

  StreamDevice(const StreamDevice&) = delete;
  StreamDevice& operator=(const StreamDevice&) = delete;
};

Model::~Model() = default;

std::unique_ptr<StreamDevice> Model::AcquireDevice(std::size_t frames) const {
  {
    // The smallest idle device that holds such passes.
    std::lock_guard lock(pool_mutex_);
    auto chosen = pool_.end();
    for (auto it = pool_.begin(); it != pool_.end(); ++it)
      if ((*it)->frames >= frames &&
          (chosen == pool_.end() || (*it)->frames < (*chosen)->frames))
        chosen = it;
    if (chosen != pool_.end()) {
      auto device = std::move(*chosen);
      pool_.erase(chosen);
      return device;
    }
  }
  return std::make_unique<StreamDevice>(config_, *weights_, frames);
}

void Model::ReleaseDevice(std::unique_ptr<StreamDevice> device) const {
  // Work a cancelled stream left queued finishes before another stream
  // reuses the buffers; a device whose stream failed is dropped.
  if (hipStreamSynchronize(device->stream) != hipSuccess)
    return;
  std::lock_guard lock(pool_mutex_);
  if (pool_.size() < kPooledDevices)
    pool_.push_back(std::move(device));
}

Stream::Stream(const Model& model, std::size_t lookahead, std::size_t prompt,
               std::size_t pass_frames)
    : model_(model),
      config_(model.config()),
      lookahead_(lookahead),
      chunk_(lookahead + 1),
      first_mel_(FirstChunkFrames(model.config(), lookahead)),
      chunk_mel_(ChunkFrames(model.config(), lookahead)),
      pass_chunks_(std::max<std::size_t>(
          1, std::min(pass_frames, kMaxPassFrames) / (lookahead + 1))),
      prompt_(prompt),
      features_(model.config().features, model.config().encoder.mel_bins),
      text_(model.tokenizer()) {
  const EncoderConfig& e = config_.encoder;
  if (std::find(e.lookaheads.begin(), e.lookaheads.end(), lookahead) ==
      e.lookaheads.end())
    throw std::invalid_argument("unsupported lookahead " +
                                std::to_string(lookahead));
  if (prompt >= config_.prompts)
    throw std::invalid_argument("language prompt out of range");
  device_ = model.AcquireDevice(pass_chunks_ * chunk_);
  StreamDevice& d = *device_;
  const std::size_t h = e.hidden;
  Check(hipMemsetAsync(d.conv_cache.get(), 0,
                       e.layers * (e.conv_kernel - 1) * h * sizeof(float),
                       d.stream),
        "hipMemset conv cache");
  const std::size_t width = config_.decoder_hidden;
  Check(
      hipMemsetAsync(d.h.get(), 0,
                     config_.decoder_layers * width * sizeof(float), d.stream),
      "hipMemset h");
  Check(
      hipMemsetAsync(d.c.get(), 0,
                     config_.decoder_layers * width * sizeof(float), d.stream),
      "hipMemset c");
  // The prediction network starts from the blank token (a zero embedding).
  DecoderStep(static_cast<std::uint32_t>(config_.blank));
  Check(hipStreamSynchronize(d.stream), "decoder start");
}

Stream::~Stream() {
  if (device_ != nullptr)
    model_.ReleaseDevice(std::move(device_));
}

void Stream::CheckCancelled() const {
  if (cancelled_ && cancelled_())
    throw Cancelled();
}

void Stream::Push(std::span<const float> samples, bool eager) {
  if (finished_)
    throw std::logic_error("Push after Finish");
  const auto begin = Clock::now();
  features_.Push(samples, &pending_);
  feature_ms_ += Since(begin);
  EncodeReady(eager ? Pending::kEager : Pending::kLazy);
}

// The end of the audio as offline transcription sees it: the valid frames
// of a centered pass (samples / hop), the signal padded with zeros, and a
// last chunk with only the frames that exist.
void Stream::Finish() {
  if (finished_)
    return;
  finished_ = true;
  const std::size_t total = features_.FramesFor(features_.samples());
  if (total == 0)
    return;
  const auto begin = Clock::now();
  features_.Flush(total, &pending_);
  feature_ms_ += Since(begin);
  EncodeReady(Pending::kFinal);
  if (!pending_.empty())
    throw std::logic_error("frames left after the last pass");
}

// Encodes the complete chunks pending, at most pass_chunks_ per pass:
// eagerly all of them, lazily only while more than a pass would remain, and
// at the end everything, the last chunk partial, in passes of even size
// (a pass costs nearly as much for one chunk as for many).
void Stream::EncodeReady(Pending mode) {
  const std::size_t bins = config_.encoder.mel_bins;
  std::size_t at = 0;  // frames of pending_ encoded
  for (;;) {
    CheckCancelled();
    const std::size_t available = pending_.size() / bins - at;
    if (available == 0)
      break;
    const bool first = frames_done_ == 0;
    const auto whole_frames = [&](std::size_t chunks) -> std::size_t {
      if (chunks == 0)
        return 0;
      return first ? first_mel_ + (chunks - 1) * chunk_mel_
                   : chunks * chunk_mel_;
    };
    std::size_t whole = available / chunk_mel_;
    if (first)
      whole = available < first_mel_
                  ? 0
                  : 1 + (available - first_mel_) / chunk_mel_;
    std::size_t frames = 0;
    bool last = false;
    if (mode == Pending::kEager) {
      if (whole == 0)
        break;
      frames = whole_frames(std::min(whole, pass_chunks_));
    } else if (mode == Pending::kLazy) {
      if (whole <= pass_chunks_)
        break;
      frames = whole_frames(pass_chunks_);
    } else {
      const std::size_t partial = available - whole_frames(whole);
      const std::size_t chunks = whole + (partial > 0 ? 1 : 0);
      const std::size_t passes = (chunks + pass_chunks_ - 1) / pass_chunks_;
      const std::size_t take = (chunks + passes - 1) / passes;
      last = take >= chunks;
      frames = last ? available : whole_frames(take);
    }
    EncodePass(
        std::span<const float>(pending_.data() + at * bins, frames * bins),
        nullptr, true, last);
    at += frames;
  }
  pending_.erase(pending_.begin(),
                 pending_.begin() + static_cast<std::ptrdiff_t>(at * bins));
}

void Stream::EncodeFeatures(std::span<const float> mel, EncoderTrace* trace,
                            bool decode) {
  EncodePass(mel, trace, decode, false);
}

void Stream::EncodePass(std::span<const float> mel, EncoderTrace* trace,
                        bool decode, bool final) {
  const std::size_t bins = config_.encoder.mel_bins;
  if (mel.size() % bins != 0)
    throw std::invalid_argument("mel features are not whole frames");
  const std::size_t frames = mel.size() / bins;
  const std::size_t before = text_.ids().size();
  {
    // The GPU work of the pass; text callbacks follow outside the gate.
    std::string error;
    auto lease = model_.gate_.Acquire(cancelled_, &error);
    if (!lease.has_value()) {
      if (cancelled_ && cancelled_())
        throw Cancelled();
      throw std::runtime_error(error);
    }
    Encode(mel.data(), frames, trace, final);
    if (decode)
      Decode(last_pass_frames_);
  }
  if (decode && on_tokens_ && text_.ids().size() != before && !on_tokens_())
    throw Cancelled();
}

namespace {

/// Rows a causal 3-tap stride-2 convolution makes of `rows` input rows.
std::size_t Strided(std::size_t rows) {
  return rows >= 3 ? (rows - 3) / 2 + 1 : 0;
}

}  // namespace

void Stream::Encode(const float* mel, std::size_t frames, EncoderTrace* trace,
                    bool final) {
  StreamDevice& d = *device_;
  const Weights& w = model_.weights();
  const EncoderConfig& e = config_.encoder;
  const std::size_t bins = e.mel_bins;
  const std::size_t h = e.hidden;
  const std::size_t ch = e.subsampling_channels;
  const bool first = frames_done_ == 0;
  std::size_t chunks = 0;
  if (frames == 0)
    throw std::invalid_argument("a pass needs frames");
  if (!final && first) {
    if (frames < first_mel_ || (frames - first_mel_) % chunk_mel_ != 0)
      throw std::invalid_argument("the first pass needs whole chunks");
    chunks = 1 + (frames - first_mel_) / chunk_mel_;
  } else if (!final) {
    if (frames % chunk_mel_ != 0)
      throw std::invalid_argument("a pass needs whole chunks");
    chunks = frames / chunk_mel_;
  }
  if (frames > pass_chunks_ * chunk_mel_)
    throw std::invalid_argument("too many mel frames for one pass");
  const auto begin = Clock::now();
  hipStream_t s = d.stream;
  // Rows of history before each stage's input: the causal padding (two
  // zero rows) at the start, else the input row the previous pass left.
  const std::size_t pre = first ? 2 : 1;
  // At the end, each stage's input continues with a zero row, the causal
  // convolutions' right padding when they see the whole utterance.
  const std::size_t tail = final ? 1 : 0;

  // Stem input: the history rows, then the frames (and the tail).
  {
    std::vector<float> input((pre + frames + tail) * bins, 0.0F);
    if (!first)
      std::copy(last_mel_.begin(), last_mel_.end(), input.begin());
    std::copy(mel, mel + frames * bins, input.begin() + pre * bins);
    last_mel_.assign(mel + (frames - 1) * bins, mel + frames * bins);
    Check(
        hipMemcpyAsync(d.mel.get(), input.data(), input.size() * sizeof(float),
                       hipMemcpyHostToDevice, s),
        "hipMemcpy mel");
    // The copy reads `input` asynchronously: finish it before it goes away.
    Check(hipStreamSynchronize(s), "mel upload");
  }
  const std::size_t f1 = w.frequencies[1];
  const std::size_t f2 = w.frequencies[2];
  const std::size_t f3 = w.frequencies[3];
  const std::size_t t1 = Strided(pre + frames + tail);
  kernels::SubsampleInput(d.mel.get(), static_cast<int>(bins), w.stem_w,
                          w.stem_b, static_cast<int>(ch),
                          d.s1.get() + 2 * f1 * ch, static_cast<int>(t1),
                          static_cast<int>(f1), s);

  // Two depthwise-separable stages, each with its cached input row.
  struct Stage {
    float* in;  // rows from offset 2 are this pass's input
    float* cache;
    std::size_t rows;  // this pass's input rows
    std::size_t freq;  // input frequencies
    float* out_dw;
    float* out;  // pointwise output (next stage input at offset)
    std::size_t out_offset_rows;
    std::size_t out_freq;
  };
  Stage stages[2] = {
      {d.s1.get(), d.s1_cache.get(), t1, f1, d.s2_in.get(), d.s2.get(), 2, f2},
      {d.s2.get(), d.s2_cache.get(), 0, f2, d.s3_in.get(), d.s3.get(), 0, f3},
  };
  std::size_t rows_out = 0;
  for (int i = 0; i < 2; ++i) {
    Stage& st = stages[i];
    if (i == 1)
      st.rows = rows_out;
    const std::size_t row = st.freq * ch;
    if (first)
      Check(hipMemsetAsync(st.in, 0, 2 * row * sizeof(float), s),
            "hipMemset stage pad");
    else
      Check(hipMemcpyAsync(st.in + row, st.cache, row * sizeof(float),
                           hipMemcpyDeviceToDevice, s),
            "stage cache in");
    if (tail != 0)
      Check(hipMemsetAsync(st.in + (2 + st.rows) * row, 0, row * sizeof(float),
                           s),
            "hipMemset stage tail");
    const float* input = st.in + (2 - pre) * row;
    rows_out = Strided(pre + st.rows + tail);
    kernels::SubsampleDepthwise(
        input, static_cast<int>(st.freq), w.stage_dw_w[i], w.stage_dw_b[i],
        static_cast<int>(ch), st.out_dw, static_cast<int>(rows_out),
        static_cast<int>(st.out_freq), s);
    Check(hipMemcpyAsync(st.cache, st.in + (2 + st.rows - 1) * row,
                         row * sizeof(float), hipMemcpyDeviceToDevice, s),
          "stage cache out");
    float* out = st.out + st.out_offset_rows * st.out_freq * ch;
    Linear(d.blas, st.out_dw, rows_out * st.out_freq, ch, ch, w.stage_pw_w[i],
           ch, ch, out, ch);
    kernels::AddBias(out, w.stage_pw_b[i],
                     static_cast<int>(rows_out * st.out_freq),
                     static_cast<int>(ch), true, s);
  }
  const std::size_t t = rows_out;
  if (!final && t != chunks * chunk_)
    throw std::logic_error("subsampling produced an unexpected frame count");
  chunks = (t + chunk_ - 1) / chunk_;
  if (t == 0) {
    Check(hipStreamSynchronize(s), "encoder");
    last_pass_frames_ = 0;
    encoder_ms_ += Since(begin);
    return;
  }
  Linear(d.blas, d.s3.get(), t, f3 * ch, f3 * ch, w.sub_linear_w, f3 * ch, h,
         d.x.get(), h);
  kernels::AddBias(d.x.get(), w.sub_linear_b, static_cast<int>(t),
                   static_cast<int>(h), false, s);
  if (trace != nullptr) {
    trace->subsampled.resize(t * h);
    Check(hipMemcpyAsync(trace->subsampled.data(), d.x.get(),
                         t * h * sizeof(float), hipMemcpyDeviceToHost, s),
          "trace");
  }

  const std::size_t left = e.left_context();
  const std::size_t cached = std::min(left, frames_done_);
  kernels::AttentionShape shape;
  shape.queries = static_cast<int>(t);
  shape.heads = static_cast<int>(e.heads);
  shape.q_stride = static_cast<int>(3 * h);
  shape.first_query = static_cast<std::int64_t>(frames_done_);
  shape.first_key = static_cast<std::int64_t>(frames_done_ - cached);
  shape.chunk = static_cast<int>(chunk_);
  shape.left = static_cast<int>(left);
  shape.pos_zero = static_cast<int>(w.pos_zero);
  shape.pos_rows = static_cast<int>(w.pos_rows);
  shape.scale = 1.0F / std::sqrt(static_cast<float>(e.head_dim()));
  const std::size_t keep = std::min(left, cached + t);

  std::uint16_t* norm = d.norm_half.get();
  // Half-step feed-forward: x += 0.5 * W2 silu(W1 LayerNorm(x)).
  const auto feed_forward = [&](const float* norm_w, const float* norm_b,
                                const std::uint16_t* up,
                                const std::uint16_t* down) {
    kernels::LayerNormHalf(d.x.get(), norm_w, norm_b, norm, static_cast<int>(t),
                           static_cast<int>(h), false, s);
    Linear(norm, up, d.wide.get(), t, h, e.intermediate, false, s);
    kernels::SiluHalf(d.wide.get(), d.wide_half.get(),
                      static_cast<std::int64_t>(t * e.intermediate), s);
    Linear(d.wide_half.get(), down, d.x.get(), t, e.intermediate, h, true, s,
           0.5F);
  };

  for (std::size_t l = 0; l < e.layers; ++l) {
    const LayerWeights& lw = w.layers[l];
    float* key_cache = d.key_cache.get() + l * left * h;
    float* value_cache = d.value_cache.get() + l * left * h;
    float* conv_cache = d.conv_cache.get() + l * (e.conv_kernel - 1) * h;
    feed_forward(lw.ff1_norm_w, lw.ff1_norm_b, lw.ff1_up, lw.ff1_down);

    // Self-attention over the cached and the new frames.
    kernels::LayerNormHalf(d.x.get(), lw.att_norm_w, lw.att_norm_b, norm,
                           static_cast<int>(t), static_cast<int>(h), false, s);
    Linear(norm, lw.qkv, d.wide.get(), t, h, 3 * h, false, s);
    kernels::RelPosAttentionHalf(d.wide.get(), key_cache, value_cache, lw.pos,
                                 lw.bias_u, lw.bias_v, d.attention_half.get(),
                                 shape, s);
    kernels::UpdateKeyValueCache(
        key_cache, value_cache, d.wide.get(), static_cast<int>(3 * h),
        static_cast<int>(cached), static_cast<int>(t), static_cast<int>(keep),
        static_cast<int>(h), s);
    Linear(d.attention_half.get(), lw.out, d.x.get(), t, h, h, true, s);

    // Convolution module: GLU, causal depthwise convolution, LayerNorm, SiLU.
    kernels::LayerNormHalf(d.x.get(), lw.conv_norm_w, lw.conv_norm_b, norm,
                           static_cast<int>(t), static_cast<int>(h), false, s);
    Linear(norm, lw.pointwise1, d.wide.get(), t, h, 2 * h, false, s);
    kernels::GluDepthwise(d.wide.get(), conv_cache, lw.depthwise, d.mid.get(),
                          static_cast<int>(t), static_cast<int>(h),
                          static_cast<int>(e.conv_kernel), s);
    kernels::UpdateConvCache(d.wide.get(), conv_cache, static_cast<int>(t),
                             static_cast<int>(h),
                             static_cast<int>(e.conv_kernel), s);
    kernels::LayerNormHalf(d.mid.get(), lw.conv_inner_norm_w,
                           lw.conv_inner_norm_b, norm, static_cast<int>(t),
                           static_cast<int>(h), true, s);
    Linear(norm, lw.pointwise2, d.x.get(), t, h, h, true, s);

    feed_forward(lw.ff2_norm_w, lw.ff2_norm_b, lw.ff2_up, lw.ff2_down);
    kernels::LayerNorm(d.x.get(), lw.out_norm_w, lw.out_norm_b, d.x.get(),
                       static_cast<int>(t), static_cast<int>(h), s);
  }
  if (trace != nullptr) {
    trace->hidden.resize(t * h);
    Check(hipMemcpyAsync(trace->hidden.data(), d.x.get(), t * h * sizeof(float),
                         hipMemcpyDeviceToHost, s),
          "trace");
  }

  // Language prompt and projection to the joint's width.
  const std::size_t ph = config_.prompt_hidden;
  const std::size_t width = config_.decoder_hidden;
  Linear(d.blas, d.x.get(), t, h, h, w.prompt_w1, h + config_.prompts, ph,
         d.wide.get(), ph);
  kernels::AddBias(d.wide.get(), w.prompt_bias + prompt_ * ph,
                   static_cast<int>(t), static_cast<int>(ph), true, s);
  Linear(d.blas, d.wide.get(), t, ph, ph, w.prompt_w2, ph, h, d.mid.get(), h);
  kernels::AddBias(d.mid.get(), w.prompt_b2, static_cast<int>(t),
                   static_cast<int>(h), false, s);
  Linear(d.blas, d.mid.get(), t, h, h, w.proj_w, h, width, d.projected.get(),
         width);
  kernels::AddBias(d.projected.get(), w.proj_b, static_cast<int>(t),
                   static_cast<int>(width), false, s);
  if (trace != nullptr) {
    trace->projected.resize(t * width);
    Check(hipMemcpyAsync(trace->projected.data(), d.projected.get(),
                         t * width * sizeof(float), hipMemcpyDeviceToHost, s),
          "trace");
  }
  Check(hipStreamSynchronize(s), "encoder");
  frames_done_ += t;
  chunks_ += chunks;
  last_pass_frames_ = t;
  encoder_ms_ += Since(begin);
}

void Stream::DecoderStep(std::uint32_t token) {
  StreamDevice& d = *device_;
  const Weights& w = model_.weights();
  const std::size_t width = config_.decoder_hidden;
  const float* x = w.embedding + static_cast<std::size_t>(token) * width;
  for (std::size_t l = 0; l < config_.decoder_layers; ++l) {
    float* h = d.h.get() + l * width;
    kernels::Gemv(w.lstm_w[l], x, static_cast<int>(width), h,
                  static_cast<int>(width), w.lstm_b[l], d.gates.get(),
                  static_cast<int>(4 * width), d.stream);
    kernels::LstmCell(d.gates.get(), h, d.c.get() + l * width,
                      static_cast<int>(width), d.stream);
    x = h;
  }
  kernels::Gemv(w.dec_proj_w, x, static_cast<int>(width), nullptr, 0,
                w.dec_proj_b, d.decoder_out.get(), static_cast<int>(width),
                d.stream);
}

const unsigned long long* Stream::Joint(std::size_t frame, std::size_t frames) {
  StreamDevice& d = *device_;
  const Weights& w = model_.weights();
  const std::size_t width = config_.decoder_hidden;
  kernels::JointInput(d.projected.get() + frame * width, d.decoder_out.get(),
                      d.joint_input.get(), static_cast<int>(frames),
                      static_cast<int>(width), d.stream);
  Linear(d.joint_input.get(), w.head_w, d.logits.get(), frames, width,
         config_.vocab, false, d.stream);
  kernels::ArgmaxRows(d.logits.get(), w.head_b, static_cast<int>(frames),
                      static_cast<int>(config_.vocab), d.best.get(), d.stream);
  Check(hipMemcpyAsync(d.best_host, d.best.get(),
                       frames * sizeof(unsigned long long),
                       hipMemcpyDeviceToHost, d.stream),
        "hipMemcpy best");
  Check(hipStreamSynchronize(d.stream), "joint");
  return d.best_host;
}

// Greedy RNN-T: at each frame emit the arg max until it is blank or
// max_symbols tokens came out, then move on. The joint scores a window of
// frames ahead against the current decoder state; the frames before the
// first non-blank one are blank as they would be one at a time, and a
// token changes the state, so the next window starts at its frame.
void Stream::Decode(std::size_t frames) {
  const auto begin = Clock::now();
  const auto blank = static_cast<std::uint32_t>(config_.blank);
  const auto window = static_cast<std::size_t>(kernels::kMaxJointFrames);
  std::size_t frame = 0;
  std::size_t symbols = 0;  // emitted at `frame`
  while (frame < frames) {
    CheckCancelled();
    const std::size_t count = std::min(window, frames - frame);
    const unsigned long long* best = Joint(frame, count);
    std::size_t first = 0;
    while (first < count && kernels::ArgmaxToken(best[first]) == blank)
      ++first;
    if (first == count) {
      frame += count;
      symbols = 0;
      continue;
    }
    if (first > 0) {
      frame += first;
      symbols = 0;
    }
    const std::uint32_t token = kernels::ArgmaxToken(best[first]);
    text_.Push(token);
    DecoderStep(token);
    if (++symbols == config_.max_symbols) {
      ++frame;
      symbols = 0;
    }
  }
  decoder_ms_ += Since(begin);
}

std::vector<float> Stream::DecoderOutput() const {
  std::vector<float> out(config_.decoder_hidden);
  Check(hipMemcpyAsync(out.data(), device_->decoder_out.get(),
                       out.size() * sizeof(float), hipMemcpyDeviceToHost,
                       device_->stream),
        "decoder output");
  Check(hipStreamSynchronize(device_->stream), "decoder output");
  return out;
}

std::vector<float> Stream::JointLogits(std::size_t frame) const {
  const Weights& w = model_.weights();
  StreamDevice& d = *device_;
  const std::size_t width = config_.decoder_hidden;
  if (frame >= last_pass_frames_)
    throw std::out_of_range("frame outside the last pass");
  kernels::JointInput(d.projected.get() + frame * width, d.decoder_out.get(),
                      d.joint_input.get(), 1, static_cast<int>(width),
                      d.stream);
  Linear(d.joint_input.get(), w.head_w, d.logits.get(), 1, width, config_.vocab,
         false, d.stream);
  kernels::AddBias(d.logits.get(), w.head_b, 1, static_cast<int>(config_.vocab),
                   false, d.stream);
  std::vector<float> out(config_.vocab);
  Check(hipMemcpyAsync(out.data(), d.logits.get(), out.size() * sizeof(float),
                       hipMemcpyDeviceToHost, d.stream),
        "logits");
  Check(hipStreamSynchronize(d.stream), "logits");
  return out;
}

}  // namespace gufo::models::nemotron_asr
