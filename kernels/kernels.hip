// Kernels of the encoder and transducer that model.cpp ports from
// transformers' Nemotron 3.5 ASR (Apache-2.0, see licenses/transformers.txt),
// itself a port of NVIDIA NeMo's FastConformer and RNN-T (Apache-2.0) whose
// relative-position attention NeMo adopts from ESPnet (Apache-2.0; both in
// licenses/NeMo.txt).
#include "kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace gufo::models::nemotron_asr::kernels {
namespace {

constexpr int kGroup = 32;  // lanes that reduce together (wave32 or half a wave64)
constexpr int kHeadDim = 128;
constexpr int kMaxKeys = 256;

__device__ float GroupSum(float value) {
  for (int offset = kGroup / 2; offset > 0; offset /= 2)
    value += __shfl_xor(value, offset, kGroup);
  return value;
}

__device__ float GroupMax(float value) {
  for (int offset = kGroup / 2; offset > 0; offset /= 2)
    value = fmaxf(value, __shfl_xor(value, offset, kGroup));
  return value;
}

__device__ float Sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

/// Eight float16 values as floats.
__device__ void Unpack(const uint4& raw, float (&out)[8]) {
  const auto* pairs = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 pair = __half22float2(pairs[i]);
    out[2 * i] = pair.x;
    out[2 * i + 1] = pair.y;
  }
}

__device__ float Dot8(const float (&w)[8], const float4& a, const float4& b,
                      float sum) {
  sum = fmaf(w[0], a.x, sum);
  sum = fmaf(w[1], a.y, sum);
  sum = fmaf(w[2], a.z, sum);
  sum = fmaf(w[3], a.w, sum);
  sum = fmaf(w[4], b.x, sum);
  sum = fmaf(w[5], b.y, sum);
  sum = fmaf(w[6], b.z, sum);
  sum = fmaf(w[7], b.w, sum);
  return sum;
}

__device__ void Store(float* out, std::int64_t index, float value) {
  out[index] = value;
}

__device__ void Store(__half* out, std::int64_t index, float value) {
  out[index] = __float2half(value);
}

// One thread per channel, one block per output row and kStemFrequencies
// output frequencies: the block stages the three mel rows' columns it
// reads in LDS, each thread keeps its channel's nine weights.
constexpr int kStemFrequencies = 8;
constexpr int kStemColumns = 2 * kStemFrequencies + 1;

__global__ void SubsampleInputKernel(const float* mel, int bins,
                                     const float* weight, const float* bias,
                                     int channels, float* out, int out_freq) {
  __shared__ float rows[3][kStemColumns];
  const int t = blockIdx.x;
  const int first = blockIdx.y * kStemFrequencies;
  for (int i = threadIdx.x; i < 3 * kStemColumns; i += blockDim.x) {
    const int dt = i / kStemColumns;
    const int in_f = 2 * first - 2 + i % kStemColumns;
    rows[dt][i % kStemColumns] =
        in_f >= 0 && in_f < bins
            ? mel[static_cast<std::int64_t>(2 * t + dt) * bins + in_f]
            : 0.0F;
  }
  __syncthreads();
  const int c = threadIdx.x;
  if (c >= channels)
    return;
  float w[9];
#pragma unroll
  for (int i = 0; i < 9; ++i)
    w[i] = weight[c * 9 + i];
  const float b = bias[c];
  for (int j = 0; j < kStemFrequencies; ++j) {
    const int f = first + j;
    if (f >= out_freq)
      break;
    // The same terms in the same order as a per-element loop that skips
    // the padding columns (whose zeros would add nothing).
    float sum = b;
#pragma unroll
    for (int dt = 0; dt < 3; ++dt)
#pragma unroll
      for (int df = 0; df < 3; ++df) {
        const int in_f = 2 * f + df - 2;
        if (in_f < 0 || in_f >= bins)
          continue;
        sum = fmaf(w[dt * 3 + df], rows[dt][2 * j + df], sum);
      }
    out[(static_cast<std::int64_t>(t) * out_freq + f) * channels + c] =
        fmaxf(sum, 0.0F);
  }
}

__global__ void SubsampleDepthwiseKernel(const float* in, int in_freq,
                                         const float* weight,
                                         const float* bias, int channels,
                                         float* out, int out_rows,
                                         int out_freq) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total =
      static_cast<std::int64_t>(out_rows) * out_freq * channels;
  if (index >= total)
    return;
  const int c = static_cast<int>(index % channels);
  const int f = static_cast<int>((index / channels) % out_freq);
  const int t = static_cast<int>(index / (static_cast<std::int64_t>(channels) *
                                          out_freq));
  float sum = bias[c];
  for (int dt = 0; dt < 3; ++dt)
    for (int df = 0; df < 3; ++df) {
      const int in_f = 2 * f + df - 2;
      if (in_f < 0 || in_f >= in_freq)
        continue;
      sum += weight[c * 9 + dt * 3 + df] *
             in[(static_cast<std::int64_t>(2 * t + dt) * in_freq + in_f) *
                    channels +
                c];
    }
  out[index] = sum;
}

__global__ void AddBiasKernel(float* x, const float* bias, int rows, int cols,
                              bool relu) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::int64_t>(rows) * cols)
    return;
  const float value = x[index] + bias[index % cols];
  x[index] = relu ? fmaxf(value, 0.0F) : value;
}

// One block of 256 threads per row.
template<typename Out, bool kSilu>
__global__ void LayerNormKernel(const float* x, const float* weight,
                                const float* bias, Out* y, int cols) {
  __shared__ float partial[256 / kGroup];
  const float* row = x + static_cast<std::int64_t>(blockIdx.x) * cols;
  Out* out = y + static_cast<std::int64_t>(blockIdx.x) * cols;
  const int lane = threadIdx.x % kGroup;
  const int group = threadIdx.x / kGroup;
  const int groups = blockDim.x / kGroup;

  float sum = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    sum += row[c];
  sum = GroupSum(sum);
  if (lane == 0)
    partial[group] = sum;
  __syncthreads();
  float total = 0.0F;
  for (int g = 0; g < groups; ++g)
    total += partial[g];
  const float mean = total / static_cast<float>(cols);
  __syncthreads();

  float squares = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float d = row[c] - mean;
    squares += d * d;
  }
  squares = GroupSum(squares);
  if (lane == 0)
    partial[group] = squares;
  __syncthreads();
  float variance = 0.0F;
  for (int g = 0; g < groups; ++g)
    variance += partial[g];
  variance /= static_cast<float>(cols);
  const float inverse = rsqrtf(variance + 1e-5F);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float value = (row[c] - mean) * inverse * weight[c] + bias[c];
    if constexpr (kSilu)
      value *= Sigmoid(value);
    Store(out, c, value);
  }
}

__global__ void SiluHalfKernel(const float* x, __half* y, std::int64_t n) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n)
    y[index] = __float2half(x[index] * Sigmoid(x[index]));
}

__global__ void ToHalfKernel(const float* x, __half* y, std::int64_t n) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n)
    y[index] = __float2half(x[index]);
}

// One block of 128 threads (one per head dimension) per (query, head).
template<typename Out>
__global__ void RelPosAttentionKernel(const float* q, const float* cache_keys,
                                      const float* cache_values,
                                      const float* pos, const float* bias_u,
                                      const float* bias_v, Out* out,
                                      AttentionShape shape) {
  __shared__ float scores[kMaxKeys];
  __shared__ float reduce[2];
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  const int width = shape.heads * kHeadDim;
  const int column = h * kHeadDim + d;

  const std::int64_t absolute = shape.first_query + t;
  const std::int64_t chunk = absolute / shape.chunk;
  std::int64_t begin = chunk * shape.chunk - shape.left;
  std::int64_t end = (chunk + 1) * shape.chunk;
  if (begin < shape.first_key)
    begin = shape.first_key;
  if (end > shape.first_query + shape.queries)
    end = shape.first_query + shape.queries;
  const int count = static_cast<int>(end - begin);
  // Key `key` is cache row key - first_key before the pass, else the pass's
  // qkv row key - first_query (keys at column width, values at 2 * width).
  const auto key_row = [&](std::int64_t key) {
    return key < shape.first_query
               ? cache_keys + (key - shape.first_key) * width
               : q + (key - shape.first_query) * shape.q_stride + width;
  };
  const auto value_row = [&](std::int64_t key) {
    return key < shape.first_query
               ? cache_values + (key - shape.first_key) * width
               : q + (key - shape.first_query) * shape.q_stride + 2 * width;
  };

  // Each group of 32 lanes scores keys in turn; lane l holds dimensions
  // l, l + 32, l + 64 and l + 96 of its head.
  const int lane = d % kGroup;
  const int group = d / kGroup;
  const int groups = kHeadDim / kGroup;
  float qu4[4], qv4[4];
  for (int i = 0; i < 4; ++i) {
    const int dim = h * kHeadDim + lane + i * kGroup;
    const float value =
        q[static_cast<std::int64_t>(t) * shape.q_stride + dim];
    qu4[i] = value + bias_u[dim];
    qv4[i] = value + bias_v[dim];
  }
  for (int j = group; j < count; j += groups) {
    const std::int64_t key = begin + j;
    const float* k = key_row(key) + h * kHeadDim;
    const float* p = pos +
                     static_cast<std::int64_t>(shape.pos_zero +
                                               (absolute - key)) *
                         width +
                     h * kHeadDim;
    float partial = 0.0F;
    for (int i = 0; i < 4; ++i)
      partial += qu4[i] * k[lane + i * kGroup] + qv4[i] * p[lane + i * kGroup];
    partial = GroupSum(partial);
    if (lane == 0)
      scores[j] = partial * shape.scale;
  }
  __syncthreads();
  if (group == 0) {
    float best = -INFINITY;
    for (int j = lane; j < count; j += kGroup)
      best = fmaxf(best, scores[j]);
    best = GroupMax(best);
    float sum = 0.0F;
    for (int j = lane; j < count; j += kGroup) {
      const float e = expf(scores[j] - best);
      scores[j] = e;
      sum += e;
    }
    sum = GroupSum(sum);
    if (lane == 0)
      reduce[0] = 1.0F / sum;
  }
  __syncthreads();
  const float inverse = reduce[0];
  float acc = 0.0F;
  for (int j = 0; j < count; ++j)
    acc += scores[j] * value_row(begin + j)[column];
  Store(out, static_cast<std::int64_t>(t) * width + column, acc * inverse);
}

// The last `keep` rows of [cache (cached rows) | the pass's rows], written
// over the cache: one thread per column walks the rows upwards, so every
// cache row is read before it is overwritten.
__global__ void UpdateKeyValueCacheKernel(float* keys, float* values,
                                          const float* qkv, int stride,
                                          int cached, int rows, int keep,
                                          int width) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= width)
    return;
  const int first = cached + rows - keep;
  for (int i = 0; i < keep; ++i) {
    const int at = first + i;
    float k, v;
    if (at < cached) {
      k = keys[static_cast<std::int64_t>(at) * width + c];
      v = values[static_cast<std::int64_t>(at) * width + c];
    } else {
      const float* row = qkv + static_cast<std::int64_t>(at - cached) * stride;
      k = row[width + c];
      v = row[2 * width + c];
    }
    keys[static_cast<std::int64_t>(i) * width + c] = k;
    values[static_cast<std::int64_t>(i) * width + c] = v;
  }
}

__device__ float GluAt(const float* in, const float* cache, int history,
                       int index, int channels, int c) {
  if (index < 0)
    return cache[static_cast<std::int64_t>(history + index) * channels + c];
  const float* row = in + static_cast<std::int64_t>(index) * 2 * channels;
  return row[c] * Sigmoid(row[channels + c]);
}

// out[r][c] = sum_j weight[c][j] * glu[r + j - (kernel - 1)][c], the GLU
// of `in` before row 0 taken from `cache` (kernel - 1 rows).
__global__ void GluDepthwiseKernel(const float* in, const float* cache,
                                   const float* weight, float* out, int rows,
                                   int channels, int kernel) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::int64_t>(rows) * channels)
    return;
  const int r = static_cast<int>(index / channels);
  const int c = static_cast<int>(index % channels);
  const int history = kernel - 1;
  float sum = 0.0F;
  for (int j = 0; j < kernel; ++j)
    sum += weight[c * kernel + j] *
           GluAt(in, cache, history, r + j - history, channels, c);
  out[index] = sum;
}

// The conv cache becomes the last kernel - 1 GLU rows of [cache | pass].
__global__ void UpdateConvCacheKernel(const float* in, float* cache, int rows,
                                      int channels, int kernel) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  const int history = kernel - 1;
  for (int i = 0; i < history; ++i)
    cache[static_cast<std::int64_t>(i) * channels + c] =
        GluAt(in, cache, history, rows - history + i, channels, c);
}

// Eight rows per block of 256 threads: 32 lanes per row, lane l summing
// the float16 weights of columns 8l..8l + 7, 8(l + 32).. of concat(x1, x2)
// in that order.
__global__ void GemvKernel(const __half* weight, const float* x1, int k1,
                           const float* x2, int k2, const float* bias,
                           float* y, int rows) {
  const int row = blockIdx.x * (blockDim.x / kGroup) + threadIdx.x / kGroup;
  const int lane = threadIdx.x % kGroup;
  if (row >= rows)
    return;
  const int octets1 = k1 / 8;
  const int octets = octets1 + k2 / 8;
  const auto* w = reinterpret_cast<const uint4*>(
      weight + static_cast<std::int64_t>(row) * (k1 + k2));
  const auto* a = reinterpret_cast<const float4*>(x1);
  const auto* b = reinterpret_cast<const float4*>(x2);
  float sum = 0.0F;
#pragma unroll 4
  for (int o = lane; o < octets; o += kGroup) {
    float v[8];
    Unpack(w[o], v);
    const float4* x = o < octets1 ? a + 2 * o : b + 2 * (o - octets1);
    sum = Dot8(v, x[0], x[1], sum);
  }
  sum = GroupSum(sum);
  if (lane == 0)
    y[row] = sum + (bias != nullptr ? bias[row] : 0.0F);
}

__global__ void LstmCellKernel(const float* gates, float* h, float* c,
                               int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden)
    return;
  const float input = Sigmoid(gates[i]);
  const float forget = Sigmoid(gates[hidden + i]);
  const float cell = tanhf(gates[2 * hidden + i]);
  const float output = Sigmoid(gates[3 * hidden + i]);
  const float next = forget * c[i] + input * cell;
  c[i] = next;
  h[i] = output * tanhf(next);
}

__device__ unsigned long long Pack(float value, int index) {
  unsigned int bits = __float_as_uint(value);
  bits = (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
  return (static_cast<unsigned long long>(bits) << 32) |
         static_cast<unsigned long long>(0xFFFFFFFFU -
                                         static_cast<unsigned int>(index));
}

// relu(encoder[f] + decoder) as float16, the joint's input per frame.
__global__ void JointInputKernel(const float* encoder, const float* decoder,
                                 __half* out, int width, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = __float2half(fmaxf(encoder[i] + decoder[i % width], 0.0F));
}

// One block per frame: the packed arg max of logits + bias over the row.
__global__ void ArgmaxRowsKernel(const float* logits, const float* bias,
                                 int columns, unsigned long long* best) {
  __shared__ unsigned long long partial[256 / kGroup];
  const float* row = logits + static_cast<std::int64_t>(blockIdx.x) * columns;
  unsigned long long mine = 0;
  for (int c = threadIdx.x; c < columns; c += blockDim.x) {
    const unsigned long long packed = Pack(row[c] + bias[c], c);
    mine = packed > mine ? packed : mine;
  }
  for (int offset = kGroup / 2; offset > 0; offset /= 2) {
    const unsigned long long other = __shfl_xor(mine, offset, kGroup);
    mine = other > mine ? other : mine;
  }
  if (threadIdx.x % kGroup == 0)
    partial[threadIdx.x / kGroup] = mine;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int g = 1; g < static_cast<int>(blockDim.x) / kGroup; ++g)
      mine = partial[g] > mine ? partial[g] : mine;
    best[blockIdx.x] = mine;
  }
}

int Blocks(std::int64_t n, int threads) {
  return static_cast<int>((n + threads - 1) / threads);
}

}  // namespace

void SubsampleInput(const float* mel, int bins, const float* weight,
                    const float* bias, int channels, float* out, int out_rows,
                    int out_freq, hipStream_t stream) {
  if (out_rows == 0 || out_freq == 0)
    return;
  const dim3 grid(static_cast<unsigned>(out_rows),
                  static_cast<unsigned>((out_freq + kStemFrequencies - 1) /
                                        kStemFrequencies));
  SubsampleInputKernel<<<grid, channels, 0, stream>>>(
      mel, bins, weight, bias, channels, out, out_freq);
}

void SubsampleDepthwise(const float* in, int in_freq, const float* weight,
                        const float* bias, int channels, float* out,
                        int out_rows, int out_freq, hipStream_t stream) {
  const std::int64_t n =
      static_cast<std::int64_t>(out_rows) * out_freq * channels;
  if (n == 0)
    return;
  SubsampleDepthwiseKernel<<<Blocks(n, 256), 256, 0, stream>>>(
      in, in_freq, weight, bias, channels, out, out_rows, out_freq);
}

void AddBias(float* x, const float* bias, int rows, int cols, bool relu,
             hipStream_t stream) {
  const std::int64_t n = static_cast<std::int64_t>(rows) * cols;
  if (n == 0)
    return;
  AddBiasKernel<<<Blocks(n, 256), 256, 0, stream>>>(x, bias, rows, cols,
                                                     relu);
}

void LayerNorm(const float* x, const float* weight, const float* bias,
               float* y, int rows, int cols, hipStream_t stream) {
  if (rows == 0)
    return;
  LayerNormKernel<float, false><<<rows, 256, 0, stream>>>(x, weight, bias, y,
                                                          cols);
}

void LayerNormHalf(const float* x, const float* weight, const float* bias,
                   std::uint16_t* y, int rows, int cols, bool silu,
                   hipStream_t stream) {
  if (rows == 0)
    return;
  auto* out = reinterpret_cast<__half*>(y);
  if (silu)
    LayerNormKernel<__half, true><<<rows, 256, 0, stream>>>(x, weight, bias,
                                                            out, cols);
  else
    LayerNormKernel<__half, false><<<rows, 256, 0, stream>>>(x, weight, bias,
                                                             out, cols);
}

void SiluHalf(const float* x, std::uint16_t* y, std::int64_t n,
              hipStream_t stream) {
  if (n == 0)
    return;
  SiluHalfKernel<<<Blocks(n, 256), 256, 0, stream>>>(
      x, reinterpret_cast<__half*>(y), n);
}

void ToHalf(const float* x, std::uint16_t* y, std::int64_t n,
            hipStream_t stream) {
  if (n == 0)
    return;
  ToHalfKernel<<<Blocks(n, 256), 256, 0, stream>>>(
      x, reinterpret_cast<__half*>(y), n);
}

void RelPosAttentionHalf(const float* qkv, const float* cache_keys,
                         const float* cache_values, const float* pos,
                         const float* bias_u, const float* bias_v,
                         std::uint16_t* out, const AttentionShape& shape,
                         hipStream_t stream) {
  if (shape.queries == 0)
    return;
  RelPosAttentionKernel<__half><<<dim3(shape.queries, shape.heads), kHeadDim,
                                  0, stream>>>(
      qkv, cache_keys, cache_values, pos, bias_u, bias_v,
      reinterpret_cast<__half*>(out), shape);
}

void UpdateKeyValueCache(float* keys, float* values, const float* qkv,
                         int stride, int cached, int rows, int keep, int width,
                         hipStream_t stream) {
  if (keep == 0)
    return;
  UpdateKeyValueCacheKernel<<<Blocks(width, 256), 256, 0, stream>>>(
      keys, values, qkv, stride, cached, rows, keep, width);
}

void GluDepthwise(const float* in, const float* cache, const float* weight,
                  float* out, int rows, int channels, int kernel,
                  hipStream_t stream) {
  const std::int64_t n = static_cast<std::int64_t>(rows) * channels;
  if (n == 0)
    return;
  GluDepthwiseKernel<<<Blocks(n, 256), 256, 0, stream>>>(
      in, cache, weight, out, rows, channels, kernel);
}

void UpdateConvCache(const float* in, float* cache, int rows, int channels,
                     int kernel, hipStream_t stream) {
  UpdateConvCacheKernel<<<Blocks(channels, 256), 256, 0, stream>>>(
      in, cache, rows, channels, kernel);
}

void Gemv(const std::uint16_t* weight, const float* x1, int k1,
          const float* x2, int k2, const float* bias, float* y, int rows,
          hipStream_t stream) {
  constexpr int kThreads = 256;
  GemvKernel<<<Blocks(rows, kThreads / kGroup), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(weight), x1, k1, x2, k2, bias, y, rows);
}

void LstmCell(const float* gates, float* h, float* c, int hidden,
              hipStream_t stream) {
  LstmCellKernel<<<Blocks(hidden, 256), 256, 0, stream>>>(gates, h, c,
                                                          hidden);
}

void JointInput(const float* encoder, const float* decoder,
                std::uint16_t* out, int frames, int width,
                hipStream_t stream) {
  const int count = frames * width;
  if (count == 0)
    return;
  JointInputKernel<<<Blocks(count, 256), 256, 0, stream>>>(
      encoder, decoder, reinterpret_cast<__half*>(out), width, count);
}

void ArgmaxRows(const float* logits, const float* bias, int rows, int columns,
                unsigned long long* best, hipStream_t stream) {
  if (rows == 0)
    return;
  ArgmaxRowsKernel<<<rows, 256, 0, stream>>>(logits, bias, columns, best);
}

}  // namespace gufo::models::nemotron_asr::kernels
