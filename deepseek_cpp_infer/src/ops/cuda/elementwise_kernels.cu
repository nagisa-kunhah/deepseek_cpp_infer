#include "ds/models/deepseek/cuda_kernels.h"

#if DS_USE_CUDA

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ds::rt::cuda::kernels {
namespace {

constexpr int kThreads = 256;

__global__ void ds_rmsnorm_kernel(const float* x, const float* w, int n, float eps, float* y) {
  __shared__ float partial[kThreads];
  int tid = threadIdx.x;
  float local = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) local += x[i] * x[i];
  partial[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }

  float inv = rsqrtf(partial[0] / static_cast<float>(n) + eps);
  for (int i = tid; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

__global__ void ds_add_inplace_kernel(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void ds_scale_add_kernel(float* dst, const float* src, float scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += scale * src[i];
}

__global__ void ds_silu_mul_kernel(float* gate, const float* up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = gate[i];
  gate[i] = (x / (1.0f + expf(-x))) * up[i];
}

__global__ void ds_rope_inplace_kernel(float* x, int dim, int pos, float theta) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int half = dim / 2;
  if (pair >= half) return;
  int i0 = pair * 2;
  int i1 = i0 + 1;
  float freq = powf(theta, -(static_cast<float>(i0) / static_cast<float>(dim)));
  float angle = static_cast<float>(pos) * freq;
  float c = cosf(angle);
  float s = sinf(angle);
  float a = x[i0];
  float b = x[i1];
  x[i0] = a * c - b * s;
  x[i1] = a * s + b * c;
}

__global__ void ds_f16_row_to_f32_kernel(const std::uint16_t* src, int n, float* dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __half2float(reinterpret_cast<const __half*>(src)[i]);
}

__global__ void ds_bf16_row_to_f32_kernel(const std::uint16_t* src, int n, float* dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(src)[i]);
}

__global__ void ds_f32_to_f16_kernel(const float* src, int n, std::uint16_t* dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  reinterpret_cast<__half*>(dst)[i] = __float2half_rn(src[i]);
}

__global__ void ds_f32_to_bf16_kernel(const float* src, int n, std::uint16_t* dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  reinterpret_cast<__nv_bfloat16*>(dst)[i] = __float2bfloat16(src[i]);
}

void clear_last_error() { static_cast<void>(cudaGetLastError()); }
bool launched_ok() { return cudaPeekAtLastError() == cudaSuccess; }

} // namespace

bool launch_rmsnorm(cudaStream_t stream, const float* x, const float* w, int n, float eps, float* y) {
  clear_last_error();
  ds_rmsnorm_kernel<<<1, kThreads, 0, stream>>>(x, w, n, eps, y);
  return launched_ok();
}

bool launch_add_inplace(cudaStream_t stream, float* dst, const float* src, int n) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  ds_add_inplace_kernel<<<grid, kThreads, 0, stream>>>(dst, src, n);
  return launched_ok();
}

bool launch_scale_add(cudaStream_t stream, float* dst, const float* src, float scale, int n) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  ds_scale_add_kernel<<<grid, kThreads, 0, stream>>>(dst, src, scale, n);
  return launched_ok();
}

bool launch_silu_mul(cudaStream_t stream, float* gate, const float* up, int n) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  ds_silu_mul_kernel<<<grid, kThreads, 0, stream>>>(gate, up, n);
  return launched_ok();
}

bool launch_rope_inplace(cudaStream_t stream, float* x, int dim, int pos, float theta) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>(((dim / 2) + kThreads - 1) / kThreads);
  ds_rope_inplace_kernel<<<grid, kThreads, 0, stream>>>(x, dim, pos, theta);
  return launched_ok();
}

bool launch_f16_row_to_f32(cudaStream_t stream, const std::uint16_t* src, int n, float* dst) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  ds_f16_row_to_f32_kernel<<<grid, kThreads, 0, stream>>>(src, n, dst);
  return launched_ok();
}

bool launch_bf16_row_to_f32(cudaStream_t stream, const std::uint16_t* src, int n, float* dst) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  ds_bf16_row_to_f32_kernel<<<grid, kThreads, 0, stream>>>(src, n, dst);
  return launched_ok();
}

bool launch_f32_to_f16(cudaStream_t stream, const float* src, int n, std::uint16_t* dst) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  ds_f32_to_f16_kernel<<<grid, kThreads, 0, stream>>>(src, n, dst);
  return launched_ok();
}

bool launch_f32_to_bf16(cudaStream_t stream, const float* src, int n, std::uint16_t* dst) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  ds_f32_to_bf16_kernel<<<grid, kThreads, 0, stream>>>(src, n, dst);
  return launched_ok();
}

} // namespace ds::rt::cuda::kernels

#endif
