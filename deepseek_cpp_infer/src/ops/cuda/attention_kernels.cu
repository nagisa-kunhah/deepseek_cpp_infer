#include "ds/runtime/cuda_kernels.h"

#if DS_USE_CUDA

namespace ds::rt::cuda::kernels {
namespace {

constexpr int kThreads = 256;

__global__ void ds_scores_kernel(const float* q, const float* k_base, int q_head, int seq_len, int key_stride,
                                 float scale, float* scores) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= seq_len) return;
  const float* k = k_base + t * key_stride;
  float acc = 0.0f;
  for (int i = 0; i < q_head; ++i) acc += q[i] * k[i];
  scores[t] = acc * scale;
}

__global__ void ds_softmax_kernel(float* x, int n) {
  __shared__ float ssum;
  if (threadIdx.x == 0) {
    float m = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > m) m = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      x[i] = expf(x[i] - m);
      sum += x[i];
    }
    ssum = sum;
  }
  __syncthreads();
  float inv = ssum > 0.0f ? (1.0f / ssum) : 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] *= inv;
}

__global__ void ds_weighted_sum_kernel(const float* scores, const float* v_base, int seq_len, int v_head, int v_stride,
                                       float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= v_head) return;
  float acc = 0.0f;
  for (int t = 0; t < seq_len; ++t) acc += scores[t] * v_base[t * v_stride + i];
  out[i] = acc;
}

void clear_last_error() { static_cast<void>(cudaGetLastError()); }
bool launched_ok() { return cudaPeekAtLastError() == cudaSuccess; }

} // namespace

bool launch_scores(cudaStream_t stream, const float* q, const float* k_base, int q_head, int seq_len, int key_stride,
                   float scale, float* scores) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((seq_len + kThreads - 1) / kThreads);
  ds_scores_kernel<<<grid, kThreads, 0, stream>>>(q, k_base, q_head, seq_len, key_stride, scale, scores);
  return launched_ok();
}

bool launch_softmax(cudaStream_t stream, float* x, int n) {
  clear_last_error();
  ds_softmax_kernel<<<1, kThreads, 0, stream>>>(x, n);
  return launched_ok();
}

bool launch_weighted_sum(cudaStream_t stream, const float* scores, const float* v_base, int seq_len, int v_head,
                         int v_stride, float* out) {
  clear_last_error();
  const unsigned grid = static_cast<unsigned>((v_head + kThreads - 1) / kThreads);
  ds_weighted_sum_kernel<<<grid, kThreads, 0, stream>>>(scores, v_base, seq_len, v_head, v_stride, out);
  return launched_ok();
}

} // namespace ds::rt::cuda::kernels

#endif
