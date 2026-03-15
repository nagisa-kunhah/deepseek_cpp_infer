#include "ds/models/deepseek/cuda_kernels.h"

#if DS_USE_CUDA

namespace ds::rt::cuda::kernels {
namespace {

__global__ void ds_select_topk_kernel(const float* scores, int n, int topk, int normalize, float routed_scale,
                                      int* out_ids, float* out_probs) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  float best_scores[8];
  int best_ids[8];
  for (int i = 0; i < 8; ++i) {
    best_scores[i] = -3.402823466e+38F;
    best_ids[i] = 0;
  }

  for (int i = 0; i < n; ++i) {
    float s = scores[i];
    for (int j = 0; j < topk; ++j) {
      if (s > best_scores[j]) {
        for (int k = topk - 1; k > j; --k) {
          best_scores[k] = best_scores[k - 1];
          best_ids[k] = best_ids[k - 1];
        }
        best_scores[j] = s;
        best_ids[j] = i;
        break;
      }
    }
  }

  float denom = 0.0f;
  for (int i = 0; i < topk; ++i) denom += best_scores[i];
  if (!normalize || denom <= 0.0f) denom = 1.0f;

  for (int i = 0; i < topk; ++i) {
    out_ids[i] = best_ids[i];
    float p = best_scores[i];
    if (normalize) p /= denom;
    out_probs[i] = p * routed_scale;
  }
}

void clear_last_error() { static_cast<void>(cudaGetLastError()); }
bool launched_ok() { return cudaPeekAtLastError() == cudaSuccess; }

} // namespace

bool launch_select_topk(cudaStream_t stream, const float* scores, int n, int topk, bool normalize, float routed_scale,
                        int* out_ids, float* out_probs) {
  clear_last_error();
  ds_select_topk_kernel<<<1, 1, 0, stream>>>(scores, n, topk, normalize ? 1 : 0, routed_scale, out_ids, out_probs);
  return launched_ok();
}

} // namespace ds::rt::cuda::kernels

#endif
