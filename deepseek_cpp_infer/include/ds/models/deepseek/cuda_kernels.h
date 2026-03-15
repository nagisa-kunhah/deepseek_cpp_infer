#pragma once

#if DS_USE_CUDA

#include <cstdint>
#include <cuda_runtime_api.h>

namespace ds::rt::cuda::kernels {

bool launch_rmsnorm(cudaStream_t stream, const float* x, const float* w, int n, float eps, float* y);
bool launch_add_inplace(cudaStream_t stream, float* dst, const float* src, int n);
bool launch_scale_add(cudaStream_t stream, float* dst, const float* src, float scale, int n);
bool launch_silu_mul(cudaStream_t stream, float* gate, const float* up, int n);
bool launch_rope_inplace(cudaStream_t stream, float* x, int dim, int pos, float theta);
bool launch_scores(cudaStream_t stream, const float* q, const float* k_base, int q_head, int seq_len, int key_stride,
                   float scale, float* scores);
bool launch_softmax(cudaStream_t stream, float* x, int n);
bool launch_weighted_sum(cudaStream_t stream, const float* scores, const float* v_base, int seq_len, int v_head,
                         int v_stride, float* out);
bool launch_select_topk(cudaStream_t stream, const float* scores, int n, int topk, bool normalize, float routed_scale,
                        int* out_ids, float* out_probs);
bool launch_f16_row_to_f32(cudaStream_t stream, const std::uint16_t* src, int n, float* dst);
bool launch_bf16_row_to_f32(cudaStream_t stream, const std::uint16_t* src, int n, float* dst);
bool launch_f32_to_f16(cudaStream_t stream, const float* src, int n, std::uint16_t* dst);
bool launch_f32_to_bf16(cudaStream_t stream, const float* src, int n, std::uint16_t* dst);

} // namespace ds::rt::cuda::kernels

#endif
