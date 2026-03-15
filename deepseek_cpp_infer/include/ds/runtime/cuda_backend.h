#pragma once

#include "ds/hf/config.h"
#include "ds/hf/model_loader.h"
#include "ds/runtime/weights.h"

#include <cstddef>
#include <cstdint>
namespace ds::rt::cuda {

enum class DeviceBufferSlot {
  Hidden,
  Norm,
  Delta,
  Logits,
  Tmp0,
  Tmp1,
  Tmp2,
  Tmp3,
  Tmp4,
  Tmp5,
  Tmp6,
  Tmp7,
};

struct CudaStats {
  std::size_t embedding_cuda_hits = 0;
  std::size_t embedding_cuda_fallbacks = 0;
  std::size_t linear_cuda_hits = 0;
  std::size_t linear_cuda_fallbacks = 0;
  std::size_t mla_cuda_hits = 0;
  std::size_t mla_cuda_fallbacks = 0;
  std::size_t moe_cuda_hits = 0;
  std::size_t moe_cuda_fallbacks = 0;
  std::size_t cached_weight_hits = 0;
  std::size_t cached_weight_uploads = 0;
  std::size_t cached_weight_bytes = 0;
  std::size_t stream_linear_fallbacks = 0;

  std::size_t fallback_unsupported_shape = 0;
  std::size_t fallback_alloc_failed = 0;
  std::size_t fallback_cuda_init_failed = 0;
  std::size_t fallback_kernel_error = 0;
  std::size_t fallback_cublas_error = 0;
};

class CudaExecutorState;

bool available();
void ensure_initialized();
void reset_stats();
const CudaStats& stats();

CudaExecutorState* create_executor_state(const ds::hf::DeepSeekConfig& cfg, std::size_t max_seq, std::size_t n_layers);
void destroy_executor_state(CudaExecutorState* state);
void reset_executor_state(CudaExecutorState* state);

bool preload_tensor(const ds::hf::TensorSlice& weight);
bool embedding_to_slot(CudaExecutorState* state, const ds::hf::TensorSlice& embedding, std::int32_t token_id,
                       DeviceBufferSlot slot, std::size_t hidden_size);
bool upload_to_slot(CudaExecutorState* state, DeviceBufferSlot slot, const float* src, std::size_t n);
bool download_from_slot(CudaExecutorState* state, DeviceBufferSlot slot, float* dst, std::size_t n);
bool zero_slot(CudaExecutorState* state, DeviceBufferSlot slot, std::size_t n);
bool add_inplace(CudaExecutorState* state, DeviceBufferSlot dst, DeviceBufferSlot src, std::size_t n);

bool linear_to_slot(CudaExecutorState* state, const ds::hf::TensorSlice& weight, DeviceBufferSlot x, std::size_t in,
                    DeviceBufferSlot y, std::size_t out);
bool rmsnorm_to_slot(CudaExecutorState* state, DeviceBufferSlot x, const NormWeights& norm, std::size_t n, float eps,
                     DeviceBufferSlot y);

bool dense_mlp_to_slot(CudaExecutorState* state, const DenseMLPWeights& mlp, std::size_t hidden_size, DeviceBufferSlot x,
                       DeviceBufferSlot y);
bool moe_to_slot(CudaExecutorState* state, const ds::hf::DeepSeekConfig& cfg, const MoEWeights& moe,
                 std::size_t hidden_size, DeviceBufferSlot x, DeviceBufferSlot y);
bool mla_decode_to_slot(CudaExecutorState* state, const ds::hf::DeepSeekConfig& cfg, const AttentionWeights& attn,
                        std::size_t layer_id, std::size_t pos, std::size_t hidden_size, DeviceBufferSlot hidden,
                        DeviceBufferSlot out_hidden);

bool linear_try(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y, std::size_t out);
bool rmsnorm_try(const float* x, const float* w, std::size_t n, float eps, float* y);

} // namespace ds::rt::cuda
