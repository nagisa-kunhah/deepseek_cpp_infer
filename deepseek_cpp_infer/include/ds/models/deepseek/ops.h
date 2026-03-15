#pragma once

#include "ds/hf/model_loader.h"
#include "ds/models/deepseek/config.h"
#include "ds/models/deepseek/weights.h"
#include "ds/runtime/backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ds::rt {

void linear(const LinearWeights& linear, const float* x, std::size_t in, float* y, std::size_t out);
void linear(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y, std::size_t out);
void linear_backend(BackendKind backend, const LinearWeights& linear, const float* x, std::size_t in, float* y,
                    std::size_t out);
void linear_backend(BackendKind backend, const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y,
                    std::size_t out);

void rmsnorm_backend(BackendKind backend, const float* x, const float* w, std::size_t n, float eps, float* y);
void dense_mlp_step(BackendKind backend, const DenseMLPWeights& mlp, const float* x, std::size_t hidden_size, float* out);
void moe_step(BackendKind backend, const ds::hf::DeepSeekConfig& cfg, const MoEWeights& moe, const float* x,
              std::size_t hidden_size, float* out);
void lm_head_logits(BackendKind backend, const ds::hf::TensorSlice& lm_head_weight, const float* hidden,
                    std::vector<float>* logits);

} // namespace ds::rt
