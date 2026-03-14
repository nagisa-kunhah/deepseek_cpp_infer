#pragma once

#include "ds/hf/config.h"
#include "ds/hf/model_loader.h"
#include "ds/runtime/weights.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ds::rt {

void linear(const LinearWeights& linear, const float* x, std::size_t in, float* y, std::size_t out);
void linear(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y, std::size_t out);

void dense_mlp_step_cpu(const DenseMLPWeights& mlp, const float* x, std::size_t hidden_size, float* out);
void moe_step_cpu(const ds::hf::DeepSeekConfig& cfg, const MoEWeights& moe, const float* x, std::size_t hidden_size,
                  float* out);
void lm_head_logits_cpu(const ds::hf::TensorSlice& lm_head_weight, const float* hidden, std::vector<float>* logits);

} // namespace ds::rt
