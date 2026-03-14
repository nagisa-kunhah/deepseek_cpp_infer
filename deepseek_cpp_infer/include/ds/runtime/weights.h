#pragma once

#include "ds/hf/config.h"
#include "ds/hf/model_loader.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ds::rt {

struct NormWeights {
  const ds::hf::TensorSlice* weight = nullptr;
  std::vector<float> f32;

  bool valid() const { return weight != nullptr; }
  const float* data() const { return f32.data(); }
};

struct LinearWeights {
  const ds::hf::TensorSlice* weight = nullptr;
  const ds::hf::TensorSlice* bias = nullptr;

  bool valid() const { return weight != nullptr; }
};

struct DenseMLPWeights {
  LinearWeights gate_proj;
  LinearWeights up_proj;
  LinearWeights down_proj;

  bool valid() const { return gate_proj.valid() && up_proj.valid() && down_proj.valid(); }
};

struct ExpertWeights {
  DenseMLPWeights ffn;
};

struct AttentionWeights {
  LinearWeights q_proj;
  LinearWeights q_a_proj;
  NormWeights q_a_layernorm;
  LinearWeights q_b_proj;

  LinearWeights kv_a_proj_with_mqa;
  NormWeights kv_a_layernorm;
  LinearWeights kv_b_proj;
  LinearWeights o_proj;

  bool use_q_lora = false;
};

struct MoEWeights {
  LinearWeights gate;
  DenseMLPWeights shared_experts;
  std::vector<ExpertWeights> experts;

  bool valid() const { return gate.valid() && !experts.empty(); }
};

enum class LayerKind {
  Dense,
  MoE,
};

struct DecoderLayerWeights {
  NormWeights input_layernorm;
  NormWeights post_attention_layernorm;
  AttentionWeights attention;
  LayerKind kind = LayerKind::Dense;
  DenseMLPWeights dense_mlp;
  MoEWeights moe;
};

struct GlobalWeights {
  const ds::hf::TensorSlice* embed_tokens = nullptr;
  NormWeights final_norm;
  const ds::hf::TensorSlice* lm_head = nullptr;
};

class WeightRegistry {
 public:
  static WeightRegistry from_model(const ds::hf::DeepSeekConfig& cfg, const ds::hf::LoadedModel& model);

  const GlobalWeights& global() const { return global_; }
  const std::vector<DecoderLayerWeights>& layers() const { return layers_; }

 private:
  GlobalWeights global_;
  std::vector<DecoderLayerWeights> layers_;
};

std::string layer_kind_name(LayerKind kind);

} // namespace ds::rt
