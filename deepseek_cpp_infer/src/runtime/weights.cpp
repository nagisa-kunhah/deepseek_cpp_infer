#include "ds/runtime/weights.h"

#include "ds/hf/decode.h"

#include <stdexcept>

namespace ds::rt {
namespace {

const ds::hf::TensorSlice* find_required(const ds::hf::LoadedModel& model, const std::string& name) {
  auto it = model.tensors.find(name);
  if (it == model.tensors.end()) throw std::runtime_error("missing tensor: " + name);
  return &it->second;
}

const ds::hf::TensorSlice* find_optional(const ds::hf::LoadedModel& model, const std::string& name) {
  auto it = model.tensors.find(name);
  if (it == model.tensors.end()) return nullptr;
  return &it->second;
}

NormWeights required_norm(const ds::hf::LoadedModel& model, const std::string& name) {
  NormWeights out;
  out.weight = find_required(model, name);
  out.f32 = ds::hf::decode_to_f32(*out.weight);
  return out;
}

NormWeights optional_norm(const ds::hf::LoadedModel& model, const std::string& name) {
  NormWeights out;
  out.weight = find_optional(model, name);
  if (out.weight != nullptr) out.f32 = ds::hf::decode_to_f32(*out.weight);
  return out;
}

LinearWeights required_linear(const ds::hf::LoadedModel& model, const std::string& weight_name,
                              const std::string& bias_name = std::string()) {
  LinearWeights out;
  out.weight = find_required(model, weight_name);
  if (!bias_name.empty()) out.bias = find_optional(model, bias_name);
  return out;
}

DenseMLPWeights required_dense_mlp(const ds::hf::LoadedModel& model, const std::string& prefix) {
  DenseMLPWeights out;
  out.gate_proj = required_linear(model, prefix + "gate_proj.weight");
  out.up_proj = required_linear(model, prefix + "up_proj.weight");
  out.down_proj = required_linear(model, prefix + "down_proj.weight");
  return out;
}

} // namespace

DeepSeekWeightRegistry DeepSeekWeightRegistry::from_model(const ds::hf::DeepSeekConfig& cfg, const ds::hf::LoadedModel& model) {
  DeepSeekWeightRegistry registry;
  registry.global_.embed_tokens = find_required(model, "model.embed_tokens.weight");
  registry.global_.final_norm = required_norm(model, "model.norm.weight");
  registry.global_.lm_head = find_required(model, "lm_head.weight");

  registry.layers_.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
  for (std::int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
    auto& layer = registry.layers_[static_cast<std::size_t>(i)];
    const std::string base = "model.layers." + std::to_string(i) + ".";
    const std::string attn = base + "self_attn.";
    const std::string mlp = base + "mlp.";

    layer.input_layernorm = required_norm(model, base + "input_layernorm.weight");
    layer.post_attention_layernorm = required_norm(model, base + "post_attention_layernorm.weight");

    layer.attention.q_proj.weight = find_optional(model, attn + "q_proj.weight");
    layer.attention.q_a_proj.weight = find_optional(model, attn + "q_a_proj.weight");
    layer.attention.q_a_layernorm = optional_norm(model, attn + "q_a_layernorm.weight");
    layer.attention.q_b_proj.weight = find_optional(model, attn + "q_b_proj.weight");
    layer.attention.kv_a_proj_with_mqa = required_linear(model, attn + "kv_a_proj_with_mqa.weight");
    layer.attention.kv_a_layernorm = required_norm(model, attn + "kv_a_layernorm.weight");
    layer.attention.kv_b_proj = required_linear(model, attn + "kv_b_proj.weight");
    layer.attention.o_proj = required_linear(model, attn + "o_proj.weight");
    layer.attention.use_q_lora = layer.attention.q_a_proj.valid() && layer.attention.q_b_proj.valid();

    if (!layer.attention.use_q_lora && !layer.attention.q_proj.valid()) {
      throw std::runtime_error("layer " + std::to_string(i) + " is missing q_proj and q_lora path");
    }

    const auto* expert0 = find_optional(model, mlp + "experts.0.gate_proj.weight");
    if (expert0 == nullptr) {
      layer.kind = LayerKind::Dense;
      layer.dense_mlp = required_dense_mlp(model, mlp);
      continue;
    }

    layer.kind = LayerKind::MoE;
    layer.moe.gate = required_linear(model, mlp + "gate.weight");

    const auto* shared_gate = find_optional(model, mlp + "shared_experts.gate_proj.weight");
    if (shared_gate != nullptr) {
      layer.moe.shared_experts = required_dense_mlp(model, mlp + "shared_experts.");
    }

    for (std::int32_t expert_id = 0; expert_id < cfg.n_experts; ++expert_id) {
      const std::string expert_prefix = mlp + "experts." + std::to_string(expert_id) + ".";
      if (find_optional(model, expert_prefix + "gate_proj.weight") == nullptr) break;
      ExpertWeights expert;
      expert.ffn = required_dense_mlp(model, expert_prefix);
      layer.moe.experts.push_back(std::move(expert));
    }

    if (!layer.moe.valid()) {
      throw std::runtime_error("layer " + std::to_string(i) + " MoE weights are incomplete");
    }
  }

  return registry;
}

std::string layer_kind_name(LayerKind kind) {
  switch (kind) {
    case LayerKind::Dense: return "dense";
    case LayerKind::MoE: return "moe";
  }
  return "unknown";
}

} // namespace ds::rt
