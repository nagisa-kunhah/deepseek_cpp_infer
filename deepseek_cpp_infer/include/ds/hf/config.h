#pragma once

#include <cstdint>
#include <string>

namespace ds::hf {

struct DeepSeekConfig {
  // Only the subset we need to bootstrap model wiring.
  std::string model_type;
  std::int32_t hidden_size = 0;
  std::int32_t intermediate_size = 0;
  std::int32_t num_hidden_layers = 0;
  std::int32_t num_attention_heads = 0;
  std::int32_t num_key_value_heads = 0;
  float rms_norm_eps = 1e-5f;
  std::int32_t vocab_size = 0;
  std::int32_t max_position_embeddings = 0;

  // MoE-ish knobs (names vary per repo; we parse with fallbacks).
  std::int32_t n_experts = 0;
  std::int32_t n_shared_experts = 0;
  std::int32_t moe_top_k = 2;
  std::int32_t moe_intermediate_size = 0;
  std::int32_t first_k_dense_replace = 0;
  std::int32_t moe_layer_freq = 1;
  float routed_scaling_factor = 1.0f;
  bool norm_topk_prob = false;
  std::string scoring_func = "softmax";
  std::string topk_method = "greedy";

  // DeepSeek-V2 attention / MLA-ish knobs.
  std::int32_t q_lora_rank = 0; // 0 means disabled / null in HF config.
  std::int32_t kv_lora_rank = 0;
  std::int32_t qk_nope_head_dim = 0;
  std::int32_t qk_rope_head_dim = 0;
  std::int32_t v_head_dim = 0;

  // Rotary.
  float rope_theta = 10000.0f;
};

DeepSeekConfig load_config_json(const std::string& config_path);

} // namespace ds::hf
