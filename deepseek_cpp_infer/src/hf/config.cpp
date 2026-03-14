#include "ds/hf/config.h"

#include "ds/util/fs.h"
#include "ds/util/json.h"

#include <stdexcept>

namespace ds::hf {

DeepSeekConfig load_config_json(const std::string& config_path) {
  const auto txt = ds::util::read_text_file(config_path);
  const auto j = ds::util::parse_json(txt);

  DeepSeekConfig c;
  c.model_type = ds::util::get_string_or(j, "model_type", "");
  c.hidden_size = static_cast<std::int32_t>(ds::util::get_int_or(j, "hidden_size", 0));
  c.intermediate_size = static_cast<std::int32_t>(ds::util::get_int_or(j, "intermediate_size", 0));
  c.num_hidden_layers = static_cast<std::int32_t>(ds::util::get_int_or(j, "num_hidden_layers", 0));
  c.num_attention_heads = static_cast<std::int32_t>(ds::util::get_int_or(j, "num_attention_heads", 0));
  c.num_key_value_heads = static_cast<std::int32_t>(ds::util::get_int_or(j, "num_key_value_heads", 0));
  c.rms_norm_eps = static_cast<float>(ds::util::get_double_or(j, "rms_norm_eps", 1e-5));
  c.vocab_size = static_cast<std::int32_t>(ds::util::get_int_or(j, "vocab_size", 0));
  c.max_position_embeddings = static_cast<std::int32_t>(ds::util::get_int_or(j, "max_position_embeddings", 0));
  c.rope_theta = static_cast<float>(ds::util::get_double_or(j, "rope_theta", 10000.0));

  // MoE key names are not fully standardized across repos. We try a few likely ones.
  c.n_experts = static_cast<std::int32_t>(ds::util::get_int_or(j, "n_routed_experts", c.n_experts));
  c.n_experts = static_cast<std::int32_t>(ds::util::get_int_or(j, "num_experts", c.n_experts));
  c.n_experts = static_cast<std::int32_t>(ds::util::get_int_or(j, "n_experts", c.n_experts));
  c.n_shared_experts = static_cast<std::int32_t>(ds::util::get_int_or(j, "n_shared_experts", c.n_shared_experts));

  c.moe_top_k = static_cast<std::int32_t>(ds::util::get_int_or(j, "moe_top_k", c.moe_top_k));
  c.moe_top_k = static_cast<std::int32_t>(ds::util::get_int_or(j, "num_experts_per_tok", c.moe_top_k));

  c.moe_intermediate_size = static_cast<std::int32_t>(ds::util::get_int_or(j, "moe_intermediate_size", c.moe_intermediate_size));
  c.moe_intermediate_size = static_cast<std::int32_t>(ds::util::get_int_or(j, "intermediate_size", c.moe_intermediate_size));
  c.first_k_dense_replace = static_cast<std::int32_t>(ds::util::get_int_or(j, "first_k_dense_replace", c.first_k_dense_replace));
  c.moe_layer_freq = static_cast<std::int32_t>(ds::util::get_int_or(j, "moe_layer_freq", c.moe_layer_freq));
  c.routed_scaling_factor = static_cast<float>(ds::util::get_double_or(j, "routed_scaling_factor", c.routed_scaling_factor));
  c.norm_topk_prob = ds::util::get_int_or(j, "norm_topk_prob", c.norm_topk_prob ? 1 : 0) != 0;
  c.scoring_func = ds::util::get_string_or(j, "scoring_func", c.scoring_func);
  c.topk_method = ds::util::get_string_or(j, "topk_method", c.topk_method);

  const auto* q_lora = j.find("q_lora_rank");
  if (q_lora && !q_lora->is_null()) {
    c.q_lora_rank = static_cast<std::int32_t>(q_lora->as_int());
  }
  c.kv_lora_rank = static_cast<std::int32_t>(ds::util::get_int_or(j, "kv_lora_rank", c.kv_lora_rank));
  c.qk_nope_head_dim = static_cast<std::int32_t>(ds::util::get_int_or(j, "qk_nope_head_dim", c.qk_nope_head_dim));
  c.qk_rope_head_dim = static_cast<std::int32_t>(ds::util::get_int_or(j, "qk_rope_head_dim", c.qk_rope_head_dim));
  c.v_head_dim = static_cast<std::int32_t>(ds::util::get_int_or(j, "v_head_dim", c.v_head_dim));

  if (c.hidden_size <= 0 || c.num_hidden_layers <= 0 || c.vocab_size <= 0) {
    throw std::runtime_error("config.json missing required fields (hidden_size/num_hidden_layers/vocab_size)");
  }

  return c;
}

} // namespace ds::hf
