#include "ds/models/deepseek/package.h"

#include "ds/models/deepseek/config.h"
#include "ds/models/deepseek/model.h"
#include "ds/models/deepseek/weights.h"

#include "ds/hf/model_loader.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace ds::models::deepseek {
namespace {

std::string lower_ascii(std::string s) {
  for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

bool has_prefix(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

bool matches(const ds::hf::ModelConfig& config) {
  return lower_ascii(config.model_type).find("deepseek") != std::string::npos;
}

ds::hf::DeepSeekConfig require_deepseek_config(const ds::hf::ModelConfig& config) {
  return ds::hf::parse_deepseek_config(config);
}

void print_config(std::ostream& os, const ds::hf::ModelConfig& config) {
  const auto cfg = require_deepseek_config(config);
  os << "Loaded config:\n";
  os << "  family: deepseek\n";
  os << "  model_type: " << cfg.model_type << "\n";
  os << "  layers: " << cfg.num_hidden_layers << "\n";
  os << "  hidden: " << cfg.hidden_size << "\n";
  os << "  heads: " << cfg.num_attention_heads << "\n";
  os << "  kv_heads: " << cfg.num_key_value_heads << "\n";
  os << "  vocab: " << cfg.vocab_size << "\n";
  os << "  max_pos: " << cfg.max_position_embeddings << "\n";
  os << "  rope_theta: " << cfg.rope_theta << "\n";
  os << "  n_experts: " << cfg.n_experts << "\n";
  os << "  n_shared_experts: " << cfg.n_shared_experts << "\n";
  os << "  moe_top_k: " << cfg.moe_top_k << "\n";
  os << "  kv_lora_rank: " << cfg.kv_lora_rank << "\n";
  os << "  qk_nope_head_dim: " << cfg.qk_nope_head_dim << "\n";
  os << "  qk_rope_head_dim: " << cfg.qk_rope_head_dim << "\n";
  os << "  v_head_dim: " << cfg.v_head_dim << "\n";
}

ds::models::core::IndexInspection inspect_index(const ds::hf::ModelConfig& config,
                                                const std::vector<std::string>& tensor_names) {
  const auto cfg = require_deepseek_config(config);
  std::unordered_set<std::string> keys;
  keys.reserve(tensor_names.size());
  for (const auto& key : tensor_names) keys.emplace(key);

  auto require_key = [&](const std::string& key) {
    if (keys.find(key) == keys.end()) throw std::runtime_error("missing tensor in index: " + key);
  };

  auto any_with_prefix = [&](const std::string& prefix) {
    for (const auto& key : tensor_names) {
      if (has_prefix(key, prefix)) return true;
    }
    return false;
  };

  require_key("lm_head.weight");
  require_key("model.embed_tokens.weight");
  require_key("model.norm.weight");

  ds::models::core::IndexInspection out;
  out.key_count = tensor_names.size();
  out.layer_kinds.resize(static_cast<std::size_t>(cfg.num_hidden_layers));

  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    const auto base = "model.layers." + std::to_string(i) + ".";
    require_key(base + "input_layernorm.weight");
    require_key(base + "post_attention_layernorm.weight");
    require_key(base + "self_attn.q_proj.weight");
    require_key(base + "self_attn.o_proj.weight");
    require_key(base + "self_attn.kv_a_layernorm.weight");
    require_key(base + "self_attn.kv_a_proj_with_mqa.weight");
    require_key(base + "self_attn.kv_b_proj.weight");

    const auto mlp = base + "mlp.";
    const bool is_moe = any_with_prefix(mlp + "experts.");
    out.layer_kinds[static_cast<std::size_t>(i)] = is_moe ? "MoE" : "dense";

    if (!is_moe) {
      require_key(mlp + "gate_proj.weight");
      require_key(mlp + "up_proj.weight");
      require_key(mlp + "down_proj.weight");
      continue;
    }

    require_key(mlp + "gate.weight");
    require_key(mlp + "shared_experts.gate_proj.weight");
    require_key(mlp + "shared_experts.up_proj.weight");
    require_key(mlp + "shared_experts.down_proj.weight");
    require_key(mlp + "experts.0.gate_proj.weight");
    require_key(mlp + "experts.0.up_proj.weight");
    require_key(mlp + "experts.0.down_proj.weight");

    if (cfg.n_experts > 1) {
      const auto last = std::to_string(cfg.n_experts - 1);
      require_key(mlp + "experts." + last + ".gate_proj.weight");
      require_key(mlp + "experts." + last + ".up_proj.weight");
      require_key(mlp + "experts." + last + ".down_proj.weight");
    }
  }

  return out;
}

ds::models::core::LoadInspection inspect_loaded(const ds::hf::ModelConfig& config, const ds::hf::LoadedModel& model) {
  const auto cfg = require_deepseek_config(config);
  const auto registry = ds::rt::DeepSeekWeightRegistry::from_model(cfg, model);

  ds::models::core::LoadInspection out;
  out.layer_kinds.reserve(registry.layers().size());
  for (const auto& layer : registry.layers()) out.layer_kinds.push_back(ds::rt::layer_kind_name(layer.kind));
  return out;
}

std::shared_ptr<const ds::rt::Model> load(const std::string& model_dir, const ds::hf::ModelConfig& config) {
  return std::make_shared<const ds::rt::DeepSeekModel>(require_deepseek_config(config), ds::hf::load_model_dir(model_dir));
}

} // namespace

const ds::models::core::ModelPackageDescriptor& package_descriptor() {
  static const ds::models::core::ModelPackageDescriptor descriptor{
      .package_id = "deepseek",
      .matches = matches,
      .print_config = print_config,
      .inspect_index = inspect_index,
      .inspect_loaded = inspect_loaded,
      .load = load,
  };
  return descriptor;
}

} // namespace ds::models::deepseek
