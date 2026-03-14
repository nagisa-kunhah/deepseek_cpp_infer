#include "ds/core/math.h"
#include "ds/hf/config.h"
#include "ds/runtime/mla.h"
#include "ds/runtime/ops.h"
#include "ds/runtime/tokenizer.h"
#include "ds/runtime/weights.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace {

ds::hf::TensorSlice make_tensor(const std::string& name, const std::vector<std::int64_t>& shape, const float* data,
                                std::size_t nbytes) {
  ds::hf::TensorSlice t;
  t.name = name;
  t.dtype = ds::hf::DType::F32;
  t.shape = shape;
  t.data = reinterpret_cast<const std::uint8_t*>(data);
  t.nbytes = nbytes;
  t.shard_path = "synthetic";
  return t;
}

void test_tokenizer() {
  const std::string path = "/tmp/ds_runtime_tokenizer_test.json";
  std::ofstream f(path);
  f << R"({
    "model": {
      "type": "WordLevel",
      "vocab": {
        "hello": 0,
        " world": 1,
        "!": 2
      }
    }
  })";
  f.close();

  const auto tok = ds::rt::Tokenizer::load_from_file(path);
  const auto ids = tok.encode("hello world!");
  assert(ids.size() == 3);
  assert(ids[0] == 0);
  assert(ids[1] == 1);
  assert(ids[2] == 2);
}

void test_weight_registry() {
  ds::hf::DeepSeekConfig cfg;
  cfg.hidden_size = 4;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 2;
  cfg.num_key_value_heads = 2;
  cfg.vocab_size = 8;
  cfg.qk_nope_head_dim = 1;
  cfg.qk_rope_head_dim = 1;
  cfg.v_head_dim = 1;
  cfg.kv_lora_rank = 2;
  cfg.n_experts = 2;

  static const float dense_2x4[8] = {0};
  static const float dense_4x4[16] = {0};
  static const float dense_3x4[12] = {0};
  static const float dense_4x2[8] = {0};
  static const float vec4[4] = {1, 1, 1, 1};
  static const float vec2[2] = {1, 1};
  static const float gate_2x4[8] = {0};

  ds::hf::LoadedModel model;
  model.tensors.emplace("model.embed_tokens.weight", make_tensor("model.embed_tokens.weight", {8, 4}, dense_4x4, sizeof(dense_4x4)));
  model.tensors.emplace("model.norm.weight", make_tensor("model.norm.weight", {4}, vec4, sizeof(vec4)));
  model.tensors.emplace("lm_head.weight", make_tensor("lm_head.weight", {8, 4}, dense_4x4, sizeof(dense_4x4)));

  for (int layer = 0; layer < 2; ++layer) {
    const auto base = "model.layers." + std::to_string(layer) + ".";
    model.tensors.emplace(base + "input_layernorm.weight", make_tensor(base + "input_layernorm.weight", {4}, vec4, sizeof(vec4)));
    model.tensors.emplace(base + "post_attention_layernorm.weight",
                          make_tensor(base + "post_attention_layernorm.weight", {4}, vec4, sizeof(vec4)));
    model.tensors.emplace(base + "self_attn.q_proj.weight", make_tensor(base + "self_attn.q_proj.weight", {4, 4}, dense_4x4, sizeof(dense_4x4)));
    model.tensors.emplace(base + "self_attn.kv_a_layernorm.weight",
                          make_tensor(base + "self_attn.kv_a_layernorm.weight", {2}, vec2, sizeof(vec2)));
    model.tensors.emplace(base + "self_attn.kv_a_proj_with_mqa.weight",
                          make_tensor(base + "self_attn.kv_a_proj_with_mqa.weight", {3, 4}, dense_3x4, sizeof(dense_3x4)));
    model.tensors.emplace(base + "self_attn.kv_b_proj.weight", make_tensor(base + "self_attn.kv_b_proj.weight", {4, 2}, dense_4x2, sizeof(dense_4x2)));
    model.tensors.emplace(base + "self_attn.o_proj.weight", make_tensor(base + "self_attn.o_proj.weight", {4, 2}, dense_4x2, sizeof(dense_4x2)));
  }

  model.tensors.emplace("model.layers.0.mlp.gate_proj.weight", make_tensor("model.layers.0.mlp.gate_proj.weight", {2, 4}, dense_2x4, sizeof(dense_2x4)));
  model.tensors.emplace("model.layers.0.mlp.up_proj.weight", make_tensor("model.layers.0.mlp.up_proj.weight", {2, 4}, dense_2x4, sizeof(dense_2x4)));
  model.tensors.emplace("model.layers.0.mlp.down_proj.weight", make_tensor("model.layers.0.mlp.down_proj.weight", {4, 2}, dense_4x2, sizeof(dense_4x2)));

  model.tensors.emplace("model.layers.1.mlp.gate.weight", make_tensor("model.layers.1.mlp.gate.weight", {2, 4}, gate_2x4, sizeof(gate_2x4)));
  model.tensors.emplace("model.layers.1.mlp.shared_experts.gate_proj.weight",
                        make_tensor("model.layers.1.mlp.shared_experts.gate_proj.weight", {2, 4}, dense_2x4, sizeof(dense_2x4)));
  model.tensors.emplace("model.layers.1.mlp.shared_experts.up_proj.weight",
                        make_tensor("model.layers.1.mlp.shared_experts.up_proj.weight", {2, 4}, dense_2x4, sizeof(dense_2x4)));
  model.tensors.emplace("model.layers.1.mlp.shared_experts.down_proj.weight",
                        make_tensor("model.layers.1.mlp.shared_experts.down_proj.weight", {4, 2}, dense_4x2, sizeof(dense_4x2)));
  for (int expert = 0; expert < 2; ++expert) {
    const auto prefix = "model.layers.1.mlp.experts." + std::to_string(expert) + ".";
    model.tensors.emplace(prefix + "gate_proj.weight", make_tensor(prefix + "gate_proj.weight", {2, 4}, dense_2x4, sizeof(dense_2x4)));
    model.tensors.emplace(prefix + "up_proj.weight", make_tensor(prefix + "up_proj.weight", {2, 4}, dense_2x4, sizeof(dense_2x4)));
    model.tensors.emplace(prefix + "down_proj.weight", make_tensor(prefix + "down_proj.weight", {4, 2}, dense_4x2, sizeof(dense_4x2)));
  }

  const auto registry = ds::rt::WeightRegistry::from_model(cfg, model);
  assert(registry.layers().size() == 2);
  assert(registry.layers()[0].kind == ds::rt::LayerKind::Dense);
  assert(registry.layers()[1].kind == ds::rt::LayerKind::MoE);
}

void test_mla_smoke() {
  ds::hf::DeepSeekConfig cfg;
  cfg.hidden_size = 4;
  cfg.num_attention_heads = 2;
  cfg.qk_nope_head_dim = 0;
  cfg.qk_rope_head_dim = 2;
  cfg.v_head_dim = 1;
  cfg.kv_lora_rank = 2;
  cfg.rms_norm_eps = 1e-5f;

  static const float q_proj[16] = {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
  };
  static const float kv_a_proj[16] = {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
  };
  static const float kv_a_ln[2] = {1, 1};
  static const float kv_b_proj[4] = {
      1, 0,
      0, 1,
  };
  static const float o_proj[8] = {
      1, 0,
      0, 1,
      1, 0,
      0, 1,
  };

  const auto q_proj_tensor = make_tensor("q_proj", {4, 4}, q_proj, sizeof(q_proj));
  const auto kv_a_tensor = make_tensor("kv_a", {4, 4}, kv_a_proj, sizeof(kv_a_proj));
  const auto kv_a_ln_tensor = make_tensor("kv_a_ln", {2}, kv_a_ln, sizeof(kv_a_ln));
  const auto kv_b_tensor = make_tensor("kv_b", {2, 2}, kv_b_proj, sizeof(kv_b_proj));
  const auto o_tensor = make_tensor("o_proj", {4, 2}, o_proj, sizeof(o_proj));

  ds::rt::AttentionWeights attn;
  attn.q_proj = {&q_proj_tensor, nullptr};
  attn.kv_a_proj_with_mqa = {&kv_a_tensor, nullptr};
  attn.kv_a_layernorm = {&kv_a_ln_tensor};
  attn.kv_b_proj = {&kv_b_tensor, nullptr};
  attn.o_proj = {&o_tensor, nullptr};

  ds::rt::MLACache cache;
  cache.init(4, 2, 2, 1);

  const float hidden[4] = {1, 2, 3, 4};
  float out[4] = {0, 0, 0, 0};
  ds::rt::mla_decode_step_cpu(cfg, attn, hidden, 0, &cache, out);
  for (float v : out) assert(std::isfinite(v));
}

} // namespace

int main() {
  test_tokenizer();
  test_weight_registry();
  test_mla_smoke();
  return 0;
}
