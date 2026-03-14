#include "ds/core/math.h"
#include "ds/hf/config.h"
#include "ds/runtime/deepseek_model.h"
#include "ds/runtime/mla.h"
#include "ds/runtime/model_executor.h"
#include "ds/runtime/model_factory.h"
#include "ds/runtime/ops.h"
#include "ds/runtime/tokenizer.h"
#include "ds/runtime/weights.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

static_assert(!std::is_copy_constructible_v<ds::rt::DeepSeekSession>);
static_assert(!std::is_copy_assignable_v<ds::rt::DeepSeekSession>);
static_assert(std::is_move_constructible_v<ds::rt::DeepSeekSession>);
static_assert(std::is_move_assignable_v<ds::rt::DeepSeekSession>);

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

ds::hf::DeepSeekConfig make_cfg() {
  ds::hf::DeepSeekConfig cfg;
  cfg.model_type = "deepseek_v2";
  cfg.hidden_size = 4;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 2;
  cfg.num_key_value_heads = 2;
  cfg.vocab_size = 8;
  cfg.max_position_embeddings = 8;
  cfg.qk_nope_head_dim = 0;
  cfg.qk_rope_head_dim = 2;
  cfg.v_head_dim = 1;
  cfg.kv_lora_rank = 2;
  cfg.n_experts = 2;
  cfg.moe_top_k = 1;
  cfg.rms_norm_eps = 1e-5f;
  cfg.routed_scaling_factor = 1.0f;
  cfg.norm_topk_prob = true;
  return cfg;
}

ds::hf::LoadedModel make_model() {
  static const float embed[32] = {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      1, 1, 0, 0,
      0, 1, 1, 0,
      0, 0, 1, 1,
      1, 0, 0, 1,
  };
  static const float norm4[4] = {1, 1, 1, 1};
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
  static const float dense_2x4[8] = {
      0.5f, 0.1f, 0.0f, 0.0f,
      0.0f, 0.5f, 0.1f, 0.0f,
  };
  static const float dense_4x2[8] = {
      1.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 0.0f,
      0.0f, 1.0f,
  };
  static const float gate_2x4[8] = {
      2.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 2.0f, 0.0f, 0.0f,
  };
  static const float lm_head[32] = {
      3, 0, 0, 0,
      0, 3, 0, 0,
      0, 0, 3, 0,
      0, 0, 0, 3,
      2, 1, 0, 0,
      0, 2, 1, 0,
      0, 0, 2, 1,
      1, 0, 0, 2,
  };

  ds::hf::LoadedModel model;
  model.tensors.emplace("model.embed_tokens.weight", make_tensor("model.embed_tokens.weight", {8, 4}, embed, sizeof(embed)));
  model.tensors.emplace("model.norm.weight", make_tensor("model.norm.weight", {4}, norm4, sizeof(norm4)));
  model.tensors.emplace("lm_head.weight", make_tensor("lm_head.weight", {8, 4}, lm_head, sizeof(lm_head)));

  for (int layer = 0; layer < 2; ++layer) {
    const auto base = "model.layers." + std::to_string(layer) + ".";
    model.tensors.emplace(base + "input_layernorm.weight", make_tensor(base + "input_layernorm.weight", {4}, norm4, sizeof(norm4)));
    model.tensors.emplace(base + "post_attention_layernorm.weight",
                          make_tensor(base + "post_attention_layernorm.weight", {4}, norm4, sizeof(norm4)));
    model.tensors.emplace(base + "self_attn.q_proj.weight", make_tensor(base + "self_attn.q_proj.weight", {4, 4}, q_proj, sizeof(q_proj)));
    model.tensors.emplace(base + "self_attn.kv_a_layernorm.weight",
                          make_tensor(base + "self_attn.kv_a_layernorm.weight", {2}, kv_a_ln, sizeof(kv_a_ln)));
    model.tensors.emplace(base + "self_attn.kv_a_proj_with_mqa.weight",
                          make_tensor(base + "self_attn.kv_a_proj_with_mqa.weight", {4, 4}, kv_a_proj, sizeof(kv_a_proj)));
    model.tensors.emplace(base + "self_attn.kv_b_proj.weight", make_tensor(base + "self_attn.kv_b_proj.weight", {2, 2}, kv_b_proj, sizeof(kv_b_proj)));
    model.tensors.emplace(base + "self_attn.o_proj.weight", make_tensor(base + "self_attn.o_proj.weight", {4, 2}, o_proj, sizeof(o_proj)));
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

  return model;
}

void assert_close(const std::vector<float>& a, const std::vector<float>& b, float atol) {
  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    assert(std::fabs(a[i] - b[i]) <= atol);
  }
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

void test_deepseek_weight_registry() {
  auto cfg = make_cfg();
  auto model = make_model();
  const auto registry = ds::rt::DeepSeekWeightRegistry::from_model(cfg, model);
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
  attn.kv_a_layernorm = {&kv_a_ln_tensor, {1.0f, 1.0f}};
  attn.kv_b_proj = {&kv_b_tensor, nullptr};
  attn.o_proj = {&o_tensor, nullptr};

  ds::rt::MLACache cache;
  cache.init(4, 2, 2, 1);

  const float hidden[4] = {1, 2, 3, 4};
  float out[4] = {0, 0, 0, 0};
  ds::rt::mla_decode_step(cfg, attn, hidden, 0, &cache, out, ds::rt::BackendKind::CPU);
  for (float v : out) assert(std::isfinite(v));
}

void test_model_factory_detection() {
  assert(ds::rt::detect_model_family("deepseek_v2") == ds::rt::ModelFamily::DeepSeek);
  bool threw = false;
  try {
    (void)ds::rt::detect_model_family("llama");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

void test_deepseek_session_independence() {
  auto cfg = make_cfg();
  ds::rt::DeepSeekModel model(cfg, make_model());
  auto session_a = model.create_session(ds::rt::RunConfig{.backend = ds::rt::BackendKind::CPU, .max_seq = 8});
  auto session_b = model.create_session(ds::rt::RunConfig{.backend = ds::rt::BackendKind::CPU, .max_seq = 8});

  const auto out_a0 = model.forward(*session_a, ds::rt::ForwardInput{{0}});
  assert(session_a->position() == 1);
  assert(session_b->position() == 0);

  const auto out_b0 = model.forward(*session_b, ds::rt::ForwardInput{{0}});
  assert_close(out_a0.logits, out_b0.logits, 1e-6f);
  assert(session_b->position() == 1);

  (void)model.forward(*session_a, ds::rt::ForwardInput{{1}});
  assert(session_a->position() == 2);
  assert(session_b->position() == 1);

  session_a->reset();
  assert(session_a->position() == 0);
  assert(session_b->position() == 1);
}

void test_executor_shim_matches_model() {
  auto cfg = make_cfg();

  ds::rt::DeepSeekModel model(cfg, make_model());
  auto session = model.create_session(ds::rt::RunConfig{.backend = ds::rt::BackendKind::CPU, .max_seq = 8});

  ds::rt::ModelExecutor executor(cfg, make_model(), ds::rt::RunConfig{.backend = ds::rt::BackendKind::CPU, .max_seq = 8});

  const auto direct = model.forward(*session, ds::rt::ForwardInput{{0}});
  const auto shim = executor.prefill({0});

  assert(direct.greedy_token_id == shim.greedy_token_id);
  assert_close(direct.logits, shim.logits, 1e-6f);
}

} // namespace

int main() {
  test_tokenizer();
  test_deepseek_weight_registry();
  test_mla_smoke();
  test_model_factory_detection();
  test_deepseek_session_independence();
  test_executor_shim_matches_model();
  return 0;
}
