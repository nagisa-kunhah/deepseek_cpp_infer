#include "ds/core/math.h"
#include "ds/hf/config.h"
#include "ds/runtime/deepseek_model.h"
#include "ds/runtime/model_executor.h"

#include <cassert>
#include <cmath>
#include <cstdint>
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

void test_cuda_model_matches_cpu() {
  auto cfg = make_cfg();
  ds::rt::DeepSeekModel cpu_model(cfg, make_model());
  ds::rt::DeepSeekModel cuda_model(cfg, make_model());

  auto cpu_session = cpu_model.create_session(ds::rt::RunConfig{.backend = ds::rt::BackendKind::CPU, .max_seq = 8});
  auto cuda_session = cuda_model.create_session(ds::rt::RunConfig{.backend = ds::rt::BackendKind::CUDA, .max_seq = 8});

  const auto cpu_step = cpu_model.forward(*cpu_session, ds::rt::ForwardInput{{0}});
  const auto cuda_step = cuda_model.forward(*cuda_session, ds::rt::ForwardInput{{0}});

  assert(cpu_step.greedy_token_id == cuda_step.greedy_token_id);
  assert_close(cpu_step.logits, cuda_step.logits, 1e-3f);

  const auto* deepseek_cuda_session = dynamic_cast<const ds::rt::DeepSeekSession*>(cuda_session.get());
  assert(deepseek_cuda_session != nullptr);
  const auto& stats = deepseek_cuda_session->cuda_stats();
  assert(stats.embedding_cuda_hits > 0);
  assert(stats.linear_cuda_hits > 0);
  assert(stats.mla_cuda_hits > 0);
  assert(stats.moe_cuda_hits > 0);
  assert(stats.cached_weight_hits > 0);
  assert(stats.stream_linear_fallbacks == 0);
}

void test_large_weight_preload_avoids_streaming_fallback() {
  ds::rt::cuda::ensure_initialized();
  ds::rt::cuda::reset_stats();

  ds::hf::DeepSeekConfig dummy_cfg;
  auto* state = ds::rt::cuda::create_executor_state(dummy_cfg, 1, 0);

  const std::size_t in = 1024;
  const std::size_t out = 8193;
  std::vector<float> weight(out * in, 0.0f);
  std::vector<float> input(in, 1.0f);
  std::vector<float> output(out, 0.0f);
  for (std::size_t row = 0; row < out; ++row) {
    weight[row * in] = 1.0f;
  }

  const auto tensor = make_tensor("large_linear", {static_cast<std::int64_t>(out), static_cast<std::int64_t>(in)},
                                  weight.data(), weight.size() * sizeof(float));

  assert(ds::rt::cuda::preload_tensor(tensor));
  assert(ds::rt::cuda::upload_to_slot(state, ds::rt::cuda::DeviceBufferSlot::Hidden, input.data(), in));
  assert(ds::rt::cuda::linear_to_slot(state, tensor, ds::rt::cuda::DeviceBufferSlot::Hidden, in,
                                      ds::rt::cuda::DeviceBufferSlot::Delta, out));
  assert(ds::rt::cuda::download_from_slot(state, ds::rt::cuda::DeviceBufferSlot::Delta, output.data(), out));

  for (float value : output) assert(std::fabs(value - 1.0f) <= 1e-4f);

  const auto& stats = ds::rt::cuda::stats();
  assert(stats.linear_cuda_hits == 1);
  assert(stats.cached_weight_uploads > 0);
  assert(stats.cached_weight_bytes >= weight.size() * sizeof(float));
  assert(stats.stream_linear_fallbacks == 0);

  ds::rt::cuda::destroy_executor_state(state);
}

void test_executor_shim_matches_new_cuda_path() {
  auto cfg = make_cfg();
  ds::rt::DeepSeekModel model(cfg, make_model());
  auto session = model.create_session(ds::rt::RunConfig{.backend = ds::rt::BackendKind::CUDA, .max_seq = 8});

  ds::rt::ModelExecutor executor(cfg, make_model(), ds::rt::RunConfig{.backend = ds::rt::BackendKind::CUDA, .max_seq = 8});

  const auto direct = model.forward(*session, ds::rt::ForwardInput{{0}});
  const auto shim = executor.prefill({0});

  assert(direct.greedy_token_id == shim.greedy_token_id);
  assert_close(direct.logits, shim.logits, 1e-3f);
}

} // namespace

int main() {
  test_cuda_model_matches_cpu();
  test_large_weight_preload_avoids_streaming_fallback();
  test_executor_shim_matches_new_cuda_path();
  return 0;
}
