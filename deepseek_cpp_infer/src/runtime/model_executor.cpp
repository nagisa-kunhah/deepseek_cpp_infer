#include "ds/runtime/model_executor.h"

#include "ds/core/math.h"
#include "ds/core/ops.h"
#include "ds/hf/decode.h"
#include "ds/hf/weight_ops.h"
#include "ds/runtime/ops.h"

#include <algorithm>
#include <stdexcept>

namespace ds::rt {

ModelExecutor::ModelExecutor(const ds::hf::DeepSeekConfig& cfg, ds::hf::LoadedModel model, RunConfig run_cfg)
    : cfg_(cfg), run_cfg_(run_cfg), model_(std::move(model)), registry_(WeightRegistry::from_model(cfg_, model_)) {
#if !DS_USE_CUDA
  if (run_cfg_.backend == BackendKind::CUDA) {
    throw std::runtime_error("CUDA backend requested but this build has no CUDA support");
  }
#endif
  if (run_cfg_.backend == BackendKind::CUDA) {
    throw std::runtime_error("CUDA backend interface exists, but CUDA kernels are not implemented yet");
  }

  if (run_cfg_.max_seq == 0) run_cfg_.max_seq = static_cast<std::size_t>(cfg_.max_position_embeddings);

  const std::size_t n_layers = static_cast<std::size_t>(cfg_.num_hidden_layers);
  const std::size_t n_heads = static_cast<std::size_t>(cfg_.num_attention_heads);
  const std::size_t q_head_dim = static_cast<std::size_t>(cfg_.qk_nope_head_dim + cfg_.qk_rope_head_dim);
  const std::size_t v_head_dim = static_cast<std::size_t>(cfg_.v_head_dim);

  caches_.resize(n_layers);
  for (auto& cache : caches_) cache.init(run_cfg_.max_seq, n_heads, q_head_dim, v_head_dim);
}

void ModelExecutor::reset() {
  pos_ = 0;
  for (auto& cache : caches_) {
    std::fill(cache.k.begin(), cache.k.end(), 0.0f);
    std::fill(cache.v.begin(), cache.v.end(), 0.0f);
  }
}

StepResult ModelExecutor::decode_next(std::int32_t token_id) {
  if (pos_ >= run_cfg_.max_seq) throw std::runtime_error("executor: max sequence length reached");

  const std::size_t hidden_size = static_cast<std::size_t>(cfg_.hidden_size);
  std::vector<float> hidden(hidden_size, 0.0f);
  std::vector<float> norm(hidden_size, 0.0f);
  std::vector<float> delta(hidden_size, 0.0f);

  ds::hf::embedding_lookup_f32(*registry_.global().embed_tokens, token_id, hidden.data());

  for (std::size_t layer_id = 0; layer_id < registry_.layers().size(); ++layer_id) {
    const auto& layer = registry_.layers()[layer_id];

    const auto ln_in = ds::hf::decode_to_f32(*layer.input_layernorm.weight);
    ds::core::rmsnorm_f32(hidden.data(), ln_in.data(), hidden_size, cfg_.rms_norm_eps, norm.data());
    std::fill(delta.begin(), delta.end(), 0.0f);
    mla_decode_step_cpu(cfg_, layer.attention, norm.data(), pos_, &caches_[layer_id], delta.data());
    for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += delta[i];

    const auto ln_post = ds::hf::decode_to_f32(*layer.post_attention_layernorm.weight);
    ds::core::rmsnorm_f32(hidden.data(), ln_post.data(), hidden_size, cfg_.rms_norm_eps, norm.data());
    std::fill(delta.begin(), delta.end(), 0.0f);
    if (layer.kind == LayerKind::Dense) {
      dense_mlp_step_cpu(layer.dense_mlp, norm.data(), hidden_size, delta.data());
    } else {
      moe_step_cpu(cfg_, layer.moe, norm.data(), hidden_size, delta.data());
    }
    for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += delta[i];
  }

  const auto final_ln = ds::hf::decode_to_f32(*registry_.global().final_norm);
  ds::core::rmsnorm_f32(hidden.data(), final_ln.data(), hidden_size, cfg_.rms_norm_eps, norm.data());

  StepResult out;
  lm_head_logits_cpu(*registry_.global().lm_head, norm.data(), &out.logits);
  out.greedy_token_id = static_cast<std::int32_t>(ds::core::argmax_f32(out.logits.data(), out.logits.size()));
  ++pos_;
  return out;
}

StepResult ModelExecutor::prefill(const std::vector<std::int32_t>& prompt_ids) {
  if (prompt_ids.empty()) throw std::runtime_error("executor: prompt_ids must not be empty");
  StepResult last;
  for (auto token_id : prompt_ids) last = decode_next(token_id);
  return last;
}

std::vector<std::int32_t> ModelExecutor::generate(const std::vector<std::int32_t>& prompt_ids, const GenerationConfig& gen_cfg,
                                                  const Tokenizer* tokenizer, std::vector<std::string>* text_pieces) {
  if (gen_cfg.max_new_tokens <= 0) return {};
  StepResult step = prefill(prompt_ids);

  ds::core::Sampler sampler(ds::core::SamplerConfig{
      .temperature = gen_cfg.temperature <= 0.0f ? 1.0f : gen_cfg.temperature,
      .top_k = gen_cfg.top_k,
      .top_p = gen_cfg.top_p,
      .seed = gen_cfg.seed,
  });

  std::vector<std::int32_t> generated;
  generated.reserve(static_cast<std::size_t>(gen_cfg.max_new_tokens));
  for (std::int32_t i = 0; i < gen_cfg.max_new_tokens; ++i) {
    std::int32_t next_id = step.greedy_token_id;
    if (gen_cfg.temperature > 0.0f || gen_cfg.top_k > 0 || gen_cfg.top_p > 0.0f) {
      next_id = static_cast<std::int32_t>(sampler.sample(step.logits));
    }

    generated.push_back(next_id);
    if (text_pieces != nullptr && tokenizer != nullptr) {
      text_pieces->push_back(tokenizer->decode({next_id}));
    }
    if (tokenizer != nullptr && tokenizer->metadata().eos_token_id >= 0 && next_id == tokenizer->metadata().eos_token_id) {
      break;
    }
    if (i + 1 < gen_cfg.max_new_tokens) step = decode_next(next_id);
  }

  return generated;
}

} // namespace ds::rt
