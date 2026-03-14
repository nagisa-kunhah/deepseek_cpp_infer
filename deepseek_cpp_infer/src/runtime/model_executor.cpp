#include "ds/runtime/model_executor.h"

#include "ds/core/math.h"
#include "ds/core/ops.h"
#include "ds/hf/weight_ops.h"
#include "ds/runtime/cuda_backend.h"
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
#if DS_USE_CUDA
  if (run_cfg_.backend == BackendKind::CUDA) {
    ds::rt::cuda::ensure_initialized();
    ds::rt::cuda::reset_stats();
  }
#endif

  if (run_cfg_.max_seq == 0) run_cfg_.max_seq = static_cast<std::size_t>(cfg_.max_position_embeddings);

  const std::size_t n_layers = static_cast<std::size_t>(cfg_.num_hidden_layers);
  const std::size_t n_heads = static_cast<std::size_t>(cfg_.num_attention_heads);
  const std::size_t q_head_dim = static_cast<std::size_t>(cfg_.qk_nope_head_dim + cfg_.qk_rope_head_dim);
  const std::size_t v_head_dim = static_cast<std::size_t>(cfg_.v_head_dim);

  caches_.resize(n_layers);
  for (auto& cache : caches_) cache.init(run_cfg_.max_seq, n_heads, q_head_dim, v_head_dim);

#if DS_USE_CUDA
  if (run_cfg_.backend == BackendKind::CUDA) {
    cuda_state_ = ds::rt::cuda::create_executor_state(cfg_, run_cfg_.max_seq, n_layers);
  }
#endif
}

ModelExecutor::~ModelExecutor() {
#if DS_USE_CUDA
  if (cuda_state_ != nullptr) {
    ds::rt::cuda::destroy_executor_state(cuda_state_);
    cuda_state_ = nullptr;
  }
#endif
}

void ModelExecutor::reset() {
  pos_ = 0;
  for (auto& cache : caches_) {
    std::fill(cache.k.begin(), cache.k.end(), 0.0f);
    std::fill(cache.v.begin(), cache.v.end(), 0.0f);
  }
#if DS_USE_CUDA
  if (cuda_state_ != nullptr) {
    ds::rt::cuda::reset_executor_state(cuda_state_);
    ds::rt::cuda::reset_stats();
  }
#endif
}

const cuda::CudaStats& ModelExecutor::cuda_stats() const {
#if DS_USE_CUDA
  return ds::rt::cuda::stats();
#else
  static const cuda::CudaStats empty;
  return empty;
#endif
}

StepResult ModelExecutor::decode_next(std::int32_t token_id) {
  if (pos_ >= run_cfg_.max_seq) throw std::runtime_error("executor: max sequence length reached");

  const std::size_t hidden_size = static_cast<std::size_t>(cfg_.hidden_size);

#if DS_USE_CUDA
  if (run_cfg_.backend == BackendKind::CUDA) {
    std::vector<float> hidden_host(hidden_size, 0.0f);
    ds::hf::embedding_lookup_f32(*registry_.global().embed_tokens, token_id, hidden_host.data());
    if (!ds::rt::cuda::upload_to_slot(cuda_state_, cuda::DeviceBufferSlot::Hidden, hidden_host.data(), hidden_size)) {
      throw std::runtime_error("executor: failed to upload hidden state to CUDA");
    }

    for (std::size_t layer_id = 0; layer_id < registry_.layers().size(); ++layer_id) {
      const auto& layer = registry_.layers()[layer_id];
      if (!ds::rt::cuda::rmsnorm_to_slot(cuda_state_, cuda::DeviceBufferSlot::Hidden, layer.input_layernorm.data(),
                                         hidden_size, cfg_.rms_norm_eps, cuda::DeviceBufferSlot::Norm)) {
        throw std::runtime_error("executor: CUDA RMSNorm failed for input layernorm");
      }
      if (!ds::rt::cuda::mla_decode_to_slot(cuda_state_, cfg_, layer.attention, layer_id, pos_, hidden_size,
                                            cuda::DeviceBufferSlot::Norm, cuda::DeviceBufferSlot::Delta)) {
        throw std::runtime_error("executor: CUDA MLA decode failed");
      }
      if (!ds::rt::cuda::add_inplace(cuda_state_, cuda::DeviceBufferSlot::Hidden, cuda::DeviceBufferSlot::Delta,
                                     hidden_size)) {
        throw std::runtime_error("executor: CUDA residual add failed after attention");
      }

      if (!ds::rt::cuda::rmsnorm_to_slot(cuda_state_, cuda::DeviceBufferSlot::Hidden, layer.post_attention_layernorm.data(),
                                         hidden_size, cfg_.rms_norm_eps, cuda::DeviceBufferSlot::Norm)) {
        throw std::runtime_error("executor: CUDA RMSNorm failed for post-attention layernorm");
      }
      bool ok = false;
      if (layer.kind == LayerKind::Dense) {
        ok = ds::rt::cuda::dense_mlp_to_slot(cuda_state_, layer.dense_mlp, hidden_size, cuda::DeviceBufferSlot::Norm,
                                             cuda::DeviceBufferSlot::Delta);
      } else {
        ok = ds::rt::cuda::moe_to_slot(cuda_state_, cfg_, layer.moe, hidden_size, cuda::DeviceBufferSlot::Norm,
                                       cuda::DeviceBufferSlot::Delta);
      }
      if (!ok) throw std::runtime_error("executor: CUDA feed-forward block failed");
      if (!ds::rt::cuda::add_inplace(cuda_state_, cuda::DeviceBufferSlot::Hidden, cuda::DeviceBufferSlot::Delta,
                                     hidden_size)) {
        throw std::runtime_error("executor: CUDA residual add failed after feed-forward");
      }
    }

    if (!ds::rt::cuda::rmsnorm_to_slot(cuda_state_, cuda::DeviceBufferSlot::Hidden, registry_.global().final_norm.data(),
                                       hidden_size, cfg_.rms_norm_eps, cuda::DeviceBufferSlot::Norm)) {
      throw std::runtime_error("executor: CUDA final norm failed");
    }

    const auto vocab_size = static_cast<std::size_t>(registry_.global().lm_head->shape[0]);
    if (!ds::rt::cuda::linear_to_slot(cuda_state_, *registry_.global().lm_head, cuda::DeviceBufferSlot::Norm, hidden_size,
                                      cuda::DeviceBufferSlot::Logits, vocab_size)) {
      throw std::runtime_error("executor: CUDA lm_head failed");
    }

    StepResult out;
    out.logits.assign(vocab_size, 0.0f);
    if (!ds::rt::cuda::download_from_slot(cuda_state_, cuda::DeviceBufferSlot::Logits, out.logits.data(), vocab_size)) {
      throw std::runtime_error("executor: failed to download CUDA logits");
    }
    out.greedy_token_id = static_cast<std::int32_t>(ds::core::argmax_f32(out.logits.data(), out.logits.size()));
    ++pos_;
    return out;
  }
#endif

  std::vector<float> hidden(hidden_size, 0.0f);
  std::vector<float> norm(hidden_size, 0.0f);
  std::vector<float> delta(hidden_size, 0.0f);

  ds::hf::embedding_lookup_f32(*registry_.global().embed_tokens, token_id, hidden.data());

  for (std::size_t layer_id = 0; layer_id < registry_.layers().size(); ++layer_id) {
    const auto& layer = registry_.layers()[layer_id];
    rmsnorm_backend(run_cfg_.backend, hidden.data(), layer.input_layernorm.data(), hidden_size, cfg_.rms_norm_eps, norm.data());
    std::fill(delta.begin(), delta.end(), 0.0f);
    mla_decode_step(cfg_, layer.attention, norm.data(), pos_, &caches_[layer_id], delta.data(), run_cfg_.backend);
    for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += delta[i];

    rmsnorm_backend(run_cfg_.backend, hidden.data(), layer.post_attention_layernorm.data(), hidden_size, cfg_.rms_norm_eps,
                    norm.data());
    std::fill(delta.begin(), delta.end(), 0.0f);
    if (layer.kind == LayerKind::Dense) {
      dense_mlp_step(run_cfg_.backend, layer.dense_mlp, norm.data(), hidden_size, delta.data());
    } else {
      moe_step(run_cfg_.backend, cfg_, layer.moe, norm.data(), hidden_size, delta.data());
    }
    for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += delta[i];
  }

  rmsnorm_backend(run_cfg_.backend, hidden.data(), registry_.global().final_norm.data(), hidden_size, cfg_.rms_norm_eps,
                  norm.data());

  StepResult out;
  lm_head_logits(run_cfg_.backend, *registry_.global().lm_head, norm.data(), &out.logits);
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
