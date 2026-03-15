#include "ds/models/deepseek/model.h"

#include "ds/core/math.h"
#include "ds/core/ops.h"
#include "ds/hf/weight_ops.h"
#include "ds/models/deepseek/ops.h"

#include <algorithm>
#include <stdexcept>

namespace ds::rt {

void CudaStateDeleter::operator()(cuda::CudaExecutorState* state) const {
#if DS_USE_CUDA
  if (state != nullptr) ds::rt::cuda::destroy_executor_state(state);
#else
  static_cast<void>(state);
#endif
}

DeepSeekSession::DeepSeekSession(const ds::hf::DeepSeekConfig& cfg, RunConfig run_cfg) : cfg_(cfg), run_cfg_(run_cfg) {
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
    cuda_state_.reset(ds::rt::cuda::create_executor_state(cfg_, run_cfg_.max_seq, n_layers));
  }
#endif
}

DeepSeekSession::~DeepSeekSession() = default;

DeepSeekSession::DeepSeekSession(DeepSeekSession&&) noexcept = default;

DeepSeekSession& DeepSeekSession::operator=(DeepSeekSession&&) noexcept = default;

void DeepSeekSession::reset() {
  pos_ = 0;
  for (auto& cache : caches_) {
    std::fill(cache.k.begin(), cache.k.end(), 0.0f);
    std::fill(cache.v.begin(), cache.v.end(), 0.0f);
  }
#if DS_USE_CUDA
  if (cuda_state_ != nullptr) {
    ds::rt::cuda::reset_executor_state(cuda_state_.get());
    ds::rt::cuda::reset_stats();
  }
#endif
}

const cuda::CudaStats& DeepSeekSession::cuda_stats() const {
#if DS_USE_CUDA
  return ds::rt::cuda::stats();
#else
  static const cuda::CudaStats empty;
  return empty;
#endif
}

DeepSeekModel::DeepSeekModel(const ds::hf::DeepSeekConfig& cfg, ds::hf::LoadedModel model)
    : cfg_(cfg), model_(std::move(model)), registry_(DeepSeekWeightRegistry::from_model(cfg_, model_)) {
  info_.family_id = "deepseek";
  info_.model_type = cfg_.model_type;
  info_.vocab_size = cfg_.vocab_size;
  info_.max_position_embeddings = cfg_.max_position_embeddings;
  info_.supported_backends.push_back(BackendKind::CPU);
#if DS_USE_CUDA
  info_.supported_backends.push_back(BackendKind::CUDA);
#endif
}

std::unique_ptr<ModelSession> DeepSeekModel::create_session(const RunConfig& run_cfg) const {
  return std::make_unique<DeepSeekSession>(cfg_, run_cfg);
}

ForwardOutput DeepSeekModel::forward(ModelSession& session, const ForwardInput& input) const {
  auto* deepseek_session = dynamic_cast<DeepSeekSession*>(&session);
  if (deepseek_session == nullptr) {
    throw std::runtime_error("deepseek model requires DeepSeekSession");
  }
  return forward_impl(*deepseek_session, input);
}

ForwardOutput DeepSeekModel::forward_impl(DeepSeekSession& session, const ForwardInput& input) const {
  if (input.token_ids.empty()) throw std::runtime_error("forward: token_ids must not be empty");

  ForwardOutput last;
  for (auto token_id : input.token_ids) last = decode_next(session, token_id);
  return last;
}

ForwardOutput DeepSeekModel::decode_next(DeepSeekSession& session, std::int32_t token_id) const {
  if (session.pos_ >= session.run_cfg_.max_seq) throw std::runtime_error("executor: max sequence length reached");

  const std::size_t hidden_size = static_cast<std::size_t>(cfg_.hidden_size);

#if DS_USE_CUDA
  if (session.run_cfg_.backend == BackendKind::CUDA) {
    if (!ds::rt::cuda::embedding_to_slot(session.cuda_state_.get(), *registry_.global().embed_tokens, token_id,
                                         cuda::DeviceBufferSlot::Hidden, hidden_size)) {
      throw std::runtime_error("executor: failed to materialize embedding on CUDA");
    }

    for (std::size_t layer_id = 0; layer_id < registry_.layers().size(); ++layer_id) {
      const auto& layer = registry_.layers()[layer_id];
      if (!ds::rt::cuda::rmsnorm_to_slot(session.cuda_state_.get(), cuda::DeviceBufferSlot::Hidden,
                                         layer.input_layernorm, hidden_size, cfg_.rms_norm_eps,
                                         cuda::DeviceBufferSlot::Norm)) {
        throw std::runtime_error("executor: CUDA RMSNorm failed for input layernorm");
      }
      if (!ds::rt::cuda::mla_decode_to_slot(session.cuda_state_.get(), cfg_, layer.attention, layer_id, session.pos_,
                                            hidden_size, cuda::DeviceBufferSlot::Norm, cuda::DeviceBufferSlot::Delta)) {
        throw std::runtime_error("executor: CUDA MLA decode failed");
      }
      if (!ds::rt::cuda::add_inplace(session.cuda_state_.get(), cuda::DeviceBufferSlot::Hidden,
                                     cuda::DeviceBufferSlot::Delta, hidden_size)) {
        throw std::runtime_error("executor: CUDA residual add failed after attention");
      }

      if (!ds::rt::cuda::rmsnorm_to_slot(session.cuda_state_.get(), cuda::DeviceBufferSlot::Hidden,
                                         layer.post_attention_layernorm, hidden_size, cfg_.rms_norm_eps,
                                         cuda::DeviceBufferSlot::Norm)) {
        throw std::runtime_error("executor: CUDA RMSNorm failed for post-attention layernorm");
      }
      bool ok = false;
      if (layer.kind == LayerKind::Dense) {
        ok = ds::rt::cuda::dense_mlp_to_slot(session.cuda_state_.get(), layer.dense_mlp, hidden_size,
                                             cuda::DeviceBufferSlot::Norm, cuda::DeviceBufferSlot::Delta);
      } else {
        ok = ds::rt::cuda::moe_to_slot(session.cuda_state_.get(), cfg_, layer.moe, hidden_size,
                                       cuda::DeviceBufferSlot::Norm, cuda::DeviceBufferSlot::Delta);
      }
      if (!ok) throw std::runtime_error("executor: CUDA feed-forward block failed");
      if (!ds::rt::cuda::add_inplace(session.cuda_state_.get(), cuda::DeviceBufferSlot::Hidden,
                                     cuda::DeviceBufferSlot::Delta, hidden_size)) {
        throw std::runtime_error("executor: CUDA residual add failed after feed-forward");
      }
    }

    if (!ds::rt::cuda::rmsnorm_to_slot(session.cuda_state_.get(), cuda::DeviceBufferSlot::Hidden,
                                       registry_.global().final_norm, hidden_size, cfg_.rms_norm_eps,
                                       cuda::DeviceBufferSlot::Norm)) {
      throw std::runtime_error("executor: CUDA final norm failed");
    }

    const auto vocab_size = static_cast<std::size_t>(registry_.global().lm_head->shape[0]);
    if (!ds::rt::cuda::linear_to_slot(session.cuda_state_.get(), *registry_.global().lm_head,
                                      cuda::DeviceBufferSlot::Norm, hidden_size, cuda::DeviceBufferSlot::Logits,
                                      vocab_size)) {
      throw std::runtime_error("executor: CUDA lm_head failed");
    }

    ForwardOutput out;
    out.logits.assign(vocab_size, 0.0f);
    if (!ds::rt::cuda::download_from_slot(session.cuda_state_.get(), cuda::DeviceBufferSlot::Logits, out.logits.data(),
                                          vocab_size)) {
      throw std::runtime_error("executor: failed to download CUDA logits");
    }
    out.greedy_token_id = static_cast<std::int32_t>(ds::core::argmax_f32(out.logits.data(), out.logits.size()));
    ++session.pos_;
    out.position_after = session.pos_;
    return out;
  }
#endif

  std::vector<float> hidden(hidden_size, 0.0f);
  std::vector<float> norm(hidden_size, 0.0f);
  std::vector<float> delta(hidden_size, 0.0f);

  ds::hf::embedding_lookup_f32(*registry_.global().embed_tokens, token_id, hidden.data());

  for (std::size_t layer_id = 0; layer_id < registry_.layers().size(); ++layer_id) {
    const auto& layer = registry_.layers()[layer_id];
    rmsnorm_backend(session.run_cfg_.backend, hidden.data(), layer.input_layernorm.data(), hidden_size, cfg_.rms_norm_eps,
                    norm.data());
    std::fill(delta.begin(), delta.end(), 0.0f);
    mla_decode_step(cfg_, layer.attention, norm.data(), session.pos_, &session.caches_[layer_id], delta.data(),
                    session.run_cfg_.backend);
    for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += delta[i];

    rmsnorm_backend(session.run_cfg_.backend, hidden.data(), layer.post_attention_layernorm.data(), hidden_size,
                    cfg_.rms_norm_eps, norm.data());
    std::fill(delta.begin(), delta.end(), 0.0f);
    if (layer.kind == LayerKind::Dense) {
      dense_mlp_step(session.run_cfg_.backend, layer.dense_mlp, norm.data(), hidden_size, delta.data());
    } else {
      moe_step(session.run_cfg_.backend, cfg_, layer.moe, norm.data(), hidden_size, delta.data());
    }
    for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += delta[i];
  }

  rmsnorm_backend(session.run_cfg_.backend, hidden.data(), registry_.global().final_norm.data(), hidden_size,
                  cfg_.rms_norm_eps, norm.data());

  ForwardOutput out;
  lm_head_logits(session.run_cfg_.backend, *registry_.global().lm_head, norm.data(), &out.logits);
  out.greedy_token_id = static_cast<std::int32_t>(ds::core::argmax_f32(out.logits.data(), out.logits.size()));
  ++session.pos_;
  out.position_after = session.pos_;
  return out;
}

} // namespace ds::rt
