#include "ds/models/deepseek/mla.h"

#include "ds/core/math.h"
#include "ds/core/ops.h"
#include "ds/core/rope.h"
#include "ds/models/deepseek/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ds::rt {

void MLACache::init(std::size_t max_seq_, std::size_t n_heads_, std::size_t q_head_dim_, std::size_t v_head_dim_) {
  max_seq = max_seq_;
  n_heads = n_heads_;
  q_head_dim = q_head_dim_;
  v_head_dim = v_head_dim_;
  k.assign(max_seq * n_heads * q_head_dim, 0.0f);
  v.assign(max_seq * n_heads * v_head_dim, 0.0f);
}

float* MLACache::key_ptr(std::size_t pos, std::size_t head) {
  return k.data() + ((pos * n_heads + head) * q_head_dim);
}

float* MLACache::val_ptr(std::size_t pos, std::size_t head) {
  return v.data() + ((pos * n_heads + head) * v_head_dim);
}

const float* MLACache::key_ptr(std::size_t pos, std::size_t head) const {
  return k.data() + ((pos * n_heads + head) * q_head_dim);
}

const float* MLACache::val_ptr(std::size_t pos, std::size_t head) const {
  return v.data() + ((pos * n_heads + head) * v_head_dim);
}

void mla_decode_step(const ds::hf::DeepSeekConfig& cfg, const AttentionWeights& attn, const float* hidden,
                     std::size_t pos, MLACache* cache, float* out_hidden, BackendKind backend) {
  if (cache == nullptr) throw std::runtime_error("mla: cache is null");
  if (pos >= cache->max_seq) throw std::runtime_error("mla: position out of range");

  const std::size_t hidden_size = static_cast<std::size_t>(cfg.hidden_size);
  const std::size_t n_heads = static_cast<std::size_t>(cfg.num_attention_heads);
  const std::size_t qk_nope = static_cast<std::size_t>(cfg.qk_nope_head_dim);
  const std::size_t qk_rope = static_cast<std::size_t>(cfg.qk_rope_head_dim);
  const std::size_t v_head = static_cast<std::size_t>(cfg.v_head_dim);
  const std::size_t q_head = qk_nope + qk_rope;
  const std::size_t kv_rank = static_cast<std::size_t>(cfg.kv_lora_rank);

  if (cache->n_heads != n_heads || cache->q_head_dim != q_head || cache->v_head_dim != v_head) {
    throw std::runtime_error("mla: cache shape mismatch");
  }

  std::vector<float> q_full(n_heads * q_head, 0.0f);
  if (attn.use_q_lora) {
    if (!attn.q_a_layernorm.valid()) throw std::runtime_error("mla: q_a_layernorm missing");
    const auto q_rank = static_cast<std::size_t>(attn.q_a_proj.weight->shape[0]);
    std::vector<float> q_a(q_rank, 0.0f);
    std::vector<float> q_a_norm(q_rank, 0.0f);
    linear_backend(backend, attn.q_a_proj, hidden, hidden_size, q_a.data(), q_rank);
    rmsnorm_backend(backend, q_a.data(), attn.q_a_layernorm.data(), q_rank, cfg.rms_norm_eps, q_a_norm.data());
    linear_backend(backend, attn.q_b_proj, q_a_norm.data(), q_rank, q_full.data(), n_heads * q_head);
  } else {
    linear_backend(backend, attn.q_proj, hidden, hidden_size, q_full.data(), n_heads * q_head);
  }

  std::vector<float> kv_a(kv_rank + qk_rope, 0.0f);
  linear_backend(backend, attn.kv_a_proj_with_mqa, hidden, hidden_size, kv_a.data(), kv_rank + qk_rope);

  std::vector<float> kv_norm(kv_rank, 0.0f);
  std::vector<float> kv_compressed(kv_rank, 0.0f);
  std::copy(kv_a.begin(), kv_a.begin() + static_cast<std::ptrdiff_t>(kv_rank), kv_compressed.begin());
  rmsnorm_backend(backend, kv_compressed.data(), attn.kv_a_layernorm.data(), kv_rank, cfg.rms_norm_eps, kv_norm.data());

  std::vector<float> kv_b(n_heads * (qk_nope + v_head), 0.0f);
  linear_backend(backend, attn.kv_b_proj, kv_norm.data(), kv_rank, kv_b.data(), n_heads * (qk_nope + v_head));

  std::vector<float> shared_k_rope(qk_rope, 0.0f);
  std::copy(kv_a.begin() + static_cast<std::ptrdiff_t>(kv_rank), kv_a.end(), shared_k_rope.begin());
  if (qk_rope > 0) ds::core::rope_inplace_f32(shared_k_rope.data(), qk_rope, pos, cfg.rope_theta);

  std::vector<float> attn_out(n_heads * v_head, 0.0f);
  const std::size_t seq_len = pos + 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(q_head));

  for (std::size_t head = 0; head < n_heads; ++head) {
    float* key_now = cache->key_ptr(pos, head);
    float* val_now = cache->val_ptr(pos, head);

    const float* q_src = q_full.data() + head * q_head;
    std::vector<float> q_vec(q_head, 0.0f);
    std::copy(q_src, q_src + static_cast<std::ptrdiff_t>(q_head), q_vec.begin());
    if (qk_rope > 0) ds::core::rope_inplace_f32(q_vec.data() + qk_nope, qk_rope, pos, cfg.rope_theta);

    const float* kv_src = kv_b.data() + head * (qk_nope + v_head);
    std::copy(kv_src, kv_src + static_cast<std::ptrdiff_t>(qk_nope), key_now);
    std::copy(shared_k_rope.begin(), shared_k_rope.end(), key_now + qk_nope);
    std::copy(kv_src + static_cast<std::ptrdiff_t>(qk_nope), kv_src + static_cast<std::ptrdiff_t>(qk_nope + v_head), val_now);

    std::vector<float> scores(seq_len, 0.0f);
    for (std::size_t t = 0; t < seq_len; ++t) {
      scores[t] = ds::core::dot_f32(q_vec.data(), cache->key_ptr(t, head), q_head) * scale;
    }
    ds::core::softmax_f32(scores.data(), scores.size());

    float* out_head = attn_out.data() + head * v_head;
    std::fill(out_head, out_head + static_cast<std::ptrdiff_t>(v_head), 0.0f);
    for (std::size_t t = 0; t < seq_len; ++t) {
      const float* vv = cache->val_ptr(t, head);
      for (std::size_t i = 0; i < v_head; ++i) out_head[i] += scores[t] * vv[i];
    }
  }

  linear_backend(backend, attn.o_proj, attn_out.data(), n_heads * v_head, out_hidden, hidden_size);
}

} // namespace ds::rt
