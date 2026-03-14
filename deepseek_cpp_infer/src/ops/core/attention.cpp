#include "ds/core/attention.h"

#include "ds/core/math.h"

#include <cmath>
#include <vector>

namespace ds::core {

void attn_one_head_f32(const float* q, std::size_t head_dim, const KVCache& cache, std::size_t kv_head,
                      std::size_t seq_len, float* out) {
  // scores[t] = q dot k[t] / sqrt(d)
  std::vector<float> scores;
  scores.resize(seq_len);

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (std::size_t t = 0; t < seq_len; ++t) {
    const float* k = cache.key_ptr(t, kv_head);
    scores[t] = dot_f32(q, k, head_dim) * scale;
  }

  softmax_f32(scores.data(), scores.size());

  // out = sum_t p[t] * v[t]
  for (std::size_t i = 0; i < head_dim; ++i) out[i] = 0.0f;
  for (std::size_t t = 0; t < seq_len; ++t) {
    const float* v = cache.val_ptr(t, kv_head);
    const float p = scores[t];
    for (std::size_t i = 0; i < head_dim; ++i) out[i] += p * v[i];
  }
}

} // namespace ds::core
