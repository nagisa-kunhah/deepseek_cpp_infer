#include "ds/core/mha.h"

#include "ds/core/attention.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ds::core {

void mha_decode_f32(const float* q, const float* k, const float* v, std::size_t n_heads, std::size_t n_kv_heads,
                    std::size_t head_dim, KVCache& cache, std::size_t pos, float* out) {
  if (n_heads == 0 || n_kv_heads == 0 || head_dim == 0) throw std::runtime_error("mha: bad dims");
  if (cache.max_seq == 0) throw std::runtime_error("mha: cache not initialized");
  if (pos >= cache.max_seq) throw std::runtime_error("mha: pos out of range");
  if (cache.n_kv_heads != n_kv_heads || cache.head_dim != head_dim) throw std::runtime_error("mha: cache shape mismatch");

  // Write k/v into cache at this position.
  for (std::size_t h = 0; h < n_kv_heads; ++h) {
    std::memcpy(cache.key_ptr(pos, h), k + h * head_dim, head_dim * sizeof(float));
    std::memcpy(cache.val_ptr(pos, h), v + h * head_dim, head_dim * sizeof(float));
  }

  // Compute attention per head. For GQA/MQA: map query head -> kv head.
  const std::size_t seq_len = pos + 1;
  std::vector<float> tmp;
  tmp.resize(head_dim);

  for (std::size_t h = 0; h < n_heads; ++h) {
    const std::size_t kvh = (n_kv_heads == n_heads) ? h : (h * n_kv_heads) / n_heads;
    const float* qh = q + h * head_dim;
    attn_one_head_f32(qh, head_dim, cache, kvh, seq_len, tmp.data());
    std::memcpy(out + h * head_dim, tmp.data(), head_dim * sizeof(float));
  }
}

} // namespace ds::core
