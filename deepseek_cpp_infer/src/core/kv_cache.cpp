#include "ds/core/kv_cache.h"

#include <stdexcept>

namespace ds::core {

void KVCache::init(std::size_t max_seq_, std::size_t n_kv_heads_, std::size_t head_dim_) {
  if (max_seq_ == 0 || n_kv_heads_ == 0 || head_dim_ == 0) throw std::runtime_error("kvcache: bad dims");
  max_seq = max_seq_;
  n_kv_heads = n_kv_heads_;
  head_dim = head_dim_;
  k.assign(max_seq * n_kv_heads * head_dim, 0.0f);
  v.assign(max_seq * n_kv_heads * head_dim, 0.0f);
}

static std::size_t idx(std::size_t pos, std::size_t h, std::size_t head_dim, std::size_t n_kv_heads) {
  return (pos * n_kv_heads + h) * head_dim;
}

float* KVCache::key_ptr(std::size_t pos, std::size_t h) {
  if (pos >= max_seq || h >= n_kv_heads) throw std::runtime_error("kvcache: out of range");
  return k.data() + idx(pos, h, head_dim, n_kv_heads);
}

float* KVCache::val_ptr(std::size_t pos, std::size_t h) {
  if (pos >= max_seq || h >= n_kv_heads) throw std::runtime_error("kvcache: out of range");
  return v.data() + idx(pos, h, head_dim, n_kv_heads);
}

const float* KVCache::key_ptr(std::size_t pos, std::size_t h) const {
  if (pos >= max_seq || h >= n_kv_heads) throw std::runtime_error("kvcache: out of range");
  return k.data() + idx(pos, h, head_dim, n_kv_heads);
}

const float* KVCache::val_ptr(std::size_t pos, std::size_t h) const {
  if (pos >= max_seq || h >= n_kv_heads) throw std::runtime_error("kvcache: out of range");
  return v.data() + idx(pos, h, head_dim, n_kv_heads);
}

} // namespace ds::core
