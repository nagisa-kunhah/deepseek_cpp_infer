#pragma once

#include <cstddef>
#include <vector>

namespace ds::core {

// CPU KV cache for a single layer.
// Layout (bootstrap):
// - keys:  [max_seq, n_kv_heads, head_dim]
// - values:[max_seq, n_kv_heads, head_dim]
struct KVCache {
  std::size_t max_seq = 0;
  std::size_t n_kv_heads = 0;
  std::size_t head_dim = 0;

  std::vector<float> k; // size = max_seq * n_kv_heads * head_dim
  std::vector<float> v; // size = max_seq * n_kv_heads * head_dim

  void init(std::size_t max_seq_, std::size_t n_kv_heads_, std::size_t head_dim_);

  float* key_ptr(std::size_t pos, std::size_t h);
  float* val_ptr(std::size_t pos, std::size_t h);
  const float* key_ptr(std::size_t pos, std::size_t h) const;
  const float* val_ptr(std::size_t pos, std::size_t h) const;
};

} // namespace ds::core
