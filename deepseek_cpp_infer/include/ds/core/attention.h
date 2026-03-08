#pragma once

#include "ds/core/kv_cache.h"

#include <cstddef>

namespace ds::core {

// Naive single-token attention for one head.
// q: [head_dim]
// cache: keys/values for this kv_head for positions [0, seq_len)
// out: [head_dim]
void attn_one_head_f32(const float* q, std::size_t head_dim, const KVCache& cache, std::size_t kv_head,
                      std::size_t seq_len, float* out);

} // namespace ds::core
