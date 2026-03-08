#pragma once

#include "ds/core/kv_cache.h"

#include <cstddef>

namespace ds::core {

// Naive multi-head attention for a single token.
// - q: [n_heads, head_dim]
// - k/v: [n_kv_heads, head_dim]
// - kv cache is updated at `pos` before computing output.
// - out: [n_heads, head_dim]
// Mapping: for MHA, n_kv_heads == n_heads; for GQA/MQA, heads share kv heads.
void mha_decode_f32(const float* q, const float* k, const float* v, std::size_t n_heads, std::size_t n_kv_heads,
                    std::size_t head_dim, KVCache& cache, std::size_t pos, float* out);

} // namespace ds::core
