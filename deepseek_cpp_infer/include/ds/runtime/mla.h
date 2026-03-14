#pragma once

#include "ds/hf/config.h"
#include "ds/runtime/backend.h"
#include "ds/runtime/weights.h"

#include <cstddef>
#include <vector>

namespace ds::rt {

struct MLACache {
  std::size_t max_seq = 0;
  std::size_t n_heads = 0;
  std::size_t q_head_dim = 0;
  std::size_t v_head_dim = 0;

  std::vector<float> k;
  std::vector<float> v;

  void init(std::size_t max_seq_, std::size_t n_heads_, std::size_t q_head_dim_, std::size_t v_head_dim_);

  float* key_ptr(std::size_t pos, std::size_t head);
  float* val_ptr(std::size_t pos, std::size_t head);
  const float* key_ptr(std::size_t pos, std::size_t head) const;
  const float* val_ptr(std::size_t pos, std::size_t head) const;
};

void mla_decode_step(const ds::hf::DeepSeekConfig& cfg, const AttentionWeights& attn, const float* hidden,
                     std::size_t pos, MLACache* cache, float* out_hidden, BackendKind backend);

} // namespace ds::rt
