#pragma once

#include "ds/hf/model_loader.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ds::hf {

// Embedding lookup: emb weight is [vocab, hidden]. Writes hidden floats.
void embedding_lookup_f32(const TensorSlice& emb_weight, std::int32_t token_id, float* out_hidden);

// Greedy argmax over vocab for lm_head weight [vocab, hidden].
// Returns token id with max logit.
std::int32_t lm_head_argmax(const TensorSlice& lm_head_weight, const float* hidden);

} // namespace ds::hf
