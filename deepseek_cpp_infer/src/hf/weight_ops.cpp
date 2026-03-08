#include "ds/hf/weight_ops.h"

#include "ds/core/dtype.h"
#include "ds/hf/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ds::hf {
namespace {

std::size_t numel(const std::vector<std::int64_t>& shape) {
  std::size_t n = 1;
  for (auto d : shape) {
    if (d <= 0) throw std::runtime_error("weight_ops: invalid shape");
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

float read_bf16(const std::uint8_t* p) {
  ds::core::bf16 b;
  std::memcpy(&b.v, p, 2);
  return ds::core::bf16_to_f32(b);
}

float read_f16(const std::uint8_t* p) {
  ds::core::f16 h;
  std::memcpy(&h.v, p, 2);
  return ds::core::f16_to_f32(h);
}

} // namespace

void embedding_lookup_f32(const TensorSlice& emb_weight, std::int32_t token_id, float* out_hidden) {
  if (emb_weight.shape.size() != 2) throw std::runtime_error("embedding: expected 2D weight");
  const auto vocab = static_cast<std::size_t>(emb_weight.shape[0]);
  const auto hidden = static_cast<std::size_t>(emb_weight.shape[1]);
  if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab) throw std::runtime_error("embedding: token out of range");

  const std::size_t row = static_cast<std::size_t>(token_id);
  const std::size_t elem = ds::hf::dtype_nbytes(emb_weight.dtype);
  if (elem == 0) throw std::runtime_error("embedding: unknown dtype");
  if (emb_weight.nbytes != vocab * hidden * elem) throw std::runtime_error("embedding: nbytes mismatch");

  const std::uint8_t* base = emb_weight.data + row * hidden * elem;
  switch (emb_weight.dtype) {
    case DType::F32: {
      std::memcpy(out_hidden, base, hidden * sizeof(float));
      return;
    }
    case DType::BF16: {
      for (std::size_t i = 0; i < hidden; ++i) out_hidden[i] = read_bf16(base + i * 2);
      return;
    }
    case DType::F16: {
      for (std::size_t i = 0; i < hidden; ++i) out_hidden[i] = read_f16(base + i * 2);
      return;
    }
    default:
      throw std::runtime_error("embedding: unsupported dtype");
  }
}

std::int32_t lm_head_argmax(const TensorSlice& lm_head_weight, const float* hidden_vec) {
  if (lm_head_weight.shape.size() != 2) throw std::runtime_error("lm_head: expected 2D weight");
  const auto vocab = static_cast<std::size_t>(lm_head_weight.shape[0]);
  const auto hidden = static_cast<std::size_t>(lm_head_weight.shape[1]);

  const std::size_t elem = ds::hf::dtype_nbytes(lm_head_weight.dtype);
  if (elem == 0) throw std::runtime_error("lm_head: unknown dtype");
  if (lm_head_weight.nbytes != vocab * hidden * elem) throw std::runtime_error("lm_head: nbytes mismatch");

  float best = -INFINITY;
  std::int32_t best_id = 0;

  // Greedy scan over vocab. Very slow but streaming (no huge logits buffer).
  for (std::size_t tok = 0; tok < vocab; ++tok) {
    const std::uint8_t* row = lm_head_weight.data + tok * hidden * elem;
    float acc = 0.0f;

    switch (lm_head_weight.dtype) {
      case DType::F32: {
        const float* w = reinterpret_cast<const float*>(row);
        for (std::size_t i = 0; i < hidden; ++i) acc += hidden_vec[i] * w[i];
        break;
      }
      case DType::BF16: {
        for (std::size_t i = 0; i < hidden; ++i) acc += hidden_vec[i] * read_bf16(row + i * 2);
        break;
      }
      case DType::F16: {
        for (std::size_t i = 0; i < hidden; ++i) acc += hidden_vec[i] * read_f16(row + i * 2);
        break;
      }
      default:
        throw std::runtime_error("lm_head: unsupported dtype");
    }

    if (acc > best) {
      best = acc;
      best_id = static_cast<std::int32_t>(tok);
    }
  }

  return best_id;
}

} // namespace ds::hf
