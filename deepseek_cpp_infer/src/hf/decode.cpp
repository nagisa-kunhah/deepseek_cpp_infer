#include "ds/hf/decode.h"

#include "ds/hf/safetensors.h"

#include <cstring>
#include <stdexcept>

namespace ds::hf {

static std::size_t numel(const std::vector<std::int64_t>& shape) {
  std::size_t n = 1;
  for (auto d : shape) {
    if (d <= 0) throw std::runtime_error("decode: invalid shape");
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

float read_scalar_f32(const TensorSlice& t, std::size_t idx) {
  const auto n = numel(t.shape);
  if (idx >= n) throw std::runtime_error("decode: index out of range");

  switch (t.dtype) {
    case DType::F32: {
      float v;
      std::memcpy(&v, t.data + idx * 4, 4);
      return v;
    }
    case DType::BF16: {
      ds::core::bf16 b;
      std::memcpy(&b.v, t.data + idx * 2, 2);
      return ds::core::bf16_to_f32(b);
    }
    case DType::F16: {
      ds::core::f16 h;
      std::memcpy(&h.v, t.data + idx * 2, 2);
      return ds::core::f16_to_f32(h);
    }
    default:
      throw std::runtime_error("decode: unsupported dtype");
  }
}

std::vector<float> decode_to_f32(const TensorSlice& t) {
  const auto n = numel(t.shape);
  const auto elem = ds::hf::dtype_nbytes(t.dtype);
  if (elem == 0) throw std::runtime_error("decode: unknown dtype");
  if (t.nbytes != n * elem) {
    // Safetensors stores exact payload sizes; mismatch usually means wrong offsets.
    throw std::runtime_error("decode: nbytes mismatch");
  }

  std::vector<float> out;
  out.resize(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = read_scalar_f32(t, i);
  return out;
}

} // namespace ds::hf
