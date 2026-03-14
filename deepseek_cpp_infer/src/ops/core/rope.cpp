#include "ds/core/rope.h"

#include <cmath>
#include <stdexcept>

namespace ds::core {

void rope_inplace_f32(float* x, std::size_t dim, std::size_t pos, float theta) {
  if ((dim % 2) != 0) throw std::runtime_error("rope: dim must be even");
  // Standard RoPE: rotate each pair (2i, 2i+1) with frequency theta^{-2i/dim}.
  for (std::size_t i = 0; i < dim; i += 2) {
    const std::size_t k = i / 2;
    const float inv_freq = std::pow(theta, -static_cast<float>(2.0 * k) / static_cast<float>(dim));
    const float ang = static_cast<float>(pos) * inv_freq;
    const float c = std::cos(ang);
    const float s = std::sin(ang);

    const float x0 = x[i];
    const float x1 = x[i + 1];
    x[i] = x0 * c - x1 * s;
    x[i + 1] = x0 * s + x1 * c;
  }
}

} // namespace ds::core
