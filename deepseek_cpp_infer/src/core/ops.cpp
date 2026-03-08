#include "ds/core/ops.h"

#include <cmath>

namespace ds::core {

void linear_f32(const float* x, std::size_t in, const float* w, std::size_t out, float* y) {
  // w is row-major [out, in]
  for (std::size_t o = 0; o < out; ++o) {
    const float* row = w + o * in;
    float acc = 0.0f;
    for (std::size_t i = 0; i < in; ++i) acc += x[i] * row[i];
    y[o] = acc;
  }
}

void rmsnorm_f32(const float* x, const float* w, std::size_t n, float eps, float* y) {
  float mean2 = 0.0f;
  for (std::size_t i = 0; i < n; ++i) mean2 += x[i] * x[i];
  mean2 /= static_cast<float>(n);
  const float inv = 1.0f / std::sqrt(mean2 + eps);
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

} // namespace ds::core
