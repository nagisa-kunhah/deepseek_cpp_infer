#include "ds/core/math.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ds::core {

float dot_f32(const float* a, const float* b, std::size_t n) {
  float acc = 0.0f;
  for (std::size_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

void softmax_f32(float* x, std::size_t n) {
  if (n == 0) return;
  float mx = x[0];
  for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, x[i]);
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  if (!(sum > 0.0f)) throw std::runtime_error("softmax: sum <= 0");
  const float inv = 1.0f / sum;
  for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
}

std::size_t argmax_f32(const float* x, std::size_t n) {
  if (n == 0) return 0;
  std::size_t best = 0;
  float bv = x[0];
  for (std::size_t i = 1; i < n; ++i) {
    if (x[i] > bv) {
      bv = x[i];
      best = i;
    }
  }
  return best;
}

} // namespace ds::core
