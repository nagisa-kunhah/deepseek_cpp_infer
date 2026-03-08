#pragma once

#include "ds/core/tensor.h"

#include <cstddef>

namespace ds::core {

// y = x * W^T, where x is [in], W is [out, in], y is [out].
void linear_f32(const float* x, std::size_t in, const float* w, std::size_t out, float* y);

// RMSNorm: y[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)
void rmsnorm_f32(const float* x, const float* w, std::size_t n, float eps, float* y);

} // namespace ds::core
