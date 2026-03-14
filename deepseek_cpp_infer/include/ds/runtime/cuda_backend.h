#pragma once

#include "ds/hf/model_loader.h"

#include <cstddef>

namespace ds::rt::cuda {

bool available();
void ensure_initialized();
bool linear_try(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y, std::size_t out);
bool rmsnorm_try(const float* x, const float* w, std::size_t n, float eps, float* y);

} // namespace ds::rt::cuda
