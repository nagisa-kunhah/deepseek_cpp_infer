#pragma once

#include "ds/core/dtype.h"
#include "ds/hf/model_loader.h"

#include <cstddef>
#include <vector>

namespace ds::hf {

// Decode a tensor slice into float32. Intended for small/medium tensors during bootstrap.
// For huge weights (e.g. expert matrices), we will later implement blocked decode/compute.
std::vector<float> decode_to_f32(const TensorSlice& t);

float read_scalar_f32(const TensorSlice& t, std::size_t idx);

} // namespace ds::hf
