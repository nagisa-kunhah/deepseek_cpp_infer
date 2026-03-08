#pragma once

#include <cstddef>

namespace ds::core {

// Apply RoPE to a single vector in-place.
// x shape: [dim] where dim is even, rotary dims = dim (bootstrap).
// pos is the absolute position.
void rope_inplace_f32(float* x, std::size_t dim, std::size_t pos, float theta);

} // namespace ds::core
