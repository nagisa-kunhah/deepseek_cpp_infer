#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ds::core {

float dot_f32(const float* a, const float* b, std::size_t n);

// In-place stable softmax.
void softmax_f32(float* x, std::size_t n);

// Return index of max element.
std::size_t argmax_f32(const float* x, std::size_t n);

} // namespace ds::core
