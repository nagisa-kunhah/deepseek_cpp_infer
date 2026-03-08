#pragma once

#include <cstdint>

namespace ds::core {

// Minimal fp16/bf16 helpers for CPU decode.

struct bf16 {
  std::uint16_t v;
};

struct f16 {
  std::uint16_t v;
};

float bf16_to_f32(bf16 x);
float f16_to_f32(f16 x);

} // namespace ds::core
