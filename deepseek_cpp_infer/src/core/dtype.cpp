#include "ds/core/dtype.h"

#include <cstdint>
#include <cstring>

namespace ds::core {

float bf16_to_f32(bf16 x) {
  // BF16 stores the high 16 bits of IEEE754 float.
  std::uint32_t u = static_cast<std::uint32_t>(x.v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

static float f16_to_f32_impl(std::uint16_t h) {
  // IEEE754 half -> float (software).
  const std::uint32_t sign = (h >> 15) & 0x1;
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;

  std::uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign << 31;
    } else {
      // Subnormal.
      std::uint32_t m = mant;
      std::uint32_t e = 127 - 15 + 1;
      while ((m & 0x400) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FF;
      out = (sign << 31) | (e << 23) | (m << 13);
    }
  } else if (exp == 31) {
    // Inf/NaN.
    out = (sign << 31) | (0xFF << 23) | (mant << 13);
  } else {
    // Normalized.
    const std::uint32_t e = exp + (127 - 15);
    out = (sign << 31) | (e << 23) | (mant << 13);
  }

  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

float f16_to_f32(f16 x) { return f16_to_f32_impl(x.v); }

} // namespace ds::core
