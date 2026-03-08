#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ds::core {

// CPU tensor for activations (float32 only for bootstrap correctness).
struct Tensor {
  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<float> f32;

  std::size_t numel() const;
  void resize_like(const std::vector<std::int64_t>& new_shape);
};

} // namespace ds::core
