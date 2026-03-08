#include "ds/core/tensor.h"

#include <stdexcept>

namespace ds::core {

static std::size_t numel_of(const std::vector<std::int64_t>& shape) {
  std::size_t n = 1;
  for (auto d : shape) {
    if (d <= 0) throw std::runtime_error("tensor: invalid shape");
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

std::size_t Tensor::numel() const { return numel_of(shape); }

void Tensor::resize_like(const std::vector<std::int64_t>& new_shape) {
  shape = new_shape;
  f32.resize(numel_of(shape));
}

} // namespace ds::core
