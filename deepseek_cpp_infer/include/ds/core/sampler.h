#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace ds::core {

struct SamplerConfig {
  float temperature = 1.0f;
  int top_k = 0;      // 0 disables
  float top_p = 0.0f; // 0 disables
  std::uint32_t seed = 0;
};

class Sampler {
 public:
  explicit Sampler(SamplerConfig cfg);

  // Sample from logits (size=vocab). Returns token id.
  int sample(const std::vector<float>& logits);

 private:
  SamplerConfig cfg_;
  std::mt19937 rng_;
};

} // namespace ds::core
