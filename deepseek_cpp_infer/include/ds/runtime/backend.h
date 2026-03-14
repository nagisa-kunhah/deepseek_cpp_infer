#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ds::rt {

enum class BackendKind {
  CPU,
  CUDA,
};

struct RunConfig {
  BackendKind backend = BackendKind::CPU;
  std::size_t max_seq = 0;
  bool verbose = false;
};

struct GenerationConfig {
  std::int32_t max_new_tokens = 1;
  float temperature = 0.0f;
  std::int32_t top_k = 0;
  float top_p = 0.0f;
  std::uint32_t seed = 0;
};

BackendKind parse_backend_kind(const std::string& s);
std::string to_string(BackendKind kind);

} // namespace ds::rt
