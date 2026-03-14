#include "ds/runtime/backend.h"

#include <stdexcept>

namespace ds::rt {

BackendKind parse_backend_kind(const std::string& s) {
  if (s == "cpu") return BackendKind::CPU;
  if (s == "cuda") return BackendKind::CUDA;
  throw std::runtime_error("backend must be one of: cpu, cuda");
}

std::string to_string(BackendKind kind) {
  switch (kind) {
    case BackendKind::CPU: return "cpu";
    case BackendKind::CUDA: return "cuda";
  }
  return "unknown";
}

} // namespace ds::rt
