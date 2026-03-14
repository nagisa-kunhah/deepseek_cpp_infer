#include "ds/models/core/registry.h"

#include "ds/models/deepseek/package.h"

#include <stdexcept>

namespace ds::models::core {
namespace {

const ModelPackageDescriptor* builtin_packages() {
  static const ModelPackageDescriptor packages[] = {
      ds::models::deepseek::package_descriptor(),
  };
  return packages;
}

} // namespace

const ModelPackageDescriptor& resolve_package(const ds::hf::ModelConfig& config) {
  static constexpr std::size_t kNumPackages = 1;
  const auto* packages = builtin_packages();
  for (std::size_t i = 0; i < kNumPackages; ++i) {
    if (packages[i].matches != nullptr && packages[i].matches(config)) return packages[i];
  }
  throw std::runtime_error("unsupported model_type: " + config.model_type);
}

} // namespace ds::models::core
