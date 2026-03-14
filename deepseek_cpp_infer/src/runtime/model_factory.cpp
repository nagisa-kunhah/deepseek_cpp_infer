#include "ds/runtime/model_factory.h"

#include "ds/models/core/registry.h"

namespace ds::rt {
std::shared_ptr<const Model> load_model(const std::string& model_dir) {
  const auto cfg = ds::hf::load_config_json(model_dir + "/config.json");
  const auto& package = ds::models::core::resolve_package(cfg);
  return package.load(model_dir, cfg);
}

} // namespace ds::rt
