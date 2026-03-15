#pragma once

#include "ds/hf/config.h"
#include "ds/hf/model_loader.h"
#include "ds/runtime/model.h"

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace ds::models::core {

struct IndexInspection {
  std::size_t key_count = 0;
  std::vector<std::string> layer_kinds;
};

struct LoadInspection {
  std::vector<std::string> layer_kinds;
};

struct ModelPackageDescriptor {
  const char* package_id = "";
  bool (*matches)(const ds::hf::ModelConfig& config) = nullptr;
  void (*print_config)(std::ostream& os, const ds::hf::ModelConfig& config) = nullptr;
  IndexInspection (*inspect_index)(const ds::hf::ModelConfig& config, const std::vector<std::string>& tensor_names) = nullptr;
  LoadInspection (*inspect_loaded)(const ds::hf::ModelConfig& config, const ds::hf::LoadedModel& model) = nullptr;
  std::shared_ptr<const ds::rt::Model> (*load)(const std::string& model_dir, const ds::hf::ModelConfig& config) = nullptr;
};

const ModelPackageDescriptor& resolve_package(const ds::hf::ModelConfig& config);

} // namespace ds::models::core
