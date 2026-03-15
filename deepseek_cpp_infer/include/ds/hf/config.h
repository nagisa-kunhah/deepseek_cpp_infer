#pragma once

#include "ds/util/json.h"

#include <string>

namespace ds::hf {

struct ModelConfig {
  std::string path;
  std::string model_type;
  ds::util::Json root;
};

ModelConfig load_config_json(const std::string& config_path);

} // namespace ds::hf
