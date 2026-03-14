#pragma once

#include "ds/runtime/model.h"

#include <memory>
#include <string>

namespace ds::rt {

ModelFamily detect_model_family(const std::string& model_type);
std::unique_ptr<Model> load_model(const std::string& model_dir);

} // namespace ds::rt
