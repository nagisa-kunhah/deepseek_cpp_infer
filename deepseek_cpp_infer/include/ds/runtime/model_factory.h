#pragma once

#include "ds/hf/config.h"
#include "ds/runtime/model.h"

#include <memory>
#include <string>

namespace ds::rt {

std::shared_ptr<const Model> load_model(const std::string& model_dir);

} // namespace ds::rt
