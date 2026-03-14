#include "ds/hf/config.h"

#include "ds/util/fs.h"

namespace ds::hf {

ModelConfig load_config_json(const std::string& config_path) {
  ModelConfig cfg;
  cfg.path = config_path;
  cfg.root = ds::util::parse_json(ds::util::read_text_file(config_path));
  cfg.model_type = ds::util::get_string_or(cfg.root, "model_type", "");
  return cfg;
}

} // namespace ds::hf
