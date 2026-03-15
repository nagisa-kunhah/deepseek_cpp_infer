#include "ds/runtime/model_factory.h"

#include "ds/hf/config.h"
#include "ds/hf/model_loader.h"
#include "ds/runtime/deepseek_model.h"

#include <cctype>
#include <stdexcept>

namespace ds::rt {
namespace {

std::string lower_ascii(std::string s) {
  for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

} // namespace

ModelFamily detect_model_family(const std::string& model_type) {
  const auto lowered = lower_ascii(model_type);
  if (lowered.find("deepseek") != std::string::npos) return ModelFamily::DeepSeek;
  throw std::runtime_error("unsupported model_type: " + model_type);
}

std::unique_ptr<Model> load_model(const std::string& model_dir) {
  const auto cfg = ds::hf::load_config_json(model_dir + "/config.json");
  switch (detect_model_family(cfg.model_type)) {
    case ModelFamily::DeepSeek: return std::make_unique<DeepSeekModel>(cfg, ds::hf::load_model_dir(model_dir));
  }
  throw std::runtime_error("unsupported model family for model_type: " + cfg.model_type);
}

} // namespace ds::rt
