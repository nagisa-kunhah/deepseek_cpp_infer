#pragma once

#include <string>
#include <vector>

namespace ds::hf {

struct SafetensorsIndex {
  std::string path;
  std::vector<std::string> shard_filenames; // e.g. model-00001-of-000004.safetensors
  std::vector<std::string> tensor_names;    // keys of weight_map
};

// Parse HuggingFace `model.safetensors.index.json` and return shard filenames + tensor names.
SafetensorsIndex load_safetensors_index(const std::string& path);

} // namespace ds::hf
