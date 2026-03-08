#pragma once

#include "ds/hf/safetensors.h"
#include "ds/hf/weights_index.h"
#include "ds/util/mmap.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ds::hf {

struct LoadedShard {
  ds::util::MMapFile file;
  SafeTensorsHeader header;
};

struct TensorSlice {
  std::string name;
  DType dtype = DType::UNKNOWN;
  std::vector<std::int64_t> shape;
  const std::uint8_t* data = nullptr;
  std::size_t nbytes = 0;
  std::string shard_path;
};

struct LoadedModel {
  std::string model_dir;
  std::unordered_map<std::string, TensorSlice> tensors;
  std::vector<LoadedShard> shards; // keep mmap alive
};

// "Load" means: open shards (mmap) + use index to resolve tensor -> shard -> offsets,
// then create non-owning views. This does not run inference yet.
LoadedModel load_model_dir(const std::string& model_dir);

} // namespace ds::hf
