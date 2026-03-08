#include "ds/hf/model_loader.h"

#include "ds/util/path.h"

#include <stdexcept>

namespace ds::hf {
namespace {

std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

} // namespace

LoadedModel load_model_dir(const std::string& model_dir) {
  LoadedModel m;
  m.model_dir = model_dir;

  const std::string idx_path = join_path(model_dir, "model.safetensors.index.json");
  if (!ds::util::file_exists(idx_path)) {
    throw std::runtime_error("missing model.safetensors.index.json: " + idx_path);
  }

  const auto idx = load_safetensors_index(idx_path);
  if (idx.shard_filenames.empty()) {
    throw std::runtime_error("index has no shards: " + idx_path);
  }

  // Open/mmap each shard.
  m.shards.reserve(idx.shard_filenames.size());
  for (const auto& shard : idx.shard_filenames) {
    const auto path = join_path(model_dir, shard);
    if (!ds::util::file_exists(path)) {
      throw std::runtime_error("missing shard: " + path);
    }

    LoadedShard ls;
    ls.file.open_readonly(path);
    ls.header = load_safetensors_header(path);
    m.shards.push_back(std::move(ls));
  }

  // Build a quick lookup shard_path -> shard index.
  std::unordered_map<std::string, std::size_t> shard_idx;
  shard_idx.reserve(m.shards.size());
  for (std::size_t i = 0; i < m.shards.size(); ++i) {
    const auto sp = m.shards[i].file.path();
    shard_idx.emplace(sp, i);
  }

  // Resolve tensor views.
  m.tensors.reserve(idx.tensor_names.size());
  for (const auto& name : idx.tensor_names) {
    const auto shard_file = find_tensor_shard(idx, name);
    if (shard_file.empty()) {
      throw std::runtime_error("index missing shard for tensor: " + name);
    }

    const auto shard_path = join_path(model_dir, shard_file);
    auto it = shard_idx.find(shard_path);
    if (it == shard_idx.end()) {
      throw std::runtime_error("internal: shard not opened: " + shard_path);
    }

    auto& s = m.shards[it->second];
    auto ht = s.header.tensors.find(name);
    if (ht == s.header.tensors.end()) {
      // Index says tensor is in this shard, but header disagrees.
      throw std::runtime_error("tensor not found in shard header: " + name + " shard=" + shard_path);
    }

    const auto& meta = ht->second;
    const auto abs_start = s.header.data_offset + meta.data_start;
    const auto abs_end = s.header.data_offset + meta.data_end;
    if (abs_end < abs_start || abs_end > s.file.size()) {
      throw std::runtime_error("tensor out of range: " + name);
    }

    TensorSlice tv;
    tv.name = name;
    tv.dtype = meta.dtype;
    tv.shape = meta.shape;
    tv.data = s.file.data() + abs_start;
    tv.nbytes = static_cast<std::size_t>(abs_end - abs_start);
    tv.shard_path = shard_path;

    m.tensors.emplace(name, std::move(tv));
  }

  return m;
}

} // namespace ds::hf
