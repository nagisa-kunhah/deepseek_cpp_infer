#include "ds/hf/weights_index.h"

#include "ds/util/fs.h"
#include "ds/util/json.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace ds::hf {

SafetensorsIndex load_safetensors_index(const std::string& path) {
  SafetensorsIndex idx;
  idx.path = path;

  const auto txt = ds::util::read_text_file(path);
  const auto j = ds::util::parse_json(txt);
  if (!j.is_object()) throw std::runtime_error("safetensors index: root is not an object");

  const auto* wm = j.find("weight_map");
  if (!wm || !wm->is_object()) throw std::runtime_error("safetensors index: missing weight_map object");

  std::unordered_set<std::string> uniq;
  uniq.reserve(wm->as_object().v.size());

  idx.tensor_names.reserve(wm->as_object().v.size());
  idx.tensor_shards.reserve(wm->as_object().v.size());

  // Preserve the same order for (tensor_names, tensor_shards) pairs.
  for (const auto& kv : wm->as_object().v) {
    idx.tensor_names.push_back(kv.first);

    const auto& v = *kv.second;
    idx.tensor_shards.push_back(v.is_string() ? v.as_string() : std::string{});

    if (v.is_string()) uniq.emplace(v.as_string());
  }

  // Sort pairs by tensor name for stable lookups.
  std::vector<std::size_t> order(idx.tensor_names.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return idx.tensor_names[a] < idx.tensor_names[b];
  });

  std::vector<std::string> names_sorted;
  std::vector<std::string> shards_sorted;
  names_sorted.reserve(order.size());
  shards_sorted.reserve(order.size());
  for (const auto i : order) {
    names_sorted.push_back(std::move(idx.tensor_names[i]));
    shards_sorted.push_back(std::move(idx.tensor_shards[i]));
  }
  idx.tensor_names = std::move(names_sorted);
  idx.tensor_shards = std::move(shards_sorted);

  idx.shard_filenames.assign(uniq.begin(), uniq.end());
  std::sort(idx.shard_filenames.begin(), idx.shard_filenames.end());
  return idx;
}

std::string find_tensor_shard(const SafetensorsIndex& idx, const std::string& tensor_name) {
  auto it = std::lower_bound(idx.tensor_names.begin(), idx.tensor_names.end(), tensor_name);
  if (it == idx.tensor_names.end() || *it != tensor_name) return {};
  const auto pos = static_cast<std::size_t>(it - idx.tensor_names.begin());
  if (pos >= idx.tensor_shards.size()) return {};
  return idx.tensor_shards[pos];
}

} // namespace ds::hf
