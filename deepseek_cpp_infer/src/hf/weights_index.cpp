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

  for (const auto& kv : wm->as_object().v) {
    const auto& v = *kv.second;
    if (!v.is_string()) continue;
    uniq.emplace(v.as_string());
  }

  idx.shard_filenames.assign(uniq.begin(), uniq.end());
  std::sort(idx.shard_filenames.begin(), idx.shard_filenames.end());
  return idx;
}

} // namespace ds::hf
