#include "ds/hf/config.h"
#include "ds/hf/safetensors.h"
#include "ds/util/path.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {

void print_config(const ds::hf::DeepSeekConfig& cfg) {
  std::cout << "Loaded config:\n";
  std::cout << "  model_type: " << cfg.model_type << "\n";
  std::cout << "  layers: " << cfg.num_hidden_layers << "\n";
  std::cout << "  hidden: " << cfg.hidden_size << "\n";
  std::cout << "  heads: " << cfg.num_attention_heads << "\n";
  std::cout << "  kv_heads: " << cfg.num_key_value_heads << "\n";
  std::cout << "  vocab: " << cfg.vocab_size << "\n";
  std::cout << "  max_pos: " << cfg.max_position_embeddings << "\n";
  std::cout << "  rope_theta: " << cfg.rope_theta << "\n";
  std::cout << "  n_experts: " << cfg.n_experts << "\n";
  std::cout << "  moe_top_k: " << cfg.moe_top_k << "\n";
}

[[noreturn]] void usage() {
  throw std::runtime_error(
      "usage: ds_chat <model_dir> [info|verify]\n"
      "  info:   print config + shard list\n"
      "  verify: parse safetensors headers (default)\n");
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) usage();

    const std::string model_dir = argv[1];
    const std::string cmd = (argc >= 3) ? argv[2] : "verify";

    const std::string config_path = model_dir + "/config.json";
    if (!ds::util::file_exists(config_path)) {
      throw std::runtime_error("missing config.json in model_dir: " + config_path);
    }

    const auto cfg = ds::hf::load_config_json(config_path);

    auto shards = ds::util::list_files_with_suffix(model_dir, ".safetensors");
    std::sort(shards.begin(), shards.end());

    if (cmd == "info") {
      print_config(cfg);
      std::cout << "Found " << shards.size() << " safetensors shard(s).\n";
      for (const auto& s : shards) std::cout << "  " << s << "\n";
      return 0;
    }

    if (cmd == "verify") {
      print_config(cfg);
      if (shards.empty()) {
        throw std::runtime_error("no *.safetensors shards found in: " + model_dir);
      }

      std::uint64_t total_tensors = 0;
      for (const auto& s : shards) {
        const auto h = ds::hf::load_safetensors_header(s);
        std::cout << "Shard: " << s << " tensors=" << h.tensors.size() << "\n";
        total_tensors += static_cast<std::uint64_t>(h.tensors.size());
      }
      std::cout << "OK (parsed headers). total_tensors=" << total_tensors << "\n";
      return 0;
    }

    usage();
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
}
