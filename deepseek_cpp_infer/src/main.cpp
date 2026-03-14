#include "ds/hf/config.h"
#include "ds/hf/model_loader.h"
#include "ds/hf/safetensors.h"
#include "ds/hf/weights_index.h"
#include "ds/runtime/backend.h"
#include "ds/runtime/model_executor.h"
#include "ds/runtime/tokenizer.h"
#include "ds/runtime/weights.h"
#include "ds/util/path.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace {

bool has_prefix(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

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
  std::cout << "  n_shared_experts: " << cfg.n_shared_experts << "\n";
  std::cout << "  moe_top_k: " << cfg.moe_top_k << "\n";
  std::cout << "  kv_lora_rank: " << cfg.kv_lora_rank << "\n";
  std::cout << "  qk_nope_head_dim: " << cfg.qk_nope_head_dim << "\n";
  std::cout << "  qk_rope_head_dim: " << cfg.qk_rope_head_dim << "\n";
  std::cout << "  v_head_dim: " << cfg.v_head_dim << "\n";
}

[[noreturn]] void usage() {
  throw std::runtime_error(
      "usage: ds_chat <model_dir> [info|verify|strict|load|run|generate] [options]\n"
      "  info:   print config + index + shard list\n"
      "  verify: validate required tensor keys via index; if shards exist, parse headers (default)\n"
      "  strict: verify + print dense/MoE layer map inferred from index\n"
      "  load:   mmap shards and build tensor views\n"
      "  run:    execute prompt prefill and print the next-token prediction\n"
      "  generate: autoregressive decode\n"
      "options for run/generate:\n"
      "  --prompt <text>\n"
      "  --prompt-ids <id0,id1,...>\n"
      "  --max-new-tokens <n>\n"
      "  --temperature <float>\n"
      "  --top-k <int>\n"
      "  --top-p <float>\n"
      "  --backend <cpu|cuda>\n"
      "  --seed <int>\n");
}

struct CmdOptions {
  std::string prompt;
  std::string prompt_ids_csv;
  std::int32_t max_new_tokens = 1;
  float temperature = 0.0f;
  std::int32_t top_k = 0;
  float top_p = 0.0f;
  std::string backend = "cpu";
  std::uint32_t seed = 0;
};

std::vector<std::int32_t> parse_prompt_ids_csv(const std::string& csv) {
  if (csv.empty()) return {};
  std::vector<std::int32_t> ids;
  std::size_t start = 0;
  while (start < csv.size()) {
    const auto end = csv.find(',', start);
    const auto token = csv.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!token.empty()) ids.push_back(static_cast<std::int32_t>(std::stol(token)));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return ids;
}

CmdOptions parse_cmd_options(int argc, char** argv, int start_arg) {
  CmdOptions opts;
  for (int i = start_arg; i < argc; ++i) {
    const std::string_view key = argv[i];
    auto require_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
      ++i;
      return argv[i];
    };

    if (key == "--prompt") {
      opts.prompt = require_value("--prompt");
    } else if (key == "--prompt-ids") {
      opts.prompt_ids_csv = require_value("--prompt-ids");
    } else if (key == "--max-new-tokens") {
      opts.max_new_tokens = static_cast<std::int32_t>(std::stol(require_value("--max-new-tokens")));
    } else if (key == "--temperature") {
      opts.temperature = std::stof(require_value("--temperature"));
    } else if (key == "--top-k") {
      opts.top_k = static_cast<std::int32_t>(std::stol(require_value("--top-k")));
    } else if (key == "--top-p") {
      opts.top_p = std::stof(require_value("--top-p"));
    } else if (key == "--backend") {
      opts.backend = require_value("--backend");
    } else if (key == "--seed") {
      opts.seed = static_cast<std::uint32_t>(std::stoul(require_value("--seed")));
    } else {
      throw std::runtime_error("unknown option: " + std::string(key));
    }
  }
  return opts;
}

std::vector<std::int32_t> resolve_prompt_ids(const CmdOptions& opts, const std::string& tok_json, bool has_tokenizer,
                                             ds::rt::Tokenizer* tokenizer) {
  if (!opts.prompt_ids_csv.empty()) return parse_prompt_ids_csv(opts.prompt_ids_csv);
  if (opts.prompt.empty()) {
    throw std::runtime_error("run/generate requires --prompt or --prompt-ids");
  }
  if (!has_tokenizer) {
    throw std::runtime_error("tokenizer.json is required to encode --prompt; use --prompt-ids instead");
  }
  *tokenizer = ds::rt::Tokenizer::load_from_file(tok_json);
  return tokenizer->encode(opts.prompt);
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

    const std::string tok_json = model_dir + "/tokenizer.json";
    const bool has_tokenizer = ds::util::file_exists(tok_json);

    const std::string idx_path = model_dir + "/model.safetensors.index.json";
    const bool has_index = ds::util::file_exists(idx_path);

    std::vector<std::string> expected_shards;
    std::vector<std::string> index_tensor_names;
    if (has_index) {
      const auto idx = ds::hf::load_safetensors_index(idx_path);
      expected_shards = idx.shard_filenames;
      index_tensor_names = idx.tensor_names;
    }

    if (cmd == "info") {
      print_config(cfg);
      std::cout << "Tokenizer: " << (has_tokenizer ? "OK" : "MISSING") << " (" << tok_json << ")\n";
      std::cout << "Index: " << (has_index ? "OK" : "(none)") << " (" << idx_path << ")\n";

      if (!expected_shards.empty()) {
        std::cout << "Index expects " << expected_shards.size() << " shard(s):\n";
        for (const auto& s : expected_shards) std::cout << "  " << s << "\n";
      }

      std::cout << "Found " << shards.size() << " shard(s) on disk.\n";
      for (const auto& s : shards) std::cout << "  " << s << "\n";
      return 0;
    }

    if (cmd == "load") {
      const auto m = ds::hf::load_model_dir(model_dir);
      std::cout << "Loaded model views: tensors=" << m.tensors.size() << " shards=" << m.shards.size() << "\n";

      auto it = m.tensors.find("model.embed_tokens.weight");
      if (it != m.tensors.end()) {
        std::cout << "Sample tensor: " << it->second.name << " bytes=" << it->second.nbytes << " shard=" << it->second.shard_path << "\n";
      }

      const auto registry = ds::rt::WeightRegistry::from_model(cfg, m);
      std::cout << "Layer registry:\n";
      for (std::size_t i = 0; i < registry.layers().size(); ++i) {
        std::cout << "  layer " << i << ": " << ds::rt::layer_kind_name(registry.layers()[i].kind) << "\n";
      }
      std::cout << "OK\n";
      return 0;
    }

    if (cmd == "run" || cmd == "generate") {
      const auto opts = parse_cmd_options(argc, argv, 3);
      ds::rt::Tokenizer tokenizer;
      const auto prompt_ids = resolve_prompt_ids(opts, tok_json, has_tokenizer, &tokenizer);
      if (prompt_ids.empty()) throw std::runtime_error("prompt resolved to zero token ids");

      auto model = ds::hf::load_model_dir(model_dir);
      if (has_tokenizer && tokenizer.empty()) tokenizer = ds::rt::Tokenizer::load_from_file(tok_json);
      ds::rt::ModelExecutor executor(
          cfg, std::move(model),
          ds::rt::RunConfig{
              .backend = ds::rt::parse_backend_kind(opts.backend),
              .max_seq = static_cast<std::size_t>(cfg.max_position_embeddings),
              .verbose = false,
          });

      if (cmd == "run") {
        const auto step = executor.prefill(prompt_ids);
        const auto next_id = step.greedy_token_id;
        std::cout << "Prompt tokens: " << prompt_ids.size() << "\n";
        std::cout << "Next token id: " << next_id << "\n";
        if (has_tokenizer) {
          if (tokenizer.empty()) tokenizer = ds::rt::Tokenizer::load_from_file(tok_json);
          std::cout << "Next token text: " << tokenizer.decode({next_id}) << "\n";
        }
        return 0;
      }

      ds::rt::GenerationConfig gen_cfg{
          .max_new_tokens = opts.max_new_tokens,
          .temperature = opts.temperature,
          .top_k = opts.top_k,
          .top_p = opts.top_p,
          .seed = opts.seed,
      };
      std::vector<std::string> pieces;
      const auto ids = executor.generate(prompt_ids, gen_cfg, has_tokenizer ? &tokenizer : nullptr, has_tokenizer ? &pieces : nullptr);
      std::cout << "Generated ids:";
      for (auto id : ids) std::cout << " " << id;
      std::cout << "\n";
      if (has_tokenizer) {
        std::cout << "Generated text: ";
        for (const auto& piece : pieces) std::cout << piece;
        std::cout << "\n";
      }
      return 0;
    }

    if (cmd == "verify" || cmd == "strict") {
      print_config(cfg);
      if (!has_tokenizer) {
        std::cout << "warn: tokenizer.json missing: " << tok_json << "\n";
      }

      // If we have an index, we can validate required tensor keys before shards are fully downloaded.
      std::vector<bool> layer_is_moe;
      layer_is_moe.resize(static_cast<std::size_t>(cfg.num_hidden_layers), false);

      if (!index_tensor_names.empty()) {
        std::unordered_set<std::string> keys;
        keys.reserve(index_tensor_names.size());
        for (const auto& k : index_tensor_names) keys.emplace(k);

        auto require_key = [&](const std::string& k) {
          if (keys.find(k) == keys.end()) {
            throw std::runtime_error("missing tensor in index: " + k);
          }
        };

        auto any_with_prefix = [&](const std::string& p) {
          for (const auto& k : index_tensor_names) {
            if (has_prefix(k, p)) return true;
          }
          return false;
        };

        // Global weights.
        require_key("lm_head.weight");
        require_key("model.embed_tokens.weight");
        require_key("model.norm.weight");

        // Per-layer weights. Checks are structural and match the DeepSeek-V2-Lite-Chat HF naming.
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
          const auto base = "model.layers." + std::to_string(i) + ".";

          require_key(base + "input_layernorm.weight");
          require_key(base + "post_attention_layernorm.weight");

          require_key(base + "self_attn.q_proj.weight");
          require_key(base + "self_attn.o_proj.weight");
          require_key(base + "self_attn.kv_a_layernorm.weight");
          require_key(base + "self_attn.kv_a_proj_with_mqa.weight");
          require_key(base + "self_attn.kv_b_proj.weight");

          const auto mlp = base + "mlp.";
          const bool is_moe = any_with_prefix(mlp + "experts.");
          layer_is_moe[static_cast<std::size_t>(i)] = is_moe;

          if (!is_moe) {
            require_key(mlp + "gate_proj.weight");
            require_key(mlp + "up_proj.weight");
            require_key(mlp + "down_proj.weight");
          } else {
            require_key(mlp + "gate.weight");
            require_key(mlp + "shared_experts.gate_proj.weight");
            require_key(mlp + "shared_experts.up_proj.weight");
            require_key(mlp + "shared_experts.down_proj.weight");

            // Check at least one expert; if config exposes n_experts, also check the last one.
            require_key(mlp + "experts.0.gate_proj.weight");
            require_key(mlp + "experts.0.up_proj.weight");
            require_key(mlp + "experts.0.down_proj.weight");

            if (cfg.n_experts > 1) {
              const auto last = std::to_string(cfg.n_experts - 1);
              require_key(mlp + "experts." + last + ".gate_proj.weight");
              require_key(mlp + "experts." + last + ".up_proj.weight");
              require_key(mlp + "experts." + last + ".down_proj.weight");
            }
          }
        }

        std::cout << "Index OK (required tensor keys present). keys=" << index_tensor_names.size() << "\n";
      }

      if (cmd == "strict" && !index_tensor_names.empty()) {
        std::cout << "Layer map (from index):\n";
        int moe = 0;
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
          const bool is_moe = layer_is_moe[static_cast<std::size_t>(i)];
          if (is_moe) ++moe;
          std::cout << "  layer " << i << ": " << (is_moe ? "MoE" : "dense") << "\n";
        }
        std::cout << "MoE layers: " << moe << "/" << cfg.num_hidden_layers << "\n";
      }

      if (shards.empty()) {
        if (!expected_shards.empty()) {
          std::cout << "Found 0 shard(s) on disk, but index expects " << expected_shards.size() << ":\n";
          for (const auto& s : expected_shards) {
            const auto p = model_dir + "/" + s;
            std::cout << "  MISSING " << p << "\n";
          }
          throw std::runtime_error("no *.safetensors shards present yet");
        }
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
