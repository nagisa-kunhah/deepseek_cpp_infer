#include "ds/runtime/model.h"

#include "ds/runtime/tokenizer.h"

#include <stdexcept>

namespace ds::rt {

std::string to_string(ModelFamily family) {
  switch (family) {
    case ModelFamily::DeepSeek: return "deepseek";
  }
  return "unknown";
}

bool ModelInfo::supports_backend(BackendKind backend) const {
  for (auto supported : supported_backends) {
    if (supported == backend) return true;
  }
  return false;
}

std::vector<std::int32_t> generate(const Model& model, ModelSession& session, const std::vector<std::int32_t>& prompt_ids,
                                   const GenerationConfig& gen_cfg, const Tokenizer* tokenizer,
                                   std::vector<std::string>* text_pieces) {
  if (gen_cfg.max_new_tokens <= 0) return {};

  ForwardOutput step = model.forward(session, ForwardInput{prompt_ids});

  ds::core::Sampler sampler(ds::core::SamplerConfig{
      .temperature = gen_cfg.temperature <= 0.0f ? 1.0f : gen_cfg.temperature,
      .top_k = gen_cfg.top_k,
      .top_p = gen_cfg.top_p,
      .seed = gen_cfg.seed,
  });

  std::vector<std::int32_t> generated;
  generated.reserve(static_cast<std::size_t>(gen_cfg.max_new_tokens));
  for (std::int32_t i = 0; i < gen_cfg.max_new_tokens; ++i) {
    std::int32_t next_id = step.greedy_token_id;
    if (gen_cfg.temperature > 0.0f || gen_cfg.top_k > 0 || gen_cfg.top_p > 0.0f) {
      next_id = static_cast<std::int32_t>(sampler.sample(step.logits));
    }

    generated.push_back(next_id);
    if (text_pieces != nullptr && tokenizer != nullptr) {
      text_pieces->push_back(tokenizer->decode({next_id}));
    }
    if (tokenizer != nullptr && tokenizer->metadata().eos_token_id >= 0 && next_id == tokenizer->metadata().eos_token_id) {
      break;
    }
    if (i + 1 < gen_cfg.max_new_tokens) {
      step = model.forward(session, ForwardInput{{next_id}});
    }
  }

  return generated;
}

} // namespace ds::rt
