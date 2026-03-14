#pragma once

#include "ds/core/sampler.h"
#include "ds/runtime/backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ds::rt {

class Tokenizer;

enum class ModelFamily {
  DeepSeek,
};

std::string to_string(ModelFamily family);

struct ModelInfo {
  ModelFamily family = ModelFamily::DeepSeek;
  std::string model_type;
  std::int32_t vocab_size = 0;
  std::int32_t max_position_embeddings = 0;
  std::vector<BackendKind> supported_backends;

  bool supports_backend(BackendKind backend) const;
};

struct ForwardInput {
  std::vector<std::int32_t> token_ids;
};

struct ForwardOutput {
  std::vector<float> logits;
  std::int32_t greedy_token_id = 0;
  std::size_t position_after = 0;
};

class ModelSession {
 public:
  virtual ~ModelSession() = default;

  virtual void reset() = 0;
  virtual std::size_t position() const = 0;
};

class Model {
 public:
  virtual ~Model() = default;

  virtual const ModelInfo& info() const = 0;
  virtual std::unique_ptr<ModelSession> create_session(const RunConfig& run_cfg) const = 0;
  virtual ForwardOutput forward(ModelSession& session, const ForwardInput& input) const = 0;
};

std::vector<std::int32_t> generate(const Model& model, ModelSession& session, const std::vector<std::int32_t>& prompt_ids,
                                   const GenerationConfig& gen_cfg, const Tokenizer* tokenizer = nullptr,
                                   std::vector<std::string>* text_pieces = nullptr);

} // namespace ds::rt
