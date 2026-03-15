#pragma once

#include "ds/runtime/model.h"
#include "ds/runtime/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ds::rt {

struct StepResult {
  std::vector<float> logits;
  std::int32_t greedy_token_id = 0;
};

class ModelExecutor {
 public:
  ModelExecutor(std::shared_ptr<const Model> model, RunConfig run_cfg);
  ~ModelExecutor();
  ModelExecutor(const ModelExecutor&) = delete;
  ModelExecutor& operator=(const ModelExecutor&) = delete;
  ModelExecutor(ModelExecutor&&) noexcept;
  ModelExecutor& operator=(ModelExecutor&&) noexcept;

  void reset();
  std::size_t position() const { return pos_; }
  const ModelInfo& info() const { return model_->info(); }

  StepResult prefill(const std::vector<std::int32_t>& prompt_ids);
  StepResult decode_next(std::int32_t token_id);
  std::vector<std::int32_t> generate(const std::vector<std::int32_t>& prompt_ids, const GenerationConfig& gen_cfg,
                                     const Tokenizer* tokenizer = nullptr, std::vector<std::string>* text_pieces = nullptr);

 private:
  std::shared_ptr<const Model> model_;
  std::unique_ptr<ModelSession> session_;
  std::size_t pos_ = 0;
};

} // namespace ds::rt
