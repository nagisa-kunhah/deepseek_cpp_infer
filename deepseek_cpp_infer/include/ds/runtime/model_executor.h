#pragma once

#include "ds/core/sampler.h"
#include "ds/hf/config.h"
#include "ds/hf/model_loader.h"
#include "ds/runtime/backend.h"
#include "ds/runtime/deepseek_model.h"
#include "ds/runtime/model.h"
#include "ds/runtime/tokenizer.h"
#include "ds/runtime/weights.h"

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
  ModelExecutor(const ds::hf::DeepSeekConfig& cfg, ds::hf::LoadedModel model, RunConfig run_cfg);
  ~ModelExecutor();
  ModelExecutor(const ModelExecutor&) = delete;
  ModelExecutor& operator=(const ModelExecutor&) = delete;
  ModelExecutor(ModelExecutor&&) noexcept;
  ModelExecutor& operator=(ModelExecutor&&) noexcept;

  void reset();
  std::size_t position() const { return pos_; }

  const WeightRegistry& registry() const;
  const cuda::CudaStats& cuda_stats() const;

  StepResult prefill(const std::vector<std::int32_t>& prompt_ids);
  StepResult decode_next(std::int32_t token_id);
  std::vector<std::int32_t> generate(const std::vector<std::int32_t>& prompt_ids, const GenerationConfig& gen_cfg,
                                     const Tokenizer* tokenizer = nullptr, std::vector<std::string>* text_pieces = nullptr);

 private:
  std::unique_ptr<DeepSeekModel> model_;
  std::unique_ptr<ModelSession> session_;
  std::size_t pos_ = 0;
};

} // namespace ds::rt
