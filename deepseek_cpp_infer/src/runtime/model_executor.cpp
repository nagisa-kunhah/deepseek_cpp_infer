#include "ds/runtime/model_executor.h"

#include "ds/runtime/model.h"

#include <stdexcept>

namespace ds::rt {

namespace {

StepResult to_step_result(const ForwardOutput& out) {
  StepResult step;
  step.logits = out.logits;
  step.greedy_token_id = out.greedy_token_id;
  return step;
}

const DeepSeekSession& require_deepseek_session(const std::unique_ptr<ModelSession>& session) {
  auto* deepseek_session = dynamic_cast<const DeepSeekSession*>(session.get());
  if (deepseek_session == nullptr) throw std::runtime_error("executor: session is not DeepSeekSession");
  return *deepseek_session;
}

} // namespace

ModelExecutor::ModelExecutor(const ds::hf::DeepSeekConfig& cfg, ds::hf::LoadedModel model, RunConfig run_cfg) {
  model_ = std::make_unique<DeepSeekModel>(cfg, std::move(model));
  session_ = model_->create_session(run_cfg);
}

ModelExecutor::~ModelExecutor() = default;

ModelExecutor::ModelExecutor(ModelExecutor&&) noexcept = default;

ModelExecutor& ModelExecutor::operator=(ModelExecutor&&) noexcept = default;

void ModelExecutor::reset() {
  session_->reset();
  pos_ = 0;
}

const WeightRegistry& ModelExecutor::registry() const { return model_->registry(); }

const cuda::CudaStats& ModelExecutor::cuda_stats() const { return require_deepseek_session(session_).cuda_stats(); }

StepResult ModelExecutor::decode_next(std::int32_t token_id) {
  const auto out = model_->forward(*session_, ForwardInput{{token_id}});
  pos_ = out.position_after;
  return to_step_result(out);
}

StepResult ModelExecutor::prefill(const std::vector<std::int32_t>& prompt_ids) {
  const auto out = model_->forward(*session_, ForwardInput{prompt_ids});
  pos_ = out.position_after;
  return to_step_result(out);
}

std::vector<std::int32_t> ModelExecutor::generate(const std::vector<std::int32_t>& prompt_ids, const GenerationConfig& gen_cfg,
                                                  const Tokenizer* tokenizer, std::vector<std::string>* text_pieces) {
  const auto ids = ds::rt::generate(*model_, *session_, prompt_ids, gen_cfg, tokenizer, text_pieces);
  pos_ = session_->position();
  return ids;
}

} // namespace ds::rt
