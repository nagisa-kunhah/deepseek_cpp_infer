#include "ds/runtime/model_executor.h"

#include "ds/runtime/model.h"

namespace ds::rt {

namespace {

StepResult to_step_result(const ForwardOutput& out) {
  StepResult step;
  step.logits = out.logits;
  step.greedy_token_id = out.greedy_token_id;
  return step;
}

} // namespace

ModelExecutor::ModelExecutor(std::shared_ptr<const Model> model, RunConfig run_cfg) : model_(std::move(model)) {
  session_ = model_->create_session(run_cfg);
}

ModelExecutor::~ModelExecutor() = default;

ModelExecutor::ModelExecutor(ModelExecutor&&) noexcept = default;

ModelExecutor& ModelExecutor::operator=(ModelExecutor&&) noexcept = default;

void ModelExecutor::reset() {
  session_->reset();
  pos_ = 0;
}

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
