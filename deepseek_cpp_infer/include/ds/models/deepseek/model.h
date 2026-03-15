#pragma once

#include "ds/hf/model_loader.h"
#include "ds/models/deepseek/config.h"
#include "ds/models/deepseek/cuda_backend.h"
#include "ds/models/deepseek/mla.h"
#include "ds/models/deepseek/weights.h"
#include "ds/runtime/model.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace ds::rt {

struct CudaStateDeleter {
  void operator()(cuda::CudaExecutorState* state) const;
};

class DeepSeekSession : public ModelSession {
 public:
  DeepSeekSession(const ds::hf::DeepSeekConfig& cfg, RunConfig run_cfg);
  ~DeepSeekSession() override;
  DeepSeekSession(const DeepSeekSession&) = delete;
  DeepSeekSession& operator=(const DeepSeekSession&) = delete;
  DeepSeekSession(DeepSeekSession&&) noexcept;
  DeepSeekSession& operator=(DeepSeekSession&&) noexcept;

  void reset() override;
  std::size_t position() const override { return pos_; }

  const RunConfig& run_config() const { return run_cfg_; }
  const cuda::CudaStats& cuda_stats() const;

 private:
  friend class DeepSeekModel;

  ds::hf::DeepSeekConfig cfg_;
  RunConfig run_cfg_;
  std::vector<MLACache> caches_;
  std::unique_ptr<cuda::CudaExecutorState, CudaStateDeleter> cuda_state_;
  std::size_t pos_ = 0;
};

class DeepSeekModel : public Model {
 public:
  DeepSeekModel(const ds::hf::DeepSeekConfig& cfg, ds::hf::LoadedModel model);

  const ModelInfo& info() const override { return info_; }
  std::unique_ptr<ModelSession> create_session(const RunConfig& run_cfg) const override;
  ForwardOutput forward(ModelSession& session, const ForwardInput& input) const override;

  const ds::hf::DeepSeekConfig& config() const { return cfg_; }
  const DeepSeekWeightRegistry& registry() const { return registry_; }

 private:
  ForwardOutput forward_impl(DeepSeekSession& session, const ForwardInput& input) const;
  ForwardOutput decode_next(DeepSeekSession& session, std::int32_t token_id) const;

  ds::hf::DeepSeekConfig cfg_;
  ds::hf::LoadedModel model_;
  DeepSeekWeightRegistry registry_;
  ModelInfo info_;
};

} // namespace ds::rt
