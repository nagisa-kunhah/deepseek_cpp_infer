#include "ds/runtime/ops.h"

#include "ds/core/math.h"
#include "ds/hf/decode.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ds::rt {
namespace {

float silu(float x) { return x / (1.0f + std::exp(-x)); }

std::vector<float> matvec(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, std::size_t out) {
  if (weight.shape.size() != 2) throw std::runtime_error("linear: expected a 2D weight");
  if (static_cast<std::size_t>(weight.shape[0]) != out || static_cast<std::size_t>(weight.shape[1]) != in) {
    throw std::runtime_error("linear: shape mismatch");
  }

  std::vector<float> y(out, 0.0f);
  linear(weight, x, in, y.data(), out);
  return y;
}

void add_bias(const ds::hf::TensorSlice* bias, float* y, std::size_t n) {
  if (bias == nullptr) return;
  if (bias->shape.size() != 1 || static_cast<std::size_t>(bias->shape[0]) != n) {
    throw std::runtime_error("linear: bias shape mismatch");
  }
  for (std::size_t i = 0; i < n; ++i) y[i] += ds::hf::read_scalar_f32(*bias, i);
}

void select_topk(const std::vector<float>& scores, std::size_t top_k, std::vector<std::size_t>* ids, std::vector<float>* probs) {
  std::vector<std::size_t> order(scores.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                    [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
  ids->assign(order.begin(), order.begin() + top_k);
  probs->resize(top_k);
  for (std::size_t i = 0; i < top_k; ++i) (*probs)[i] = scores[(*ids)[i]];
}

} // namespace

void linear(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y, std::size_t out) {
  if (weight.shape.size() != 2) throw std::runtime_error("linear: expected a 2D weight");
  if (static_cast<std::size_t>(weight.shape[0]) != out || static_cast<std::size_t>(weight.shape[1]) != in) {
    throw std::runtime_error("linear: weight shape mismatch");
  }

  for (std::size_t row = 0; row < out; ++row) {
    float acc = 0.0f;
    const std::size_t base = row * in;
    for (std::size_t col = 0; col < in; ++col) {
      acc += x[col] * ds::hf::read_scalar_f32(weight, base + col);
    }
    y[row] = acc;
  }
}

void linear(const LinearWeights& linear_weight, const float* x, std::size_t in, float* y, std::size_t out) {
  if (!linear_weight.valid()) throw std::runtime_error("linear: missing weight");
  linear(*linear_weight.weight, x, in, y, out);
  add_bias(linear_weight.bias, y, out);
}

void dense_mlp_step_cpu(const DenseMLPWeights& mlp, const float* x, std::size_t hidden_size, float* out) {
  if (!mlp.valid()) throw std::runtime_error("dense mlp: missing weights");

  const auto intermediate = static_cast<std::size_t>(mlp.gate_proj.weight->shape[0]);
  std::vector<float> gate = matvec(*mlp.gate_proj.weight, x, hidden_size, intermediate);
  std::vector<float> up = matvec(*mlp.up_proj.weight, x, hidden_size, intermediate);
  for (std::size_t i = 0; i < intermediate; ++i) gate[i] = silu(gate[i]) * up[i];

  linear(*mlp.down_proj.weight, gate.data(), intermediate, out, hidden_size);
}

void moe_step_cpu(const ds::hf::DeepSeekConfig& cfg, const MoEWeights& moe, const float* x, std::size_t hidden_size,
                  float* out) {
  if (!moe.valid()) throw std::runtime_error("moe: missing routed experts");
  std::fill(out, out + hidden_size, 0.0f);

  const auto n_experts = static_cast<std::size_t>(moe.experts.size());
  std::vector<float> scores(n_experts, 0.0f);
  linear(*moe.gate.weight, x, hidden_size, scores.data(), n_experts);
  ds::core::softmax_f32(scores.data(), scores.size());

  const std::size_t top_k = std::min<std::size_t>(std::max<std::int32_t>(1, cfg.moe_top_k), n_experts);
  std::vector<std::size_t> top_ids;
  std::vector<float> top_probs;
  select_topk(scores, top_k, &top_ids, &top_probs);

  if (cfg.norm_topk_prob) {
    float sum = 0.0f;
    for (float p : top_probs) sum += p;
    if (sum > 0.0f) {
      for (auto& p : top_probs) p /= sum;
    }
  }

  std::vector<float> tmp(hidden_size, 0.0f);
  for (std::size_t i = 0; i < top_ids.size(); ++i) {
    std::fill(tmp.begin(), tmp.end(), 0.0f);
    dense_mlp_step_cpu(moe.experts[top_ids[i]].ffn, x, hidden_size, tmp.data());
    const float scale = top_probs[i] * cfg.routed_scaling_factor;
    for (std::size_t j = 0; j < hidden_size; ++j) out[j] += scale * tmp[j];
  }

  if (moe.shared_experts.valid()) {
    std::fill(tmp.begin(), tmp.end(), 0.0f);
    dense_mlp_step_cpu(moe.shared_experts, x, hidden_size, tmp.data());
    for (std::size_t j = 0; j < hidden_size; ++j) out[j] += tmp[j];
  }
}

void lm_head_logits_cpu(const ds::hf::TensorSlice& lm_head_weight, const float* hidden, std::vector<float>* logits) {
  if (lm_head_weight.shape.size() != 2) throw std::runtime_error("lm_head: expected 2D weight");
  const auto vocab = static_cast<std::size_t>(lm_head_weight.shape[0]);
  const auto hidden_size = static_cast<std::size_t>(lm_head_weight.shape[1]);
  logits->assign(vocab, 0.0f);
  linear(lm_head_weight, hidden, hidden_size, logits->data(), vocab);
}

} // namespace ds::rt
