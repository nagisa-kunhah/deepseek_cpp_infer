#include "ds/core/sampler.h"

#include "ds/core/math.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ds::core {
namespace {

struct Item {
  int id;
  float logit;
  float p;
};

} // namespace

Sampler::Sampler(SamplerConfig cfg) : cfg_(cfg), rng_(cfg.seed ? cfg.seed : std::random_device{}()) {}

int Sampler::sample(const std::vector<float>& logits_in) {
  if (logits_in.empty()) throw std::runtime_error("sampler: empty logits");

  std::vector<Item> items;
  items.reserve(logits_in.size());

  const float t = (cfg_.temperature <= 0.0f) ? 1.0f : cfg_.temperature;
  for (int i = 0; i < static_cast<int>(logits_in.size()); ++i) {
    items.push_back(Item{.id = i, .logit = logits_in[static_cast<std::size_t>(i)] / t, .p = 0.0f});
  }

  // Top-k filter.
  if (cfg_.top_k > 0 && cfg_.top_k < static_cast<int>(items.size())) {
    std::nth_element(items.begin(), items.begin() + cfg_.top_k, items.end(),
                     [](const Item& a, const Item& b) { return a.logit > b.logit; });
    items.resize(static_cast<std::size_t>(cfg_.top_k));
  }

  // Softmax over remaining.
  std::vector<float> tmp;
  tmp.reserve(items.size());
  for (const auto& it : items) tmp.push_back(it.logit);
  softmax_f32(tmp.data(), tmp.size());
  for (std::size_t i = 0; i < items.size(); ++i) items[i].p = tmp[i];

  // Top-p (nucleus) filter.
  if (cfg_.top_p > 0.0f && cfg_.top_p < 1.0f) {
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.p > b.p; });
    float cum = 0.0f;
    std::size_t keep = 0;
    for (; keep < items.size(); ++keep) {
      cum += items[keep].p;
      if (cum >= cfg_.top_p) {
        ++keep;
        break;
      }
    }
    keep = std::max<std::size_t>(1, std::min<std::size_t>(keep, items.size()));
    items.resize(keep);

    // Renormalize.
    float s = 0.0f;
    for (const auto& it : items) s += it.p;
    for (auto& it : items) it.p /= s;
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng_);
  float cum = 0.0f;
  for (const auto& it : items) {
    cum += it.p;
    if (r <= cum) return it.id;
  }
  return items.back().id;
}

} // namespace ds::core
