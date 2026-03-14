# TODO

## Milestone 0 (bootstrap)

- [x] Parse `config.json` (HF-style)
- [x] Model-dir discovery + presence checks (`config.json`, `tokenizer.json`, index, shards)
- [x] Parse safetensors headers without loading full weights (header-only)
- [x] `verify`: validate required DeepSeek-V2-Lite tensor key structure via `model.safetensors.index.json`
- [x] `strict`: print dense/MoE layer map inferred from the index
- [x] `verify`: when shards exist, parse each shard header and print tensor counts
- [x] `load`: mmap shards and build tensor views (end-to-end load success depends on shards downloaded)

## Milestone 1 (runnable)

- [x] Tokenizer loader (lightweight `tokenizer.json` path; full HF parity still pending)
- [x] Runtime tensor abstraction (CPU activations as f32) + fp16/bf16 decode helpers
- [x] Streaming weight access helpers (embedding lookup, lm_head greedy scan)
- [x] Semantic `WeightRegistry` for global/layer weights
- [ ] Implement core ops needed by DeepSeek-V2-Lite block (RMSNorm, RoPE, attention, MLP/MoE routing)
  - [x] CPU baseline: linear + RMSNorm (f32 activations)
  - [x] Decode weights (BF16/F16/F32) to f32 (bootstrap path)
  - [x] CPU baseline: RoPE + naive attention (single-head + multi-head) + KV cache structs
- [x] CPU reference: MLA-like per-layer cache + single-token attention decode path
- [x] CPU reference: dense MLP + token-local MoE routing
- [x] KV cache + incremental decode
- [ ] Sampling (temperature/top-k/top-p, repetition penalty)
  - [x] temperature/top-k/top-p (bootstrap sampler)
- [x] `run` / `generate` CLI wiring
- [x] Mock Hugging Face model generator + end-to-end CLI smoke test

## Milestone 2 (performance)

- [x] Mmap weights (read-only) / lazy view layer (no tensor materialization)
- [ ] Threading, batching
- [ ] CUDA kernels / cuBLAS integration
- [ ] Blocked decode for large expert matrices
- [ ] Full tokenizer parity with Hugging Face tokenizers
