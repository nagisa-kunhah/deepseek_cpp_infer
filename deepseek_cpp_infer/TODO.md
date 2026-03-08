# TODO

## Milestone 0 (bootstrap)

- [x] Parse `config.json` (HF-style)
- [x] Model-dir discovery + presence checks (`config.json`, `tokenizer.json`, index, shards)
- [x] Parse safetensors headers without loading full weights (header-only)
- [x] `verify`: validate required DeepSeek-V2-Lite tensor key structure via `model.safetensors.index.json`
- [x] `strict`: print dense/MoE layer map inferred from the index
- [ ] `verify`: when shards exist, parse each shard header and print tensor counts
- [ ] `load`: mmap shards and build tensor views (end-to-end load success depends on shards downloaded)

## Milestone 1 (runnable)

- [ ] Tokenizer loader (prefer `tokenizer.json`; fallback to BPE vocab files)
- [ ] Runtime tensor storage abstraction (CPU first; CUDA optional)
- [ ] Implement core ops needed by DeepSeek-V2-Lite block (RMSNorm, RoPE, attention, MLP/MoE routing)
- [ ] KV cache + incremental decode
- [ ] Sampling (temperature/top-k/top-p, repetition penalty)

## Milestone 2 (performance)

- [x] Mmap weights (read-only) / lazy view layer (no tensor materialization)
- [ ] Threading, batching
- [ ] CUDA kernels (optional)
