# TODO

## Milestone 0 (bootstrap)

- [x] Parse `config.json` (HF-style)
- [ ] Add model-dir discovery utilities (config/tokenizer/shards)
- [ ] Parse safetensors headers without loading full weights
- [ ] `verify` command: print shard list, tensor counts, basic sanity checks

## Milestone 1 (runnable)

- [ ] Tokenizer loader (prefer `tokenizer.json`; fallback to BPE vocab files)
- [ ] Runtime tensor storage abstraction (CPU first; CUDA optional)
- [ ] Implement core ops needed by DeepSeek-V2-Lite block (RMSNorm, RoPE, attention, MLP/MoE routing)
- [ ] KV cache + incremental decode
- [ ] Sampling (temperature/top-k/top-p, repetition penalty)

## Milestone 2 (performance)

- [ ] Mmap weights / lazy load
- [ ] Threading, batching
- [ ] CUDA kernels (optional)
