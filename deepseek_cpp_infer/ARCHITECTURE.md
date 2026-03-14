# Architecture (Draft)

Goal: a runnable DeepSeek-V2-Lite(-Chat) inference framework in C++ (CPU-first, CUDA later).

## Big Picture

```text
                +---------------------------+
                |         ds_chat           |
                |  CLI (info/verify/run)    |
                +-------------+-------------+
                              |
                              v
+-------------------+   +-----+-------------------+   +-----------------------+
|  Model Directory  |   |   HF Parsers/Adapters   |   |   Runtime / Execution  |
| (HF-style folder) |-->| config.json parser      |-->| Tensor storage         |
| - config.json     |   | tokenizer metadata      |   | KV cache               |
| - tokenizer.json  |   | safetensors index       |   | Operators (CPU/CUDA)   |
| - *.safetensors   |   | safetensors header      |   | Sampler                |
+-------------------+   +-------------------------+   +-----------+-----------+
                                                                  |
                                                                  v
                                                       +----------+----------+
                                                       |   DeepSeek Model    |
                                                       |  (layers, MoE, ...) |
                                                       +---------------------+
```

## Components

- CLI
  - `info`: show config, tokenizer presence, index + expected shard filenames.
  - `verify`: validate required tensor keys via index; parse safetensors headers if shards exist.
  - `strict`: like `verify`, and prints a dense/MoE layer map inferred from the index.
  - `load`: mmap shards and build tensor views, then resolve semantic layer types.
  - `run`: execute prompt prefill and print the next token prediction.
  - `generate`: autoregressive decode loop for a single session.

- HF Parsers/Adapters (`include/ds/hf/*`, `src/hf/*`)
  - `config`: reads `config.json` into `DeepSeekConfig`.
  - `weights_index`: reads `model.safetensors.index.json` to get shard filenames + tensor keys.
  - `safetensors`: header-only parsing (safe for multi-GB shards); full-file loader kept for debugging.
  - `model_loader`: mmap shards and resolves non-owning tensor slices.

- Runtime/Execution
  - `WeightRegistry`: semantic per-layer views over HF tensor names.
  - `Tokenizer`: lightweight `tokenizer.json` loader with greedy text encode fallback.
  - `ModelExecutor`: single-session incremental decode.
  - `Ops`: CPU RMSNorm, RoPE, attention decode, dense MLP, MoE routing, lm_head logits.
  - `BackendKind`: public backend selection (`cpu`, `cuda`).
  - `Sampler`: temperature/top-k/top-p sampling.

## Data Flow (planned for `run`)

```text
prompt -> tokenizer -> token ids -> prefill forward -> logits -> sampler -> next token
                                              |                         |
                                              +---- updates KV cache ----+
```

## Notes

- Model key naming is currently aligned to HF repo `deepseek-ai/DeepSeek-V2-Lite-Chat`.
- Default behavior should avoid reading whole safetensors shards into RAM.
- Attention/MoE implementation details now live in `docs/deepseek_v2_lite_architecture.md`.
- Runtime interfaces and milestones now live in `docs/runtime_design.md` and `docs/implementation_plan.md`.
