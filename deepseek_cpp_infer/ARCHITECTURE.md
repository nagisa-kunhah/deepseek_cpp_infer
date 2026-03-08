# Architecture (Draft)

Goal: a runnable DeepSeek-V2-Lite(-Chat) inference framework in C++ (CPU-first, CUDA optional).

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

- HF Parsers/Adapters (`include/ds/hf/*`, `src/hf/*`)
  - `config`: reads `config.json` into `DeepSeekConfig`.
  - `weights_index`: reads `model.safetensors.index.json` to get shard filenames + tensor keys.
  - `safetensors`: header-only parsing (safe for multi-GB shards); full-file loader kept for debugging.

- Runtime/Execution (planned)
  - `Tensor`: dtype/shape/strides + backing storage.
  - `Device`: CPU allocator first; CUDA allocator/kernels later.
  - `Ops`: RMSNorm, RoPE, Attention (with KV cache), MoE routing + expert MLP.
  - `Sampler`: temperature/top-k/top-p/repetition penalty.

## Data Flow (planned for `run`)

```text
prompt -> tokenizer -> token ids -> prefill forward -> logits -> sampler -> next token
                                              |                         |
                                              +---- updates KV cache ----+
```

## Notes

- Model key naming is currently aligned to HF repo `deepseek-ai/DeepSeek-V2-Lite-Chat`.
- Default behavior should avoid reading whole safetensors shards into RAM.
