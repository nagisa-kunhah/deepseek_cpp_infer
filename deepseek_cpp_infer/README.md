# deepseek_cpp_infer

Bootstrap C++17 inference framework skeleton for DeepSeek (target: DeepSeek-V2-Lite).

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## CLI

- `ds_chat <model_dir> info`
- `ds_chat <model_dir> verify` (default)

`<model_dir>` is expected to be a HuggingFace-style folder containing at least `config.json` and one or more `*.safetensors` shards.

## Current Status

- Loads `config.json`.
- Reads `model.safetensors.index.json` (expected shard list + tensor keys).
- Discovers `*.safetensors` shards.
- Parses safetensors headers without loading full tensor payloads.
- `verify` checks required DeepSeek-V2-Lite(-Chat) tensor key structure via index.
- `strict` prints dense/MoE layer map (from index).

Architecture: see `ARCHITECTURE.md`.

Next: weight mmap/lazy loading + tokenizer decode/encode + first end-to-end decode loop.
