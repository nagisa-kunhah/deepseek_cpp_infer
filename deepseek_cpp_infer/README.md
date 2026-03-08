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
- Discovers `*.safetensors` shards.
- Parses safetensors headers without loading full tensor payloads.
- Verifies basic invariants (e.g. at least one shard; header parse succeeds).

Next: tokenizer loading + minimal sampling loop + real weight mapping.
