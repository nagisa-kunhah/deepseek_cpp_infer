# deepseek_cpp_infer

Bootstrap C++17 inference framework for DeepSeek (target: DeepSeek-V2-Lite / Lite-Chat).

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## CLI

- `ds_chat <model_dir> info`
- `ds_chat <model_dir> verify` (default)
- `ds_chat <model_dir> strict`
- `ds_chat <model_dir> load`
- `ds_chat <model_dir> run --prompt-ids 1,2,3`
- `ds_chat <model_dir> generate --prompt-ids 1,2,3 --max-new-tokens 8`

`<model_dir>` is expected to be a HuggingFace-style folder containing at least `config.json` and one or more `*.safetensors` shards.

## Current Status

- Loads `config.json`.
- Reads `model.safetensors.index.json` (expected shard list + tensor keys).
- Discovers `*.safetensors` shards.
- Parses safetensors headers without loading full tensor payloads.
- Builds mmap-backed tensor views through `load`.
- Maps Hugging Face tensor names into a semantic `WeightRegistry`.
- Provides a CPU reference executor for single-session incremental decode.
- Supports DeepSeek-V2-Lite style attention cache layout and token-local MoE routing.
- Exposes `run` and `generate` commands.
- Includes a lightweight tokenizer loader and smoke-test coverage for runtime wiring.

## Runtime Notes

- CPU path is correctness-first and streams BF16/F16/F32 weights row-by-row.
- CUDA is part of the public API surface (`--backend cuda`) but is still a stub.
- `--prompt` works through a minimal tokenizer implementation. For exact/reliable smoke
  runs, prefer `--prompt-ids`.
- Current RoPE path uses the base theta path and does not yet implement the full
  DeepSeek-V2 scaling variants.

## End-to-End Smoke Flow

You can generate a tiny Hugging Face style mock model and run the whole CLI flow:

```bash
python3 tools/make_mock_deepseek_model.py /tmp/ds_mock_model
./build/ds_chat /tmp/ds_mock_model verify
./build/ds_chat /tmp/ds_mock_model load
./build/ds_chat /tmp/ds_mock_model run --prompt "hello world!"
./build/ds_chat /tmp/ds_mock_model generate --prompt "hello world!" --max-new-tokens 3
ctest --test-dir build --output-on-failure
```

The repository includes:

- `tests/runtime_smoke.cpp` for runtime unit smoke coverage
- `tests/run_e2e_cli.py` for the full `info/verify/strict/load/run/generate` CLI chain

Architecture and implementation notes:

- `ARCHITECTURE.md`
- `docs/deepseek_v2_lite_architecture.md`
- `docs/runtime_design.md`
- `docs/implementation_plan.md`
