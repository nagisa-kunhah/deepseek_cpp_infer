# deepseek_cpp_infer

Bootstrap C++17 inference framework for DeepSeek (target: DeepSeek-V2-Lite / Lite-Chat).

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

This project now depends on the Rust toolchain to build the external tokenizer
runtime (`vendor/tokenizers-cpp`). Install `rustup`, `rustc`, and `cargo`
before configuring CMake.

Rust is only required at build time. Once `ds_chat` is built, the runtime does
not need a separate Rust installation on the target machine.

For a fresh Ubuntu remote machine, you can install the toolchain with:

```bash
sudo tools/install_deps_ubuntu.sh
source "$HOME/.cargo/env"
```

Then use the helper below to initialize submodules, build the project, and run
the CLI in one step:

```bash
tools/bootstrap_run.sh /path/to/model verify
tools/bootstrap_run.sh /path/to/model run --prompt "hello"
tools/bootstrap_run.sh --mock generate --prompt "hello world" --max-new-tokens 3
```

If the machine already has a built `ds_chat`, you can skip the build step:

```bash
tools/bootstrap_run.sh --skip-build /path/to/model generate --prompt "hello" --max-new-tokens 8
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
- Uses `tokenizers-cpp` for the default Hugging Face `tokenizer.json` runtime and
  keeps a minimal tokenizer implementation only for tests and smoke fixtures.

## Runtime Notes

- CPU path is correctness-first and streams BF16/F16/F32 weights row-by-row.
- CUDA phase-1 path now exists for mock/small-model execution:
  - uses the installed CUDA toolkit (`NVRTC + CUDA Driver API + cuBLAS`)
  - keeps activations and per-layer MLA KV cache resident on device during decode
  - offloads `RMSNorm`, linear/GEMV, RoPE, MLA score/softmax/value accumulation, and token-local MoE routing
  - uses cached full-weight uploads for small/medium tensors and row-chunk streaming GEMV for large tensors
  - exposes internal CUDA hit/fallback stats for regression tests
- `--prompt` now uses an external tokenizer runtime via `tokenizers-cpp`, which
  is substantially closer to real Hugging Face behavior than the old minimal
  in-tree implementation.
- `--prompt-ids` remains the zero-ambiguity fallback when you want to bypass
  tokenizer behavior entirely.
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
- `tests/cuda_runtime_smoke.cpp` for CPU/CUDA executor parity and CUDA hit coverage
- `tests/run_e2e_cli.py` for the full `info/verify/strict/load/run/generate` CLI chain

If a CUDA toolkit is available in the user environment, `ctest` also runs the
same mock flow with `--backend cuda`.

Architecture and implementation notes:

- `ARCHITECTURE.md`
- `docs/remote_startup.md`
- `docs/deepseek_v2_lite_architecture.md`
- `docs/runtime_design.md`
- `docs/implementation_plan.md`
- `docs/gpu_phase1_retrospective.md`
