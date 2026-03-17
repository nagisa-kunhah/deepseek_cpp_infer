# Remote Startup Guide

This guide is for operators who need to build and run `deepseek_cpp_infer` on a remote Linux machine.

## What you need to know

- The runtime tokenizer now uses `tokenizers-cpp` by default.
- `tokenizers-cpp` needs Rust only at build time.
- Running an already built `ds_chat` binary does not require a separate Rust installation.

## Recommended environment

- Ubuntu 22.04 or similar modern Linux distribution
- `git`
- `cmake`
- `python3`
- C++ build toolchain (`g++`, `make`, or `ninja`)
- Rust toolchain (`rustc`, `cargo`) for source builds
- CUDA toolkit only if you want the CUDA backend

## First-time setup on a fresh Ubuntu machine

Clone the repository and enter the project:

```bash
git clone --recursive <repo-url>
cd deepseek_cpp_infer/deepseek_cpp_infer
```

If the repository was cloned without submodules, initialize them:

```bash
git submodule update --init --recursive
```

Install build dependencies:

```bash
sudo tools/install_deps_ubuntu.sh
source "$HOME/.cargo/env"
```

## Fastest way to build and run

Use the helper script:

```bash
tools/bootstrap_run.sh /path/to/model verify
tools/bootstrap_run.sh /path/to/model run --prompt "hello"
tools/bootstrap_run.sh /path/to/model generate --prompt "hello" --max-new-tokens 8
```

What it does:

- initializes submodules if needed
- configures CMake into `./build` by default
- builds `ds_chat`
- runs the CLI command you pass in

## Running with a mock model

If you want to smoke-test the machine before downloading a real model:

```bash
tools/bootstrap_run.sh --mock verify
tools/bootstrap_run.sh --mock run --prompt "hello world"
```

## Running without rebuilding

If the machine already has a built binary, skip the build step:

```bash
tools/bootstrap_run.sh --skip-build /path/to/model verify
tools/bootstrap_run.sh --skip-build /path/to/model generate --prompt "hello" --max-new-tokens 8
```

If the binary is in a non-default build directory:

```bash
tools/bootstrap_run.sh --build-dir /path/to/build --skip-build /path/to/model verify
```

## Manual build commands

If you do not want to use the helper script:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
./build/ds_chat /path/to/model verify
./build/ds_chat /path/to/model run --prompt "hello"
```

## Common commands

Inspect model files:

```bash
./build/ds_chat /path/to/model info
./build/ds_chat /path/to/model verify
./build/ds_chat /path/to/model strict
./build/ds_chat /path/to/model load
```

Run or generate text:

```bash
./build/ds_chat /path/to/model run --prompt "hello"
./build/ds_chat /path/to/model generate --prompt "hello" --max-new-tokens 8
./build/ds_chat /path/to/model generate --prompt-ids 1,2,3 --max-new-tokens 8
```

Use CUDA explicitly when available:

```bash
./build/ds_chat /path/to/model run --backend cuda --prompt "hello"
./build/ds_chat /path/to/model generate --backend cuda --prompt "hello" --max-new-tokens 8
```

## Troubleshooting

### `rustc` or `cargo` not found

You are trying to build from source without the Rust toolchain.

Fix:

```bash
sudo tools/install_deps_ubuntu.sh
source "$HOME/.cargo/env"
```

### `vendor/tokenizers-cpp` missing

Submodules were not initialized.

Fix:

```bash
git submodule update --init --recursive
```

### CUDA build fails on a CPU-only machine

Disable CUDA during configure:

```bash
cmake -S . -B build -DDS_USE_CUDA=OFF
cmake --build build -j$(nproc)
```

### Tokenizer behavior differs from expectations

- The default path uses the external Hugging Face-compatible runtime.
- If you need a zero-ambiguity smoke path, prefer `--prompt-ids`.

## Quick handoff snippet

For a remote operator, the shortest successful sequence is usually:

```bash
git clone --recursive <repo-url>
cd deepseek_cpp_infer/deepseek_cpp_infer
sudo tools/install_deps_ubuntu.sh
source "$HOME/.cargo/env"
tools/bootstrap_run.sh /path/to/model verify
tools/bootstrap_run.sh /path/to/model generate --prompt "hello" --max-new-tokens 8
```
