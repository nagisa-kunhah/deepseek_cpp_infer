# Progress

Date: 2026-03-14

## Current State

The repository is no longer just a model-directory verifier. It now contains a
documented runtime skeleton, a semantic weight registry, a lightweight
`tokenizer.json` loader, a single-session decode executor, and both CPU and
bootstrap CUDA execution paths.

## What Works

- `info`, `verify`, `strict`, and `load` work on Hugging Face style model
  directories.
- `run` and `generate` work end-to-end on the repository's mock DeepSeek-V2-Lite
  model.
- CPU path works on the mock model.
- `--backend cuda` also works on the mock model through a bootstrap CUDA path.
- `ctest --test-dir build --output-on-failure` passes with:
  - `ds_runtime_smoke`
  - `ds_cli_e2e`
  - `ds_cli_cuda_e2e`

## Real Model Findings

Real `DeepSeek-V2-Lite-Chat` metadata and shards were validated against this
runtime:

- real `config.json` parsing succeeds
- real `model.safetensors.index.json` parsing succeeds
- real shard header verification succeeds
- real `load` succeeds with `5291` tensors across `4` shards

CPU inference with the real model is not currently viable on this machine:

- single-token `run --prompt-ids 0` entered the real forward path
- process RSS climbed to about `18.3 GiB`
- the kernel OOM killer terminated `ds_chat`

Conclusion:

- real-model `verify/load` work
- real-model CPU `run/generate` do not currently fit this machine

## CUDA Status

The CUDA backend is a bootstrap implementation, not a full high-performance
backend yet.

- Uses `NVRTC + CUDA Driver API`
- Runtime-compiles kernels on first use
- Offloads:
  - `RMSNorm`
  - `matvec` / linear layers
- Leaves the following in the current C++ control flow:
  - attention control logic
  - RoPE orchestration
  - KV cache updates
  - MoE routing decisions

Large weights can still fall back to the CPU path in this bootstrap backend.

## Main Limitations

- real BF16 DeepSeek-V2-Lite weights do not fit for practical inference on the
  current CPU-only path
- the `RTX 4060 Laptop 8GB` cannot hold the raw full model in VRAM
- tokenizer implementation is still lightweight and not fully Hugging Face
  compatible
- CUDA path does not yet cover full attention/MoE math or cuBLAS hot paths

## Next Good Steps

- move more attention/MoE math to CUDA
- replace bootstrap CUDA matvec kernels with cuBLAS/cuBLASLt
- add a quantized or reduced-memory route for running Lite-class models on 8 GB
  GPUs
