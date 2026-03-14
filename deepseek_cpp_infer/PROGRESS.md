# Progress

Date: 2026-03-14

## Current State

The repository is no longer just a model-directory verifier. It now contains a
documented runtime skeleton, a semantic weight registry, a lightweight
`tokenizer.json` loader, a single-session decode executor, and both CPU and
phase-1 CUDA execution paths.

## What Works

- `info`, `verify`, `strict`, and `load` work on Hugging Face style model
  directories.
- `run` and `generate` work end-to-end on the repository's mock DeepSeek-V2-Lite
  model.
- CPU path works on the mock model.
- `--backend cuda` works on the mock model through a device-resident CUDA path.
- `ctest --test-dir build --output-on-failure` passes with:
  - `ds_runtime_smoke`
  - `ds_cli_e2e`
  - `ds_cuda_runtime_smoke`
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
- real-model CUDA `run --prompt-ids 0` now enters the CUDA forward path without
  immediately OOMing, but did not finish within a 90 second smoke timeout on
  this machine

## CUDA Status

The CUDA backend now covers much more of the decode path, but it is still not a
fully optimized inference backend yet.

- Uses `NVRTC + CUDA Driver API`
- Uses toolkit `cuBLAS` directly for GEMV hot paths
- Runtime-compiles helper kernels on first use
- Offloads:
  - `RMSNorm`
  - linear / `lm_head` GEMV
  - RoPE
  - MLA score/softmax/value accumulation
  - device-side MLA KV cache updates
  - token-local MoE routing and expert accumulation
- Keeps hidden-state, norm, delta, logits, and MLA cache buffers resident on the
  device inside the executor
- Streams large weights row-by-row to the GPU instead of forcing a CPU fallback
- Exposes CUDA hit/fallback stats for parity tests

## Main Limitations

- real BF16 DeepSeek-V2-Lite weights do not fit for practical inference on the
  current CPU-only path
- the `RTX 4060 Laptop 8GB` cannot hold the raw full model in VRAM
- tokenizer implementation is still lightweight and not fully Hugging Face
  compatible
- CUDA path still launches many small kernels and row-chunk GEMV uploads, so
  real-model latency is still high
- there is not yet a quantized or paged-weight path for fitting more of the
  real model in 8 GB VRAM

## Next Good Steps

- reduce kernel launch count and weight streaming overhead on the real model
- add chunked expert decode/compute for very large MoE weights
- add a quantized or reduced-memory route for running Lite-class models on 8 GB
  GPUs
