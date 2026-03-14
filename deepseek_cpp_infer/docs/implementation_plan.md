# Implementation Plan

## Milestone 1: docs and abstractions

Inputs:

- existing HF adapters and CLI verification flow
- DeepSeek-V2-Lite HF tensor naming

Outputs:

- architecture doc
- runtime design doc
- semantic weight registry
- backend and generation config types
- tokenizer loader
- model executor skeleton

Acceptance:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Risks:

- tokenizer behavior is not fully HF-compatible yet
- CUDA backend is API-only in the first pass

Rollback:

- old `info/verify/load` commands remain intact

## Milestone 2: CPU runnable path

Inputs:

- Milestone 1 interfaces
- mapped global and per-layer weights

Outputs:

- CPU decode cache per layer
- CPU attention step
- CPU dense MLP step
- CPU MoE routing and expert execution
- `run` and `generate` CLI paths

Acceptance:

```bash
./build/ds_chat <model_dir> run --prompt-ids 1,2,3
./build/ds_chat <model_dir> generate --prompt-ids 1,2,3 --max-new-tokens 1
```

Risks:

- exact DeepSeek-V2 rotary scaling and tokenizer parity are still incomplete
- performance is intentionally poor because streamed matvec is correctness-first

Rollback:

- keep `run/generate` behind explicit commands without changing existing verify

## Milestone 3: CUDA alignment

Inputs:

- working CPU executor
- stable backend interfaces

Outputs:

- backend dispatch layer
- CUDA smoke path for selected kernels
- CPU/CUDA numerical comparison fixtures

Acceptance:

```bash
cmake -S . -B build -DDS_USE_CUDA=ON
cmake --build build -j
```

Risks:

- CUDA toolkit variability across environments
- cuBLAS integration details still pending

Rollback:

- CPU build remains the default fallback

## Milestone 4: performance follow-up

Inputs:

- correct CPU execution path
- early CUDA path

Outputs:

- better buffer reuse
- staged weight decode
- CUDA hot path expansion

Acceptance:

- profiling-guided; not part of the bootstrap gate

Risks:

- optimization should not alter public executor flow
