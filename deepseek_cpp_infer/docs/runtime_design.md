# Runtime Design

## Layers

The runtime is split into five layers:

1. HF adapters
   - config parsing
   - safetensors index parsing
   - mmap-backed tensor lookup
2. Weight registry
   - converts HF tensor names into semantic per-layer structures
3. Model executor
   - owns caches, temporary buffers, and generation state
4. Backend ops
   - CPU reference implementations
   - future CUDA implementations behind the same interfaces
5. CLI
   - info, verify, strict, load, run, generate

## Public runtime types

### BackendKind

- `cpu`
- `cuda`

`cuda` is already part of the public API even though the current implementation
only provides a CPU reference path. This keeps CLI and executor interfaces
stable while CUDA kernels are introduced incrementally.

### RunConfig

- backend selection
- maximum decode length
- verbose mode flag

### GenerationConfig

- `max_new_tokens`
- `temperature`
- `top_k`
- `top_p`
- `seed`

### Tokenizer

- loads `tokenizer.json`
- exposes `encode(text)` and `decode(ids)`
- keeps tokenizer metadata such as BOS/EOS IDs when present

The default runtime tokenizer is now backed by `tokenizers-cpp`, which wraps
the Hugging Face tokenizers runtime behind a C++ interface. The project keeps a
separate minimal tokenizer implementation only for tests, mock fixtures, and
smoke paths that intentionally avoid the external dependency surface.

### WeightRegistry

- owns semantic views over global weights and per-layer weights
- classifies each decoder layer as dense or MoE
- keeps lazy tensor slices backed by model mmaps

### ModelExecutor

- owns the loaded model and weight registry
- allocates one attention cache per decoder layer
- exposes:
  - `prefill(prompt_ids)`
  - `decode_next(token_id)`
  - `generate(prompt_ids, generation_config)`

## Backend boundary

The executor owns the control flow. Backends own numerical kernels.

Current CPU backend responsibilities:

- RMSNorm
- streamed matvec from BF16/F16/F32 weights
- DeepSeek-V2 attention decode step
- dense MLP
- MoE routing and expert execution
- LM head logits

Future CUDA backend responsibilities:

- hot matvec/GEMM path through cuBLAS or cuBLASLt
- custom kernels for RMSNorm, RoPE, softmax, and MoE dispatch

The high-level executor sequence must remain identical across CPU and CUDA.

## Current limitations

- single batch only
- decode-only incremental path
- no continuous batching
- no KV cache paging
- no quantization
- tokenizer encode path is intentionally minimal
