# DeepSeek-V2-Lite Architecture Notes

This repository targets `DeepSeek-V2-Lite` and `DeepSeek-V2-Lite-Chat` in their
Hugging Face safetensors layout.

## Model shape assumptions

- Decoder-only causal transformer.
- Per-layer structure:
  - `input_layernorm`
  - DeepSeek-V2 attention block
  - residual add
  - `post_attention_layernorm`
  - dense MLP or MoE MLP
  - residual add
- Final `model.norm` and `lm_head`.

## Attention block

The Lite checkpoints exposed by Hugging Face use:

- `self_attn.q_proj.weight`
- `self_attn.kv_a_proj_with_mqa.weight`
- `self_attn.kv_a_layernorm.weight`
- `self_attn.kv_b_proj.weight`
- `self_attn.o_proj.weight`

For Lite checkpoints, `q_lora_rank` is typically absent or null, so the runtime
uses the direct `q_proj` path instead of `q_a_proj/q_b_proj`.

The runtime models the attention step as:

1. Project hidden state to per-head queries.
2. Split query heads into:
   - non-RoPE part: `qk_nope_head_dim`
   - RoPE part: `qk_rope_head_dim`
3. Project hidden state through `kv_a_proj_with_mqa`.
4. Split that result into:
   - compressed KV state of rank `kv_lora_rank`
   - shared RoPE key slice of width `qk_rope_head_dim`
5. Normalize the compressed KV state with `kv_a_layernorm`.
6. Expand with `kv_b_proj` into per-head:
   - non-RoPE key part
   - value part
7. Build a full key per head by concatenating:
   - per-head non-RoPE key
   - shared RoPE key
8. Apply RoPE to the query RoPE part and shared key RoPE part for the current
   token position.
9. Store full per-head keys and values in a decode cache.
10. Perform causal attention over cached keys/values for each head.
11. Concatenate head outputs and project with `o_proj`.

This repository currently stores full expanded keys and values in the decode
cache. That keeps the execution path simple and matches the shapes consumed by
the CPU reference path. A future optimization pass can switch to a compressed KV
cache without changing the high-level executor API.

## Dense vs MoE layers

Layer classification is discovered from weights:

- Dense layer:
  - `mlp.gate_proj.weight`
  - `mlp.up_proj.weight`
  - `mlp.down_proj.weight`
- MoE layer:
  - `mlp.gate.weight`
  - `mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`
  - `mlp.experts.<id>.{gate_proj,up_proj,down_proj}.weight`

The runtime does not keep string-based weight lookups in the hot path. During
model initialization it converts HF tensor names into a semantic registry.

## MoE execution model

For a single token:

1. Compute router logits with `mlp.gate.weight`.
2. Apply `softmax` scoring.
3. Select top-k experts using `moe_top_k`.
4. Optionally renormalize selected probabilities if `norm_topk_prob` is set.
5. Evaluate each selected routed expert.
6. Sum routed expert outputs with router weights.
7. Add the shared expert branch when present.

The first implementation target is single-batch, single-session incremental
decode. Expert dispatch is therefore token-local and CPU friendly. Future batch
support should reuse the same semantic registry and backend API.

## Weight name mapping

Global tensors:

- `model.embed_tokens.weight`
- `model.norm.weight`
- `lm_head.weight`

Per-layer tensors:

- `model.layers.<i>.input_layernorm.weight`
- `model.layers.<i>.post_attention_layernorm.weight`
- `model.layers.<i>.self_attn.*`
- `model.layers.<i>.mlp.*`

## Explicitly out of scope for this bootstrap implementation

- Quantized weights
- Paged KV cache
- Continuous batching
- Distributed execution
- Automatic chat templates
- Generic DeepSeek-V2 family support beyond Lite and Lite-Chat
