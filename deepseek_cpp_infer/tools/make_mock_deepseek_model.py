#!/usr/bin/env python3

import json
import os
import struct
import sys
from pathlib import Path


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def flatten_rows(rows):
    out = []
    for row in rows:
        out.extend(row)
    return out


def write_safetensors(path: Path, tensors) -> None:
    header = {}
    payload = bytearray()
    for name, shape, values in tensors:
        if len(values) != prod(shape):
            raise ValueError(f"tensor {name} has wrong element count")
        start = len(payload)
        payload.extend(struct.pack("<" + "f" * len(values), *values))
        end = len(payload)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [start, end],
        }

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def main() -> int:
    if len(sys.argv) != 2:
      print("usage: make_mock_deepseek_model.py <out_dir>", file=sys.stderr)
      return 1

    out_dir = Path(sys.argv[1]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_type": "deepseek_v2",
        "hidden_size": 4,
        "intermediate_size": 2,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 6,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "norm_topk_prob": 1,
        "routed_scaling_factor": 1.0,
        "scoring_func": "softmax",
        "topk_method": "greedy",
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 0,
        "qk_rope_head_dim": 2,
        "v_head_dim": 1,
    }
    write_json(out_dir / "config.json", config)

    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
        "model": {
            "type": "WordPiece",
            "unk_token": "<unk>",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": {
                "hello": 0,
                "world": 1,
                "!": 2,
                "<eos>": 3,
                "<bos>": 4,
                "<unk>": 5,
            },
        },
        "bos_token_id": 4,
        "eos_token_id": 3,
    }
    write_json(out_dir / "tokenizer.json", tokenizer)

    embed = flatten_rows([
        [0.20, 0.00, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
        [0.00, 0.00, 0.30, 0.00],
        [0.00, 0.00, 0.00, 0.35],
        [0.10, 0.10, 0.10, 0.10],
        [0.05, -0.05, 0.05, -0.05],
    ])
    final_norm = [1.0, 1.0, 1.0, 1.0]
    lm_head = flatten_rows([
        [0.20, 0.00, 0.00, 0.00],
        [0.00, 0.30, 0.00, 0.00],
        [0.00, 0.00, 0.40, 0.00],
        [0.00, 0.00, 0.00, 0.50],
        [0.10, 0.10, 0.10, 0.10],
        [0.05, -0.05, 0.05, -0.05],
    ])

    q_proj = flatten_rows([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    kv_a_proj = flatten_rows([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    kv_a_ln = [1.0, 1.0]
    kv_b_proj = flatten_rows([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    o_proj = flatten_rows([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    dense_gate = flatten_rows([
        [0.25, 0.00, 0.00, 0.00],
        [0.00, 0.25, 0.00, 0.00],
    ])
    dense_up = flatten_rows([
        [0.20, 0.00, 0.00, 0.00],
        [0.00, 0.20, 0.00, 0.00],
    ])
    dense_down = flatten_rows([
        [0.20, 0.00],
        [0.00, 0.20],
        [0.10, 0.00],
        [0.00, 0.10],
    ])

    moe_gate = flatten_rows([
        [1.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00],
    ])
    shared_gate = flatten_rows([
        [0.15, 0.00, 0.00, 0.00],
        [0.00, 0.15, 0.00, 0.00],
    ])
    shared_up = flatten_rows([
        [0.10, 0.00, 0.00, 0.00],
        [0.00, 0.10, 0.00, 0.00],
    ])
    shared_down = flatten_rows([
        [0.10, 0.00],
        [0.00, 0.10],
        [0.05, 0.00],
        [0.00, 0.05],
    ])
    expert0_gate = flatten_rows([
        [0.30, 0.00, 0.00, 0.00],
        [0.00, 0.10, 0.00, 0.00],
    ])
    expert0_up = flatten_rows([
        [0.20, 0.00, 0.00, 0.00],
        [0.00, 0.05, 0.00, 0.00],
    ])
    expert0_down = flatten_rows([
        [0.25, 0.00],
        [0.00, 0.15],
        [0.05, 0.00],
        [0.00, 0.05],
    ])
    expert1_gate = flatten_rows([
        [0.05, 0.00, 0.00, 0.00],
        [0.00, 0.30, 0.00, 0.00],
    ])
    expert1_up = flatten_rows([
        [0.05, 0.00, 0.00, 0.00],
        [0.00, 0.20, 0.00, 0.00],
    ])
    expert1_down = flatten_rows([
        [0.10, 0.00],
        [0.00, 0.25],
        [0.00, 0.05],
        [0.05, 0.00],
    ])

    tensors = [
        ("model.embed_tokens.weight", (6, 4), embed),
        ("model.norm.weight", (4,), final_norm),
        ("lm_head.weight", (6, 4), lm_head),
    ]

    for layer in range(2):
        prefix = f"model.layers.{layer}."
        tensors.extend([
            (prefix + "input_layernorm.weight", (4,), [1.0, 1.0, 1.0, 1.0]),
            (prefix + "post_attention_layernorm.weight", (4,), [1.0, 1.0, 1.0, 1.0]),
            (prefix + "self_attn.q_proj.weight", (4, 4), q_proj),
            (prefix + "self_attn.kv_a_layernorm.weight", (2,), kv_a_ln),
            (prefix + "self_attn.kv_a_proj_with_mqa.weight", (4, 4), kv_a_proj),
            (prefix + "self_attn.kv_b_proj.weight", (2, 2), kv_b_proj),
            (prefix + "self_attn.o_proj.weight", (4, 2), o_proj),
        ])

    tensors.extend([
        ("model.layers.0.mlp.gate_proj.weight", (2, 4), dense_gate),
        ("model.layers.0.mlp.up_proj.weight", (2, 4), dense_up),
        ("model.layers.0.mlp.down_proj.weight", (4, 2), dense_down),
        ("model.layers.1.mlp.gate.weight", (2, 4), moe_gate),
        ("model.layers.1.mlp.shared_experts.gate_proj.weight", (2, 4), shared_gate),
        ("model.layers.1.mlp.shared_experts.up_proj.weight", (2, 4), shared_up),
        ("model.layers.1.mlp.shared_experts.down_proj.weight", (4, 2), shared_down),
        ("model.layers.1.mlp.experts.0.gate_proj.weight", (2, 4), expert0_gate),
        ("model.layers.1.mlp.experts.0.up_proj.weight", (2, 4), expert0_up),
        ("model.layers.1.mlp.experts.0.down_proj.weight", (4, 2), expert0_down),
        ("model.layers.1.mlp.experts.1.gate_proj.weight", (2, 4), expert1_gate),
        ("model.layers.1.mlp.experts.1.up_proj.weight", (2, 4), expert1_up),
        ("model.layers.1.mlp.experts.1.down_proj.weight", (4, 2), expert1_down),
    ])

    shard_name = "model-00001-of-00001.safetensors"
    write_safetensors(out_dir / shard_name, tensors)

    index = {
        "metadata": {"total_size": os.path.getsize(out_dir / shard_name)},
        "weight_map": {name: shard_name for name, _, _ in tensors},
    }
    write_json(out_dir / "model.safetensors.index.json", index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
