# deepseek_triton_infer

一个“能跑起来”的最小推理框架雏形，目标对齐 DeepSeek-V2-Lite 风格的 **MoE FFN**（重点在路由 + expert MLP），并用 **Triton** 写关键算子。

这不是完整复刻 DeepSeek-V2-Lite（不含完整 attention / KV cache / 权重下载与转换流水线），而是把 MoE 这块抽出来做成可验证、可扩展的内核与模块，便于你后续往完整模型推理框架演进。

## 目前包含

- `ds_infer.moe.DeepSeekMoE`: MoE FFN 模块（SwiGLU 风格），支持 top-k 路由。
- Triton kernel：
  - `ds_infer.kernels.moe_swiglu`: 对单个 expert 的 `x @ W1` / `x @ W3` + SiLU + 逐元素乘 + `@ W2`（按路由后的 token 列表计算）
- 参考实现与对齐测试：用 PyTorch 实现同样的 MoE，数值对齐（fp16/bf16）。

## 快速开始

在 `deepseek_triton_infer/` 目录下：

```bash
python -m pip install -e .
pytest -q
python examples/run_moe.py
```

## 接下来可以做什么（建议）

1) 加入 KV cache + attention（再决定是否 Triton 化）。
2) 加入权重加载（HF safetensors）并做 DeepSeek-V2-Lite 结构映射。
3) MoE 的路由部分也 Triton 化（topk + prefix-sum / radix sort），并做更高效的 grouped GEMM。
