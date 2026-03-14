# GPU Phase 1 复盘

日期：2026-03-14

这份文档总结了 `deepseek_cpp_infer` 在实现第一轮 CUDA 覆盖时遇到的主要问题、最终采用的处理方式，以及当前仍然没有解决完的部分。

## 概览

这一轮 GPU phase-1 工作，已经把原来偏“演示性质”的 bootstrap CUDA 路径，推进成了一个至少能在 mock 模型上完整工作的 device-resident decode 路径：

- 激活值和 MLA 的 KV cache 现在可以在 decode 过程中常驻 GPU
- 线性层改成走 `cuBLAS GEMV`
- MLA 的 score/softmax/value accumulation 和 token-local MoE 路由已经迁到 CUDA
- `ctest` 里的 CPU/CUDA 一致性和 CUDA 命中测试已经通过

这轮实现的困难，不是某一个单独的算法 bug，而是主要卡在三类问题上：

- 在 toolkit 装好之前，CUDA wheel fallback 环境和常规 toolkit 假设并不一致
- 长生命周期 CUDA executor state 引入了新的所有权和接口设计问题
- 在“正确性、8GB 显存限制、真实模型延迟”之间存在明显取舍

## 遇到的问题

### 1. 旧的 CUDA 路径覆盖太浅

原来的 CUDA backend 实际只加速了：

- `RMSNorm`
- 小尺寸 `matvec` 风格的 linear

但在真实的 DeepSeek-V2-Lite 单 token decode 里，真正重的核心计算仍然都留在 CPU：

- MLA cache 更新
- RoPE
- QK score 计算
- softmax
- V 加权累加
- MoE gate softmax/top-k
- expert 输出累加

这会导致 `--backend cuda` 虽然表面上可用，但大部分真实工作仍在 CPU。

最终处理：

- 增加了 device-resident 的 executor state
- 新增 CUDA MLA 路径，覆盖 score/softmax/value/cache update
- 新增 CUDA MoE 路径，覆盖 gate、top-k 和 expert accumulation
- 增加 CUDA hit/fallback 统计，避免测试只能证明“结果对”，却无法证明“真的跑在 GPU 上”

### 2. 小型 norm 权重在每个 token 上重复 decode

在代码排查过程中，一个很隐蔽但很伤性能的问题是：小的 norm 权重在每个 token step 上都被重复 `decode_to_f32()`。

受影响的包括：

- `input_layernorm.weight`
- `post_attention_layernorm.weight`
- `kv_a_layernorm.weight`
- `q_a_layernorm.weight`
- final norm

这不只是代码风格问题，它会在 decode 热路径上平白增加 CPU 运算和内存流量。

最终处理：

- `NormWeights` 现在直接带一份缓存后的 `f32` 副本
- norm 向量在 `WeightRegistry` 构建时只 decode 一次
- CPU 和 CUDA 两条路径都复用这份缓存

### 3. CUDA wheel fallback 在早期够用，但不适合长期承载标准 cuBLAS 集成

在完整 toolkit 装好之前，这台机器只有：

- CUDA runtime wheel
- NVRTC wheel

但没有完整 CUDA toolkit。

这对早期的 `NVRTC + driver API` 路径是够用的，但一旦尝试直接 include `cublas_v2.h`，编译就会暴露 wheel 环境和完整 toolkit 不一致的问题，比如缺：

- `crt/host_defines.h`
- `nv/target`

也就是说，wheel 提供的头文件并不足以支撑一个标准的“编译期集成 cuBLAS”方案。

当时的临时处理：

- CMake 仍然发现并链接 wheel 里的 `libcublas.so.12`
- 但 backend 不再 include cuBLAS 头，而是改成用 `dlopen` + `dlsym` 在运行时解析：
  - `cublasCreate_v2`
  - `cublasDestroy_v2`
  - `cublasSgemv_v2`
- 另外加了一个很小的本地 shim，补 `crt/host_defines.h` 的 include 路径问题

为什么当时这是合适的折中：

- 能继续兼容当时那台机器环境
- 不需要为了编译这一部分强依赖完整 toolkit
- 仍然能拿到真正的 cuBLAS GEMV 热路径

后续状态更新：

- 现在这台机器已经装好了完整 CUDA toolkit
- 项目构建已经切回标准 `CUDAToolkit + cublas_v2.h` 直连方式
- 之前的 `dlopen + dlsym` 和本地 shim 都已经移除

### 4. `unique_ptr` 持有不完整的 CUDA state 类型时，析构阶段会报错

`ModelExecutor` 需要持有一个长生命周期的 `CudaExecutorState`，但这个 state 的完整定义只存在于 CUDA backend 的实现文件里。

第一版做法是：

- 在头文件里前置声明
- `ModelExecutor` 里放 `std::unique_ptr<CudaExecutorState>`

结果编译失败，因为默认析构路径在不合适的位置仍然需要完整类型。

最终处理：

- `ModelExecutor` 现在保存原始指针 `CudaExecutorState*`
- 生命周期显式交给：
  - `create_executor_state(...)`
  - `destroy_executor_state(...)`
- `ModelExecutor::~ModelExecutor()` 改成 out-of-line 定义，显式完成释放

这没有“完整可见 RAII 类型”那么优雅，但它保住了头文件边界，也避免把 CUDA 实现细节泄漏到 runtime 的公共接口中。

### 5. 从 host-pointer kernel 切到 device-resident 执行后，很多原先默认前提都失效了

原来的 helper 函数默认都是：

- host 输入指针
- host 输出指针
- copy in -> launch -> copy out

一旦 executor 开始把 hidden state 和 cache 放在 GPU 上，这套默认前提就不成立了。

随之出现的新需求包括：

- 稳定的 device buffer slot
- layer-local scratch 复用
- device-side residual add
- device-side MLA cache 所有权
- 既能支持 host 输入，也能支持 device 输入的 linear 路径

最终处理：

- 引入 `DeviceBufferSlot`
- 引入可复用的 `CudaExecutorState`
- 增加 slot 的 upload/download/zero/add helper
- 保留旧的 `linear_try` / `rmsnorm_try` 这类 host 风格入口，保证兼容
- 另外新增真正给 executor 用的 device-to-device CUDA 入口

### 6. 在 8GB GPU 上，大权重不可能全部常驻 device cache

原来的小模型 CUDA backend 会缓存上传过的权重，同时用一个硬阈值来决定“大矩阵直接回退 CPU”。

但对真实 Lite 权重来说，这两种做法都不合适：

- 如果大矩阵一律缓存到显存，会很快把 VRAM 顶满
- 如果大矩阵一律回退 CPU，又会让 `--backend cuda` 再次失真

最终处理：

- 小/中型权重仍然走 cached device upload
- 大型 2D 权重改成 row-chunk streaming GEMV
- 这样至少能保证正确性，同时不必强行回退到 CPU

代价也很明确：

- 真实模型路径会比生产级 backend 慢得多
- 但它仍然明显好于“直接回到 CPU”或者“立刻 OOM”

### 7. CUDA 测试不能只验证输出对不对，还必须验证路径真的命中

如果一个混合 CPU/CUDA 实现只做输出一致性测试，其实是不够的。因为某些回归可能让代码悄悄退回 CPU，但在小夹具上输出仍然完全一样。

最终处理：

- 增加了 `CudaStats`
- 测试现在同时断言：
  - CPU/CUDA logits 一致
  - linear/MLA/MoE 的 CUDA hit 计数大于 0

这样可以更早发现“看起来通过，实际上已经偷偷回退 CPU”的假成功。

### 8. 真实模型的成功标准必须重新定义

在这台机器上：

- CPU 单 token 真实模型推理会 OOM
- GPU 单 token 真实模型推理现在不会立刻 OOM
- 但当前实现仍然无法在 90 秒 smoke timeout 内完成

这说明“mock 模型能跑通”和“真实模型已经具备实用性”是两件不同的事。

这轮最重要的经验之一就是：正确性和 GPU 覆盖度，不等于可用吞吐。这一阶段解决的是前两者，不是最后那个问题。

## 哪些做法效果比较好

- 尽早拿真实模型做检查是对的，它很快暴露出 CPU fallback 仍然过大的问题
- 保留完整 CPU reference 路径对 CUDA 数学调试帮助很大
- 用小而确定性的夹具验证 MLA 和 MoE 是正确方向
- 早点引入 stats，可以避免很多“看起来没坏，其实已经回退 CPU”的回归
- 在 toolkit 没装好的阶段，用动态加载 cuBLAS 是一个务实且有效的过渡方案

## 现在仍然痛的地方

- 真实模型 CUDA 路径仍然会发射很多小 kernel
- 大权重 row-chunk streaming 是正确的，但开销依然很高
- MoE 仍然是逐 expert 做很多重复工作，没有 batching 或 grouped GEMM
- 还没有量化或 paged-weight 路线去适配 8GB GPU
- tokenizer 仍然是轻量实现，不是完整 HF 兼容实现

## 建议的下一步

### 短期

- 减少 MLA decode 内部的小 kernel 启动次数
- 降低相邻 decode step 之间的大权重重复上传
- 为超大 MoE 权重增加 chunked expert decode/compute
- 对真实模型单 token CUDA 运行做 profile，找出最重热点

### 中期

- 给真实模型增加量化权重支持
- 为 8GB GPU 增加 paged 或 staged 的权重迁移路线
- 评估为 MoE 加入 grouped GEMM 或 expert batching

### 建议保持不变的部分

- 保留 CPU 路径作为 correctness reference
- 保留 CUDA hit/fallback 统计并继续纳入测试
- 现在已经有完整 toolkit，后续优先沿标准 `CUDAToolkit` 构建链继续演进
