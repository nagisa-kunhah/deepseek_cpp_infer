# Push 触发的按需 GPU Benchmark

这份文档描述仓库中的第一版 GPU benchmark 自动化方案：GitHub Actions 负责控制面，GCP 的一次性 GPU VM 负责执行面，跑完自动销毁。

## 开发约束

这套自动化必须在独立 worktree 中开发，不直接改主 worktree。

推荐命令：

```bash
git worktree add /path/to/gpu-benchmark-ci -b feat/gpu-benchmark-ci
cd /path/to/gpu-benchmark-ci
```

建议提交顺序：

1. CPU CI 与配置骨架
2. 按需 GPU VM 执行
3. 真实性能报告与 nightly

## 文件布局

- `.github/workflows/ci.yml`
  - 每次 push / PR 的 CPU-only 构建与测试
- `.github/workflows/gpu-benchmark.yml`
  - `main` push 后的 GPU smoke，以及手动触发的 benchmark
- `.github/workflows/nightly-benchmark.yml`
  - 固定周期的 nightly benchmark
- `.github/workflows/_gpu-benchmark-run.yml`
  - 复用的 GPU benchmark 执行逻辑
- `deepseek_cpp_infer/docker/benchmark.Dockerfile`
  - GPU benchmark 镜像
- `deepseek_cpp_infer/scripts/bench/run_suite.sh`
  - benchmark 统一入口
- `deepseek_cpp_infer/scripts/bench/run_real_benchmark.py`
  - 真实模型 benchmark 执行器
- `deepseek_cpp_infer/scripts/bench/collect_report.py`
  - `perf.json` 和 `report.md` 生成器
- `deepseek_cpp_infer/scripts/gcp/gpu_benchmark_startup.sh`
  - GCP VM 启动脚本

## 运行流

### CPU 快测

`ci.yml` 在 GitHub-hosted runner 上执行：

```bash
cmake -S . -B build -DDS_USE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### GPU smoke / benchmark

`gpu-benchmark.yml` 或 `nightly-benchmark.yml` 会：

1. checkout 指定 ref
2. 用 GitHub OIDC 登录 GCP
3. 用 `deepseek_cpp_infer/` 作为 Docker build context，构建并推送 benchmark 镜像到 Artifact Registry
4. 创建 `g2-standard-4 + 1x L4` 一次性 VM
5. 通过 metadata 注入 benchmark 参数
6. VM 启动后自动拉镜像并执行 benchmark
7. 结果上传到 GCS
8. Actions 下载结果、发布 summary、上传 artifact
9. 最后删除 VM
10. 如果 Spot 创建失败或运行结果未回传，会自动重试一次 on-demand VM

## 必要的仓库变量与密钥

必须配置的 repository variables：

- `GCP_PROJECT_ID`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`
- `GCP_GCS_BENCH_BUCKET`

建议配置的 repository variables：

- `GCP_REGION`
- `GCP_ZONE`
- `GCP_ARTIFACT_REGISTRY_REPOSITORY`
- `GCP_VM_IMAGE_PROJECT`
- `GCP_VM_IMAGE_FAMILY`
- `GCP_VM_SERVICE_ACCOUNT`
- `MODEL_GCS_URI`

可选 secrets：

- `HF_TOKEN`

## GCP 权限建议

GitHub Actions 代表的 service account 需要至少具备：

- 创建和删除 Compute Engine VM
- 推送 Artifact Registry 镜像
- 读写 benchmark GCS bucket

GPU VM 使用的 service account 需要至少具备：

- 读取 Artifact Registry 镜像
- 读取模型所在 GCS 路径
- 写入 benchmark 结果到 GCS bucket

## 结果格式

`perf.json` 包含以下字段：

- `git_sha`
- `branch`
- `bench_case`
- `machine_type`
- `gpu_type`
- `backend`
- `model_profile`
- `ttft_ms`
- `decode_tokens_per_s`
- `peak_vram_mb`
- `peak_rss_mb`
- `duration_s`
- `status`
- `failure_reason`

`report.md` 包含：

- 本次 commit 和执行环境
- smoke / benchmark 结果
- 核心性能指标
- 预留的基线对比区域

## 真实模型策略

第一版不建议每次从 Hugging Face 公网下载模型。

推荐顺序：

1. 先把真实模型放到同区域 GCS
2. VM 启动时复制到本地临时目录
3. 如果拉取时间太长，再引入只读持久盘缓存

## 本地验证

CPU CI：

```bash
cmake -S . -B /tmp/ds-gpu-bench-build -DDS_USE_CUDA=OFF
cmake --build /tmp/ds-gpu-bench-build -j
ctest --test-dir /tmp/ds-gpu-bench-build --output-on-failure
```

报告脚本：

```bash
python3 scripts/bench/run_real_benchmark.py \
  --repo-root . \
  --build-dir /tmp/ds-gpu-bench-build \
  --output-dir /tmp/ds-gpu-bench-report \
  --backend cpu \
  --model-profile mock \
  --model-dir '' \
  --prompt 'hello world!' \
  --prompt-ids 0,1,2 \
  --max-new-tokens 4 \
  --git-sha local \
  --git-branch feat/gpu-benchmark-ci \
  --machine-type local \
  --gpu-type none
```
