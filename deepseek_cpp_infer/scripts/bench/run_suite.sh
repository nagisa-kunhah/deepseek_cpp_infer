#!/usr/bin/env bash

set -euo pipefail

BENCH_CASE="${BENCH_CASE:-gpu_smoke}"
REPO_ROOT="${REPO_ROOT:-/workspace}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUTPUT_DIR="${OUTPUT_DIR:-/results}"
MODEL_PROFILE="${MODEL_PROFILE:-mock}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_GCS_URI="${MODEL_GCS_URI:-}"
BACKEND="${BACKEND:-cuda}"
PROMPT="${PROMPT:-hello world!}"
PROMPT_IDS="${PROMPT_IDS:-0,1,2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
GIT_SHA="${GIT_SHA:-unknown}"
GIT_BRANCH="${GIT_BRANCH:-unknown}"
MACHINE_TYPE="${MACHINE_TYPE:-unknown}"
GPU_TYPE="${GPU_TYPE:-unknown}"

mkdir -p "${OUTPUT_DIR}"

if [ "${BENCH_CASE}" = "gpu_smoke" ]; then
  status="success"
  failure_reason=""
  if ! ctest --test-dir "${BUILD_DIR}" -R "ds_cuda_runtime_smoke|ds_cli_cuda_e2e" --output-on-failure \
    > "${OUTPUT_DIR}/raw.log" 2>&1; then
    status="failed"
    failure_reason="gpu_smoke_ctest_failed"
  fi

  python3 "${REPO_ROOT}/scripts/bench/collect_report.py" \
    --output-dir "${OUTPUT_DIR}" \
    --git-sha "${GIT_SHA}" \
    --git-branch "${GIT_BRANCH}" \
    --bench-case "${BENCH_CASE}" \
    --machine-type "${MACHINE_TYPE}" \
    --gpu-type "${GPU_TYPE}" \
    --backend "${BACKEND}" \
    --model-profile "${MODEL_PROFILE}" \
    --status "${status}" \
    --failure-reason "${failure_reason}"
  if [ "${status}" = "success" ]; then
    exit 0
  fi
  exit 1
fi

exec python3 "${REPO_ROOT}/scripts/bench/run_real_benchmark.py" \
  --repo-root "${REPO_ROOT}" \
  --build-dir "${BUILD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --backend "${BACKEND}" \
  --model-profile "${MODEL_PROFILE}" \
  --model-dir "${MODEL_DIR}" \
  --prompt "${PROMPT}" \
  --prompt-ids "${PROMPT_IDS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --git-sha "${GIT_SHA}" \
  --git-branch "${GIT_BRANCH}" \
  --machine-type "${MACHINE_TYPE}" \
  --gpu-type "${GPU_TYPE}"
