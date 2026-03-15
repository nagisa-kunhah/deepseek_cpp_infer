#!/usr/bin/env bash

set -euo pipefail

METADATA_ROOT="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
HEADER="Metadata-Flavor: Google"
RESULT_ROOT="/var/tmp/ds_results"
MODEL_ROOT="/var/tmp/ds_model"
STATUS_PATH="${RESULT_ROOT}/status.json"
RAW_LOG="${RESULT_ROOT}/host.log"

mkdir -p "${RESULT_ROOT}" "${MODEL_ROOT}"
: > "${RAW_LOG}"

meta() {
  curl -fsSL -H "${HEADER}" "${METADATA_ROOT}/$1"
}

write_status() {
  local status="$1"
  local reason="$2"
  cat > "${STATUS_PATH}" <<EOF
{
  "status": "${status}",
  "failure_reason": "${reason}",
  "instance": "$(hostname)"
}
EOF
}

upload_results() {
  local result_gcs_uri="$1"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "${STATUS_PATH}" "${result_gcs_uri}/status.json" >> "${RAW_LOG}" 2>&1 || true
    [ -f "${RESULT_ROOT}/perf.json" ] && gcloud storage cp "${RESULT_ROOT}/perf.json" "${result_gcs_uri}/perf.json" >> "${RAW_LOG}" 2>&1 || true
    [ -f "${RESULT_ROOT}/report.md" ] && gcloud storage cp "${RESULT_ROOT}/report.md" "${result_gcs_uri}/report.md" >> "${RAW_LOG}" 2>&1 || true
    [ -f "${RESULT_ROOT}/raw.log" ] && gcloud storage cp "${RESULT_ROOT}/raw.log" "${result_gcs_uri}/raw.log" >> "${RAW_LOG}" 2>&1 || true
    gcloud storage cp "${RAW_LOG}" "${result_gcs_uri}/host.log" >> "${RAW_LOG}" 2>&1 || true
  fi
}

install_prereqs() {
  if ! command -v docker >/dev/null 2>&1; then
    apt-get update >> "${RAW_LOG}" 2>&1
    apt-get install -y docker.io >> "${RAW_LOG}" 2>&1
  fi

  if ! command -v gcloud >/dev/null 2>&1; then
    export CLOUD_SDK_REPO="cloud-sdk"
    apt-get update >> "${RAW_LOG}" 2>&1
    apt-get install -y apt-transport-https ca-certificates gnupg curl >> "${RAW_LOG}" 2>&1
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt ${CLOUD_SDK_REPO} main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list
    apt-get update >> "${RAW_LOG}" 2>&1
    apt-get install -y google-cloud-cli >> "${RAW_LOG}" 2>&1
  fi

  systemctl enable --now docker >> "${RAW_LOG}" 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >> "${RAW_LOG}" 2>&1 || true
  fi
}

main() {
  local docker_image
  local bench_case
  local git_sha
  local git_branch
  local machine_type
  local gpu_type
  local model_profile
  local result_gcs_uri
  local model_gcs_uri
  local hf_token
  local container_model_dir="/model"

  docker_image="$(meta docker_image)"
  bench_case="$(meta bench_case)"
  git_sha="$(meta git_sha)"
  git_branch="$(meta git_branch)"
  machine_type="$(meta machine_type)"
  gpu_type="$(meta gpu_type)"
  model_profile="$(meta model_profile)"
  result_gcs_uri="$(meta result_gcs_uri)"
  model_gcs_uri="$(meta model_gcs_uri || true)"
  hf_token="$(meta hf_token || true)"

  trap 'write_status failed startup_or_runtime_error; upload_results "${result_gcs_uri}"' ERR

  install_prereqs

  if [ -n "${model_gcs_uri}" ] && [ "${model_profile}" != "mock" ]; then
    gcloud storage cp --recursive "${model_gcs_uri}" "${MODEL_ROOT}" >> "${RAW_LOG}" 2>&1
    mapfile -t subdirs < <(find "${MODEL_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
    mapfile -t files < <(find "${MODEL_ROOT}" -mindepth 1 -maxdepth 1 -type f | sort)
    if [ "${#subdirs[@]}" -eq 1 ] && [ "${#files[@]}" -eq 0 ]; then
      container_model_dir="/model/$(basename "${subdirs[0]}")"
    fi
  fi

  registry_host="$(printf '%s' "${docker_image}" | cut -d/ -f1)"
  gcloud auth configure-docker "${registry_host}" --quiet >> "${RAW_LOG}" 2>&1 || true
  docker pull "${docker_image}" >> "${RAW_LOG}" 2>&1

  docker run --rm --gpus all \
    -e BENCH_CASE="${bench_case}" \
    -e MODEL_PROFILE="${model_profile}" \
    -e MODEL_DIR="${container_model_dir}" \
    -e MODEL_GCS_URI="${model_gcs_uri}" \
    -e HF_TOKEN="${hf_token}" \
    -e GIT_SHA="${git_sha}" \
    -e GIT_BRANCH="${git_branch}" \
    -e MACHINE_TYPE="${machine_type}" \
    -e GPU_TYPE="${gpu_type}" \
    -e OUTPUT_DIR=/results \
    -v "${RESULT_ROOT}:/results" \
    -v "${MODEL_ROOT}:/model:ro" \
    "${docker_image}" >> "${RAW_LOG}" 2>&1

  write_status success ""
  upload_results "${result_gcs_uri}"
}

main "$@"
