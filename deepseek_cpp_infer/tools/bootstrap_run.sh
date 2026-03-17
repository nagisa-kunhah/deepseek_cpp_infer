#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

usage() {
  cat <<'EOF'
usage:
  tools/bootstrap_run.sh [--build-dir <dir>] [--skip-build] <model_dir> [ds_chat args...]
  tools/bootstrap_run.sh [--build-dir <dir>] [--skip-build] --mock [ds_chat args...]
  tools/bootstrap_run.sh <model_dir> [ds_chat args...]
  tools/bootstrap_run.sh --mock [ds_chat args...]

examples:
  tools/bootstrap_run.sh /models/deepseek verify
  tools/bootstrap_run.sh /models/deepseek run --prompt "hello"
  tools/bootstrap_run.sh --skip-build /models/deepseek generate --prompt "hello" --max-new-tokens 8
  tools/bootstrap_run.sh --mock generate --prompt "hello world" --max-new-tokens 3

notes:
  - Initializes git submodules if needed
  - Configures and builds the project into ./build by default
  - Requires rustc and cargo only for the build step
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: missing required command: $1" >&2
    exit 1
  fi
}

ensure_submodule() {
  if [[ ! -f "${ROOT_DIR}/vendor/tokenizers-cpp/CMakeLists.txt" ]]; then
    echo "Initializing submodules..."
    git -C "${ROOT_DIR}" submodule update --init --recursive
  fi
}

build_project() {
  require_cmd cmake
  require_cmd python3
  require_cmd rustc
  require_cmd cargo

  ensure_submodule

  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
}

main() {
  local skip_build=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --build-dir)
        if [[ $# -lt 2 ]]; then
          echo "error: --build-dir requires a value" >&2
          exit 1
        fi
        BUILD_DIR="$2"
        shift 2
        ;;
      --skip-build)
        skip_build=1
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        break
        ;;
    esac
  done

  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local model_dir=""
  if [[ "$1" == "--mock" ]]; then
    model_dir="/tmp/ds_mock_model"
    python3 "${ROOT_DIR}/tools/make_mock_deepseek_model.py" "${model_dir}"
    shift
    if [[ $# -eq 0 ]]; then
      set -- verify
    fi
  else
    model_dir="$1"
    shift
    if [[ $# -eq 0 ]]; then
      set -- verify
    fi
  fi

  if [[ "${skip_build}" -eq 0 ]]; then
    build_project
  elif [[ ! -x "${BUILD_DIR}/ds_chat" ]]; then
    echo "error: --skip-build was set but ${BUILD_DIR}/ds_chat does not exist" >&2
    exit 1
  fi

  exec "${BUILD_DIR}/ds_chat" "${model_dir}" "$@"
}

main "$@"
