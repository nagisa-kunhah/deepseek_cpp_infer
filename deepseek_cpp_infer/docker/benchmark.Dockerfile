FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

RUN cmake -S . -B build -DDS_USE_CUDA=ON \
    && cmake --build build -j"$(nproc)"

ENV REPO_ROOT=/workspace
ENV BUILD_DIR=/workspace/build

ENTRYPOINT ["/workspace/scripts/bench/run_suite.sh"]
