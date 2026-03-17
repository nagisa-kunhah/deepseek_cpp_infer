cmake_minimum_required(VERSION 3.24)
project(deepseek_cpp_infer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_program(DS_RUSTC_EXECUTABLE rustc HINTS "$ENV{HOME}/.cargo/bin")
find_program(DS_CARGO_EXECUTABLE cargo HINTS "$ENV{HOME}/.cargo/bin")
if(NOT DS_RUSTC_EXECUTABLE OR NOT DS_CARGO_EXECUTABLE)
  message(FATAL_ERROR
    "Rust toolchain is required to build the external tokenizer runtime. "
    "Install rustup and ensure rustc/cargo are available (for example via $HOME/.cargo/bin).")
endif()

option(DS_USE_CUDA "Build with CUDA" ON)
set(DS_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures for project CUDA targets")

# Prefer an installed toolkit under /usr/local/cuda when available.
if(DS_USE_CUDA)
  if(EXISTS "/usr/local/cuda/bin/nvcc" AND NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
  endif()
  if(EXISTS "/usr/local/cuda" AND NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root")
  endif()
  enable_language(CUDA OPTIONAL)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(WARNING "No usable CUDAToolkit found; building CPU-only. Reconfigure with -DDS_USE_CUDA=OFF to silence this.")
    set(DS_USE_CUDA OFF)
  else()
    message(STATUS "Using CUDA architectures '${DS_CUDA_ARCHITECTURES}'")
  endif()
endif()

set(DS_CUDA_RUNTIME_SOURCES)
if(DS_USE_CUDA)
  list(APPEND DS_CUDA_RUNTIME_SOURCES
    src/ops/cuda/elementwise_kernels.cu
    src/ops/cuda/attention_kernels.cu
    src/ops/cuda/routing_kernels.cu
  )
endif()

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "Disable unused sentencepiece tokenizer support" FORCE)
add_subdirectory(vendor/tokenizers-cpp)

add_executable(ds_chat
  src/main.cpp
  src/util/fs.cpp
  src/util/json.cpp
  src/util/path.cpp
  src/util/mmap.cpp
  src/core/dtype.cpp
  src/core/tensor.cpp
  src/ops/core/ops.cpp
  src/core/math.cpp
  src/core/sampler.cpp
  src/ops/core/rope.cpp
  src/core/kv_cache.cpp
  src/ops/core/attention.cpp
  src/ops/core/mha.cpp
  src/hf/config.cpp
  src/hf/safetensors.cpp
  src/hf/weights_index.cpp
  src/hf/model_loader.cpp
  src/hf/decode.cpp
  src/hf/weight_ops.cpp
  src/runtime/backend.cpp
  src/runtime/model.cpp
  src/runtime/model_factory.cpp
  src/runtime/tokenizer.cpp
  src/runtime/model_executor.cpp
  src/models/core/registry.cpp
  src/models/deepseek/config.cpp
  src/models/deepseek/package.cpp
  src/models/deepseek/model.cpp
  src/models/deepseek/weights.cpp
  src/models/deepseek/ops.cpp
  src/models/deepseek/mla.cpp
  src/models/deepseek/cuda_backend.cpp
  ${DS_CUDA_RUNTIME_SOURCES}
)

target_include_directories(ds_chat PRIVATE include)

if(DS_USE_CUDA)
  target_compile_definitions(ds_chat PRIVATE DS_USE_CUDA=1)
  set_target_properties(ds_chat PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${DS_CUDA_ARCHITECTURES}"
  )
  target_link_libraries(ds_chat PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cublasLt)
else()
  target_compile_definitions(ds_chat PRIVATE DS_USE_CUDA=0)
endif()
target_link_libraries(ds_chat PRIVATE tokenizers_cpp)

enable_testing()

add_executable(ds_tests
  tests/runtime_smoke.cpp
  src/util/fs.cpp
  src/util/json.cpp
  src/util/mmap.cpp
  src/util/path.cpp
  src/core/dtype.cpp
  src/core/math.cpp
  src/ops/core/ops.cpp
  src/ops/core/rope.cpp
  src/core/sampler.cpp
  src/hf/decode.cpp
  src/hf/safetensors.cpp
  src/hf/config.cpp
  src/hf/model_loader.cpp
  src/hf/weight_ops.cpp
  src/hf/weights_index.cpp
  src/runtime/backend.cpp
  src/runtime/model.cpp
  src/runtime/model_factory.cpp
  src/runtime/tokenizer.cpp
  src/runtime/model_executor.cpp
  src/models/core/registry.cpp
  src/models/deepseek/config.cpp
  src/models/deepseek/package.cpp
  src/models/deepseek/model.cpp
  src/models/deepseek/weights.cpp
  src/models/deepseek/ops.cpp
  src/models/deepseek/mla.cpp
)

target_include_directories(ds_tests PRIVATE include)
target_compile_definitions(ds_tests PRIVATE DS_USE_CUDA=0)
target_link_libraries(ds_tests PRIVATE tokenizers_cpp)

add_test(NAME ds_runtime_smoke COMMAND ds_tests)
add_test(
  NAME ds_cli_e2e
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_e2e_cli.py ${CMAKE_SOURCE_DIR} $<TARGET_FILE:ds_chat>
)

if(DS_USE_CUDA)
  add_executable(ds_cuda_tests
    tests/cuda_runtime_smoke.cpp
    src/util/fs.cpp
    src/util/json.cpp
    src/util/mmap.cpp
    src/util/path.cpp
    src/core/dtype.cpp
    src/core/math.cpp
    src/ops/core/ops.cpp
    src/ops/core/rope.cpp
    src/core/sampler.cpp
    src/hf/decode.cpp
    src/hf/safetensors.cpp
    src/hf/config.cpp
    src/hf/model_loader.cpp
    src/hf/weights_index.cpp
    src/hf/weight_ops.cpp
    src/runtime/backend.cpp
    src/runtime/model.cpp
    src/runtime/model_factory.cpp
    src/runtime/tokenizer.cpp
    src/runtime/model_executor.cpp
    src/models/core/registry.cpp
    src/models/deepseek/config.cpp
    src/models/deepseek/package.cpp
    src/models/deepseek/model.cpp
    src/models/deepseek/weights.cpp
    src/models/deepseek/ops.cpp
    src/models/deepseek/mla.cpp
    src/models/deepseek/cuda_backend.cpp
    ${DS_CUDA_RUNTIME_SOURCES}
  )
  target_include_directories(ds_cuda_tests PRIVATE include)
  target_compile_definitions(ds_cuda_tests PRIVATE DS_USE_CUDA=1)
  set_target_properties(ds_cuda_tests PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${DS_CUDA_ARCHITECTURES}"
  )
  target_link_libraries(ds_cuda_tests PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cublasLt tokenizers_cpp)
  add_test(NAME ds_cuda_runtime_smoke COMMAND ds_cuda_tests)
  add_test(
    NAME ds_cli_cuda_e2e
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_e2e_cli.py ${CMAKE_SOURCE_DIR} $<TARGET_FILE:ds_chat> cuda
  )
endif()
