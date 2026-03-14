cmake_minimum_required(VERSION 3.22)
project(deepseek_cpp_infer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

option(DS_USE_CUDA "Build with CUDA" ON)

# CUDA is optional during bootstrap (WSL may not have toolkit configured yet).
if(DS_USE_CUDA)
  enable_language(CUDA OPTIONAL)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(WARNING "CUDAToolkit not found; building CPU-only. Reconfigure with -DDS_USE_CUDA=OFF to silence this.")
    set(DS_USE_CUDA OFF)
  endif()
endif()

add_executable(ds_chat
  src/main.cpp
  src/util/fs.cpp
  src/util/json.cpp
  src/util/path.cpp
  src/util/mmap.cpp
  src/core/dtype.cpp
  src/core/tensor.cpp
  src/core/ops.cpp
  src/core/math.cpp
  src/core/sampler.cpp
  src/core/rope.cpp
  src/core/kv_cache.cpp
  src/core/attention.cpp
  src/core/mha.cpp
  src/hf/config.cpp
  src/hf/safetensors.cpp
  src/hf/weights_index.cpp
  src/hf/model_loader.cpp
  src/hf/decode.cpp
  src/hf/weight_ops.cpp
  src/runtime/backend.cpp
  src/runtime/weights.cpp
  src/runtime/tokenizer.cpp
  src/runtime/ops.cpp
  src/runtime/mla.cpp
  src/runtime/model_executor.cpp
)

target_include_directories(ds_chat PRIVATE include)

if(DS_USE_CUDA)
  target_compile_definitions(ds_chat PRIVATE DS_USE_CUDA=1)
  target_link_libraries(ds_chat PRIVATE CUDA::cudart)
else()
  target_compile_definitions(ds_chat PRIVATE DS_USE_CUDA=0)
endif()

enable_testing()

add_executable(ds_tests
  tests/runtime_smoke.cpp
  src/util/fs.cpp
  src/util/json.cpp
  src/util/mmap.cpp
  src/core/dtype.cpp
  src/core/math.cpp
  src/core/ops.cpp
  src/core/rope.cpp
  src/hf/decode.cpp
  src/hf/safetensors.cpp
  src/runtime/weights.cpp
  src/runtime/tokenizer.cpp
  src/runtime/ops.cpp
  src/runtime/mla.cpp
)

target_include_directories(ds_tests PRIVATE include)
target_compile_definitions(ds_tests PRIVATE DS_USE_CUDA=0)

add_test(NAME ds_runtime_smoke COMMAND ds_tests)
add_test(
  NAME ds_cli_e2e
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_e2e_cli.py ${CMAKE_SOURCE_DIR} $<TARGET_FILE:ds_chat>
)
