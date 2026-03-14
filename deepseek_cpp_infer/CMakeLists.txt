cmake_minimum_required(VERSION 3.22)
project(deepseek_cpp_infer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

option(DS_USE_CUDA "Build with CUDA" ON)
set(DS_CUDA_WHEEL_FALLBACK OFF)

# CUDA is optional during bootstrap (WSL may not have toolkit configured yet).
if(DS_USE_CUDA)
  enable_language(CUDA OPTIONAL)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c
              "import os, sys; import nvidia.cuda_runtime, nvidia.cuda_nvrtc; \
rt=os.path.dirname(nvidia.cuda_runtime.__file__); nv=os.path.dirname(nvidia.cuda_nvrtc.__file__); \
paths=[os.path.join(rt,'include'), os.path.join(rt,'lib','libcudart.so.12'), os.path.join(rt,'include','cuda.h'), os.path.join(nv,'include'), os.path.join(nv,'lib','libnvrtc.so.12'), os.path.join(nv,'lib','libnvrtc-builtins.so.12.4')]; \
print('\\n'.join(paths))"
      RESULT_VARIABLE DS_CUDA_WHEEL_RESULT
      OUTPUT_VARIABLE DS_CUDA_WHEEL_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_library(DS_LIBCUDA cuda PATHS /usr/lib/wsl/lib NO_DEFAULT_PATH)
    if(DS_CUDA_WHEEL_RESULT EQUAL 0 AND DS_LIBCUDA)
      string(REPLACE "\n" ";" DS_CUDA_WHEEL_LIST "${DS_CUDA_WHEEL_OUTPUT}")
      list(LENGTH DS_CUDA_WHEEL_LIST DS_CUDA_WHEEL_LEN)
      if(DS_CUDA_WHEEL_LEN EQUAL 6)
        list(GET DS_CUDA_WHEEL_LIST 0 DS_CUDA_RUNTIME_INCLUDE)
        list(GET DS_CUDA_WHEEL_LIST 1 DS_CUDART_LIB)
        list(GET DS_CUDA_WHEEL_LIST 2 DS_CUDA_HEADER)
        list(GET DS_CUDA_WHEEL_LIST 3 DS_NVRTC_INCLUDE)
        list(GET DS_CUDA_WHEEL_LIST 4 DS_NVRTC_LIB)
        list(GET DS_CUDA_WHEEL_LIST 5 DS_NVRTC_BUILTINS_LIB)
        if(EXISTS "${DS_CUDA_HEADER}" AND EXISTS "${DS_CUDART_LIB}" AND EXISTS "${DS_NVRTC_LIB}" AND EXISTS "${DS_NVRTC_BUILTINS_LIB}")
          set(DS_CUDA_WHEEL_FALLBACK ON)
          message(STATUS "Using CUDA wheel fallback (NVRTC + driver API)")
        endif()
      endif()
    endif()
    if(NOT DS_CUDA_WHEEL_FALLBACK)
      message(WARNING "No CUDAToolkit or CUDA wheel fallback found; building CPU-only. Reconfigure with -DDS_USE_CUDA=OFF to silence this.")
      set(DS_USE_CUDA OFF)
    endif()
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
  src/runtime/cuda_backend.cpp
)

target_include_directories(ds_chat PRIVATE include)

if(DS_USE_CUDA)
  target_compile_definitions(ds_chat PRIVATE DS_USE_CUDA=1)
  if(CUDAToolkit_FOUND)
    target_link_libraries(ds_chat PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
  elseif(DS_CUDA_WHEEL_FALLBACK)
    target_include_directories(ds_chat PRIVATE ${DS_CUDA_RUNTIME_INCLUDE} ${DS_NVRTC_INCLUDE})
    target_link_libraries(ds_chat PRIVATE ${DS_CUDART_LIB} ${DS_NVRTC_LIB} ${DS_LIBCUDA})
    set_target_properties(ds_chat PROPERTIES
      BUILD_RPATH "/usr/lib/wsl/lib;/home/nagisa/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib;/home/nagisa/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
    )
  endif()
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

if(DS_USE_CUDA)
  add_test(
    NAME ds_cli_cuda_e2e
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_e2e_cli.py ${CMAKE_SOURCE_DIR} $<TARGET_FILE:ds_chat> cuda
  )
endif()
