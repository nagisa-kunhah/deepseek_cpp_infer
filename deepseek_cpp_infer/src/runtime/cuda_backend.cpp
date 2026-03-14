#include "ds/runtime/cuda_backend.h"

#include <stdexcept>

#if DS_USE_CUDA

#include "ds/hf/decode.h"

#include <cuda.h>
#include <nvrtc.h>

#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ds::rt::cuda {
namespace {

constexpr std::size_t kMaxCachedWeightBytes = 64ull << 20;
constexpr int kThreads = 256;

const char* kKernelSource = R"CUDA(
extern "C" __global__ void ds_matvec_kernel(const float* w, const float* x, int in, int out, float* y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= out) return;
  float acc = 0.0f;
  int base = row * in;
  for (int i = 0; i < in; ++i) acc += w[base + i] * x[i];
  y[row] = acc;
}

extern "C" __global__ void ds_rmsnorm_kernel(const float* x, const float* w, int n, float eps, float* y) {
  __shared__ float partial[256];
  int tid = threadIdx.x;
  float local = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) local += x[i] * x[i];
  partial[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }

  float inv = rsqrtf(partial[0] / (float)n + eps);
  for (int i = tid; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}
)CUDA";

[[noreturn]] void fail_cuda(CUresult st, const char* where) {
  const char* name = nullptr;
  const char* msg = nullptr;
  cuGetErrorName(st, &name);
  cuGetErrorString(st, &msg);
  std::ostringstream oss;
  oss << where << " failed";
  if (name) oss << " [" << name << "]";
  if (msg) oss << ": " << msg;
  throw std::runtime_error(oss.str());
}

[[noreturn]] void fail_nvrtc(nvrtcResult st, const char* where, const std::string& log = std::string()) {
  std::ostringstream oss;
  oss << where << " failed: " << nvrtcGetErrorString(st);
  if (!log.empty()) oss << "\n" << log;
  throw std::runtime_error(oss.str());
}

void check_cuda(CUresult st, const char* where) {
  if (st != CUDA_SUCCESS) fail_cuda(st, where);
}

void check_nvrtc(nvrtcResult st, const char* where, const std::string& log = std::string()) {
  if (st != NVRTC_SUCCESS) fail_nvrtc(st, where, log);
}

struct CachedWeight {
  CUdeviceptr ptr = 0;
  std::size_t nbytes = 0;
};

class CudaContext {
 public:
  CudaContext() {
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_cuda(cuCtxCreate(&ctx_, 0, device_), "cuCtxCreate");
    compile_kernels();
  }

  ~CudaContext() {
    for (auto& kv : weight_cache_) {
      if (kv.second.ptr != 0) cuMemFree(kv.second.ptr);
    }
    if (scratch_x_ != 0) cuMemFree(scratch_x_);
    if (scratch_y_ != 0) cuMemFree(scratch_y_);
    if (scratch_w_ != 0) cuMemFree(scratch_w_);
    if (module_ != nullptr) cuModuleUnload(module_);
    if (ctx_ != nullptr) cuCtxDestroy(ctx_);
  }

  bool linear_try(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y, std::size_t out) {
    if (weight.nbytes > kMaxCachedWeightBytes) return false;
    if (weight.shape.size() != 2) return false;

    const auto* d_weight = get_or_upload_weight(weight);
    ensure_buffer(in * sizeof(float), &scratch_x_, &scratch_x_cap_);
    ensure_buffer(out * sizeof(float), &scratch_y_, &scratch_y_cap_);

    check_cuda(cuMemcpyHtoD(scratch_x_, x, in * sizeof(float)), "cuMemcpyHtoD(x)");

    void* args[] = {
        const_cast<CUdeviceptr*>(&d_weight->ptr),
        &scratch_x_,
        &in,
        &out,
        &scratch_y_,
    };
    const unsigned grid = static_cast<unsigned>((out + kThreads - 1) / kThreads);
    check_cuda(cuLaunchKernel(matvec_kernel_, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr), "cuLaunchKernel(matvec)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(matvec)");
    check_cuda(cuMemcpyDtoH(y, scratch_y_, out * sizeof(float)), "cuMemcpyDtoH(y)");
    return true;
  }

  bool rmsnorm_try(const float* x, const float* w, std::size_t n, float eps, float* y) {
    ensure_buffer(n * sizeof(float), &scratch_x_, &scratch_x_cap_);
    ensure_buffer(n * sizeof(float), &scratch_w_, &scratch_w_cap_);
    ensure_buffer(n * sizeof(float), &scratch_y_, &scratch_y_cap_);

    check_cuda(cuMemcpyHtoD(scratch_x_, x, n * sizeof(float)), "cuMemcpyHtoD(rmsnorm x)");
    check_cuda(cuMemcpyHtoD(scratch_w_, w, n * sizeof(float)), "cuMemcpyHtoD(rmsnorm w)");

    int n_i = static_cast<int>(n);
    void* args[] = {&scratch_x_, &scratch_w_, &n_i, &eps, &scratch_y_};
    check_cuda(cuLaunchKernel(rmsnorm_kernel_, 1, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr), "cuLaunchKernel(rmsnorm)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(rmsnorm)");
    check_cuda(cuMemcpyDtoH(y, scratch_y_, n * sizeof(float)), "cuMemcpyDtoH(rmsnorm y)");
    return true;
  }

 private:
  void compile_kernels() {
    int major = 0;
    int minor = 0;
    check_cuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
               "cuDeviceGetAttribute(major)");
    check_cuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_),
               "cuDeviceGetAttribute(minor)");

    nvrtcProgram prog = nullptr;
    check_nvrtc(nvrtcCreateProgram(&prog, kKernelSource, "ds_runtime_kernels.cu", 0, nullptr, nullptr), "nvrtcCreateProgram");

    const std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
    const char* opts[] = {arch.c_str(), "--std=c++14"};
    const auto compile_st = nvrtcCompileProgram(prog, 2, opts);

    std::size_t log_size = 0;
    check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size), "nvrtcGetProgramLogSize");
    std::string log(log_size, '\0');
    if (log_size > 1) check_nvrtc(nvrtcGetProgramLog(prog, log.data()), "nvrtcGetProgramLog");
    if (compile_st != NVRTC_SUCCESS) fail_nvrtc(compile_st, "nvrtcCompileProgram", log);

    std::size_t ptx_size = 0;
    check_nvrtc(nvrtcGetPTXSize(prog, &ptx_size), "nvrtcGetPTXSize");
    std::string ptx(ptx_size, '\0');
    check_nvrtc(nvrtcGetPTX(prog, ptx.data()), "nvrtcGetPTX");
    check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");

    check_cuda(cuModuleLoadDataEx(&module_, ptx.data(), 0, nullptr, nullptr), "cuModuleLoadDataEx");
    check_cuda(cuModuleGetFunction(&matvec_kernel_, module_, "ds_matvec_kernel"), "cuModuleGetFunction(matvec)");
    check_cuda(cuModuleGetFunction(&rmsnorm_kernel_, module_, "ds_rmsnorm_kernel"), "cuModuleGetFunction(rmsnorm)");
  }

  const CachedWeight* get_or_upload_weight(const ds::hf::TensorSlice& weight) {
    const void* key = weight.data;
    auto it = weight_cache_.find(key);
    if (it != weight_cache_.end()) return &it->second;

    auto decoded = ds::hf::decode_to_f32(weight);
    const std::size_t bytes = decoded.size() * sizeof(float);
    CUdeviceptr ptr = 0;
    check_cuda(cuMemAlloc(&ptr, bytes), "cuMemAlloc(weight)");
    check_cuda(cuMemcpyHtoD(ptr, decoded.data(), bytes), "cuMemcpyHtoD(weight)");

    auto [inserted_it, _] = weight_cache_.emplace(key, CachedWeight{.ptr = ptr, .nbytes = bytes});
    return &inserted_it->second;
  }

  void ensure_buffer(std::size_t bytes, CUdeviceptr* ptr, std::size_t* cap) {
    if (*cap >= bytes) return;
    if (*ptr != 0) check_cuda(cuMemFree(*ptr), "cuMemFree(scratch)");
    *ptr = 0;
    *cap = 0;
    check_cuda(cuMemAlloc(ptr, bytes), "cuMemAlloc(scratch)");
    *cap = bytes;
  }

  CUdevice device_ = 0;
  CUcontext ctx_ = nullptr;
  CUmodule module_ = nullptr;
  CUfunction matvec_kernel_ = nullptr;
  CUfunction rmsnorm_kernel_ = nullptr;

  CUdeviceptr scratch_x_ = 0;
  CUdeviceptr scratch_y_ = 0;
  CUdeviceptr scratch_w_ = 0;
  std::size_t scratch_x_cap_ = 0;
  std::size_t scratch_y_cap_ = 0;
  std::size_t scratch_w_cap_ = 0;
  std::unordered_map<const void*, CachedWeight> weight_cache_;
};

CudaContext& context() {
  static CudaContext ctx;
  return ctx;
}

std::once_flag init_once;
std::string init_error;
bool init_ok = false;

} // namespace

bool available() {
  std::call_once(init_once, [] {
    try {
      (void)context();
      init_ok = true;
    } catch (const std::exception& e) {
      init_error = e.what();
      init_ok = false;
    }
  });
  return init_ok;
}

void ensure_initialized() {
  if (!available()) {
    throw std::runtime_error("failed to initialize CUDA backend: " + init_error);
  }
}

bool linear_try(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y, std::size_t out) {
  ensure_initialized();
  int in_i = static_cast<int>(in);
  int out_i = static_cast<int>(out);
  return context().linear_try(weight, x, static_cast<std::size_t>(in_i), y, static_cast<std::size_t>(out_i));
}

bool rmsnorm_try(const float* x, const float* w, std::size_t n, float eps, float* y) {
  ensure_initialized();
  return context().rmsnorm_try(x, w, n, eps, y);
}

} // namespace ds::rt::cuda

#else

namespace ds::rt::cuda {

bool available() { return false; }
void ensure_initialized() { throw std::runtime_error("CUDA support is not built in"); }
bool linear_try(const ds::hf::TensorSlice&, const float*, std::size_t, float*, std::size_t) { return false; }
bool rmsnorm_try(const float*, const float*, std::size_t, float, float*) { return false; }

} // namespace ds::rt::cuda

#endif
