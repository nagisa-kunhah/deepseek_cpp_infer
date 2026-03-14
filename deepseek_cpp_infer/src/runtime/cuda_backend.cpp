#include "ds/runtime/cuda_backend.h"

#include <stdexcept>

#if DS_USE_CUDA

#include "ds/hf/decode.h"
#include "ds/hf/weight_ops.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ds::rt::cuda {
namespace {

constexpr std::size_t kWeightCacheThresholdBytes = 32ull << 20;
constexpr std::size_t kStreamWeightChunkBytes = 16ull << 20;
constexpr int kThreads = 256;

const char* kKernelSource = R"CUDA(
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

extern "C" __global__ void ds_add_inplace_kernel(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

extern "C" __global__ void ds_scale_add_kernel(float* dst, const float* src, float scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += scale * src[i];
}

extern "C" __global__ void ds_silu_mul_kernel(float* gate, const float* up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = gate[i];
  gate[i] = (x / (1.0f + expf(-x))) * up[i];
}

extern "C" __global__ void ds_rope_inplace_kernel(float* x, int dim, int pos, float theta) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int half = dim / 2;
  if (pair >= half) return;
  int i0 = pair * 2;
  int i1 = i0 + 1;
  float freq = powf(theta, -((float)i0 / (float)dim));
  float angle = (float)pos * freq;
  float c = cosf(angle);
  float s = sinf(angle);
  float a = x[i0];
  float b = x[i1];
  x[i0] = a * c - b * s;
  x[i1] = a * s + b * c;
}

extern "C" __global__ void ds_scores_kernel(const float* q, const float* k_base, int q_head, int seq_len,
                                             int key_stride, float scale, float* scores) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= seq_len) return;
  const float* k = k_base + t * key_stride;
  float acc = 0.0f;
  for (int i = 0; i < q_head; ++i) acc += q[i] * k[i];
  scores[t] = acc * scale;
}

extern "C" __global__ void ds_softmax_kernel(float* x, int n) {
  __shared__ float smax;
  __shared__ float ssum;
  if (threadIdx.x == 0) {
    float m = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > m) m = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      x[i] = expf(x[i] - m);
      sum += x[i];
    }
    smax = m;
    ssum = sum;
  }
  __syncthreads();
  float inv = ssum > 0.0f ? (1.0f / ssum) : 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] *= inv;
}

extern "C" __global__ void ds_weighted_sum_kernel(const float* scores, const float* v_base, int seq_len, int v_head,
                                                   int v_stride, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= v_head) return;
  float acc = 0.0f;
  for (int t = 0; t < seq_len; ++t) acc += scores[t] * v_base[t * v_stride + i];
  out[i] = acc;
}

extern "C" __global__ void ds_select_topk_kernel(const float* scores, int n, int topk, int normalize,
                                                  float routed_scale, int* out_ids, float* out_probs) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float best_scores[8];
  int best_ids[8];
  for (int i = 0; i < 8; ++i) {
    best_scores[i] = -3.402823466e+38F;
    best_ids[i] = 0;
  }

  for (int i = 0; i < n; ++i) {
    float s = scores[i];
    for (int j = 0; j < topk; ++j) {
      if (s > best_scores[j]) {
        for (int k = topk - 1; k > j; --k) {
          best_scores[k] = best_scores[k - 1];
          best_ids[k] = best_ids[k - 1];
        }
        best_scores[j] = s;
        best_ids[j] = i;
        break;
      }
    }
  }

  float denom = 0.0f;
  for (int i = 0; i < topk; ++i) denom += best_scores[i];
  if (!normalize || denom <= 0.0f) denom = 1.0f;

  for (int i = 0; i < topk; ++i) {
    out_ids[i] = best_ids[i];
    float p = best_scores[i];
    if (normalize) p /= denom;
    out_probs[i] = p * routed_scale;
  }
}
)CUDA";

enum class FallbackReason {
  UnsupportedShape,
  AllocFailed,
  CudaInitFailed,
  KernelError,
  CublasError,
};

struct DeviceBuffer {
  CUdeviceptr ptr = 0;
  std::size_t bytes = 0;
};

struct CachedWeight {
  CUdeviceptr ptr = 0;
  std::size_t bytes = 0;
};

struct CudaMLACache {
  std::size_t max_seq = 0;
  std::size_t n_heads = 0;
  std::size_t q_head_dim = 0;
  std::size_t v_head_dim = 0;
  DeviceBuffer k;
  DeviceBuffer v;
};

class CudaContext;

void bump_fallback(FallbackReason reason) {
  auto& s = const_cast<CudaStats&>(stats());
  switch (reason) {
    case FallbackReason::UnsupportedShape: ++s.fallback_unsupported_shape; break;
    case FallbackReason::AllocFailed: ++s.fallback_alloc_failed; break;
    case FallbackReason::CudaInitFailed: ++s.fallback_cuda_init_failed; break;
    case FallbackReason::KernelError: ++s.fallback_kernel_error; break;
    case FallbackReason::CublasError: ++s.fallback_cublas_error; break;
  }
}

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

[[noreturn]] void fail_cublas(cublasStatus_t st, const char* where) {
  std::ostringstream oss;
  oss << where << " failed: cublas status " << static_cast<int>(st);
  throw std::runtime_error(oss.str());
}

void check_cuda(CUresult st, const char* where) {
  if (st != CUDA_SUCCESS) fail_cuda(st, where);
}

void check_nvrtc(nvrtcResult st, const char* where, const std::string& log = std::string()) {
  if (st != NVRTC_SUCCESS) fail_nvrtc(st, where, log);
}

void check_cublas(cublasStatus_t st, const char* where) {
  if (st != CUBLAS_STATUS_SUCCESS) fail_cublas(st, where);
}

inline float* dptr(CUdeviceptr ptr) { return reinterpret_cast<float*>(static_cast<uintptr_t>(ptr)); }
inline const float* dptr_const(CUdeviceptr ptr) { return reinterpret_cast<const float*>(static_cast<uintptr_t>(ptr)); }
inline int* iptr(CUdeviceptr ptr) { return reinterpret_cast<int*>(static_cast<uintptr_t>(ptr)); }

class CudaContext {
 public:
  CudaContext() {
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_cuda(cuCtxCreate(&ctx_, nullptr, 0, device_), "cuCtxCreate");
    compile_kernels();
    check_cublas(cublasCreate(&cublas_), "cublasCreate");
  }

  ~CudaContext() {
    for (auto& kv : weight_cache_) {
      if (kv.second.ptr != 0) cuMemFree(kv.second.ptr);
    }
    if (cublas_ != nullptr) cublasDestroy(cublas_);
    if (module_ != nullptr) cuModuleUnload(module_);
    if (ctx_ != nullptr) cuCtxDestroy(ctx_);
  }

  void ensure_buffer(DeviceBuffer* buf, std::size_t bytes) {
    if (buf->bytes >= bytes) return;
    if (buf->ptr != 0) check_cuda(cuMemFree(buf->ptr), "cuMemFree(buffer)");
    buf->ptr = 0;
    buf->bytes = 0;
    check_cuda(cuMemAlloc(&buf->ptr, bytes), "cuMemAlloc(buffer)");
    buf->bytes = bytes;
  }

  void zero_buffer(DeviceBuffer* buf, std::size_t bytes) {
    ensure_buffer(buf, bytes);
    check_cuda(cuMemsetD8(buf->ptr, 0, bytes), "cuMemsetD8");
  }

  bool linear_device(const ds::hf::TensorSlice& weight, CUdeviceptr x, std::size_t in, CUdeviceptr y, std::size_t out) {
    if (weight.shape.size() != 2) return false;
    if (static_cast<std::size_t>(weight.shape[0]) != out || static_cast<std::size_t>(weight.shape[1]) != in) return false;

    try {
      const CachedWeight* cached = maybe_get_cached_weight(weight);
      if (cached != nullptr) {
        gemv_row_major(cached->ptr, out, in, x, y);
        return true;
      }

      stream_linear(weight, x, in, y, out);
      return true;
    } catch (const std::runtime_error&) {
      throw;
    }
  }

  bool rmsnorm_device(CUdeviceptr x, const float* w, std::size_t n, float eps, CUdeviceptr y) {
    ensure_buffer(&scratch_w_, n * sizeof(float));
    check_cuda(cuMemcpyHtoD(scratch_w_.ptr, w, n * sizeof(float)), "cuMemcpyHtoD(rmsnorm w)");
    int n_i = static_cast<int>(n);
    void* args[] = {&x, &scratch_w_.ptr, &n_i, &eps, &y};
    check_cuda(cuLaunchKernel(rmsnorm_kernel_, 1, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(rmsnorm)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(rmsnorm)");
    return true;
  }

  void add_inplace(CUdeviceptr dst, CUdeviceptr src, std::size_t n) {
    int n_i = static_cast<int>(n);
    void* args[] = {&dst, &src, &n_i};
    const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    check_cuda(cuLaunchKernel(add_inplace_kernel_, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(add_inplace)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(add_inplace)");
  }

  void scale_add(CUdeviceptr dst, CUdeviceptr src, float scale, std::size_t n) {
    int n_i = static_cast<int>(n);
    void* args[] = {&dst, &src, &scale, &n_i};
    const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    check_cuda(cuLaunchKernel(scale_add_kernel_, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(scale_add)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(scale_add)");
  }

  void silu_mul(CUdeviceptr gate, CUdeviceptr up, std::size_t n) {
    int n_i = static_cast<int>(n);
    void* args[] = {&gate, &up, &n_i};
    const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    check_cuda(cuLaunchKernel(silu_mul_kernel_, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(silu_mul)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(silu_mul)");
  }

  void rope_inplace(CUdeviceptr x, std::size_t dim, std::size_t pos, float theta) {
    if (dim == 0) return;
    int dim_i = static_cast<int>(dim);
    int pos_i = static_cast<int>(pos);
    void* args[] = {&x, &dim_i, &pos_i, &theta};
    const unsigned grid = static_cast<unsigned>(((dim / 2) + kThreads - 1) / kThreads);
    check_cuda(cuLaunchKernel(rope_kernel_, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(rope)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(rope)");
  }

  void scores(CUdeviceptr q, CUdeviceptr k_base, std::size_t q_head, std::size_t seq_len, std::size_t key_stride,
              float scale, CUdeviceptr out_scores) {
    int q_head_i = static_cast<int>(q_head);
    int seq_len_i = static_cast<int>(seq_len);
    int key_stride_i = static_cast<int>(key_stride);
    void* args[] = {&q, &k_base, &q_head_i, &seq_len_i, &key_stride_i, &scale, &out_scores};
    const unsigned grid = static_cast<unsigned>((seq_len + kThreads - 1) / kThreads);
    check_cuda(cuLaunchKernel(scores_kernel_, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(scores)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(scores)");
  }

  void softmax(CUdeviceptr x, std::size_t n) {
    int n_i = static_cast<int>(n);
    void* args[] = {&x, &n_i};
    check_cuda(cuLaunchKernel(softmax_kernel_, 1, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(softmax)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(softmax)");
  }

  void weighted_sum(CUdeviceptr scores_ptr, CUdeviceptr v_base, std::size_t seq_len, std::size_t v_head,
                    std::size_t v_stride, CUdeviceptr out) {
    int seq_len_i = static_cast<int>(seq_len);
    int v_head_i = static_cast<int>(v_head);
    int v_stride_i = static_cast<int>(v_stride);
    void* args[] = {&scores_ptr, &v_base, &seq_len_i, &v_head_i, &v_stride_i, &out};
    const unsigned grid = static_cast<unsigned>((v_head + kThreads - 1) / kThreads);
    check_cuda(cuLaunchKernel(weighted_sum_kernel_, grid, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(weighted_sum)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(weighted_sum)");
  }

  void select_topk(CUdeviceptr scores_ptr, std::size_t n, std::size_t top_k, bool normalize, float routed_scale,
                   DeviceBuffer* ids, DeviceBuffer* probs) {
    ensure_buffer(ids, top_k * sizeof(std::int32_t));
    ensure_buffer(probs, top_k * sizeof(float));
    int n_i = static_cast<int>(n);
    int topk_i = static_cast<int>(top_k);
    int normalize_i = normalize ? 1 : 0;
    void* args[] = {&scores_ptr, &n_i, &topk_i, &normalize_i, &routed_scale, &ids->ptr, &probs->ptr};
    check_cuda(cuLaunchKernel(select_topk_kernel_, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(select_topk)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(select_topk)");
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
    check_nvrtc(nvrtcCreateProgram(&prog, kKernelSource, "ds_runtime_kernels.cu", 0, nullptr, nullptr),
                "nvrtcCreateProgram");

    const std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
    const char* opts[] = {arch.c_str(), "--std=c++14"};
    const auto compile_st = nvrtcCompileProgram(prog, 2, opts);

    std::size_t log_size = 0;
    check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size), "nvrtcGetProgramLogSize");
    std::string log(log_size, '\0');
    if (log_size > 1) check_nvrtc(nvrtcGetProgramLog(prog, log.data()), "nvrtcGetProgramLog");
    if (compile_st != NVRTC_SUCCESS) fail_nvrtc(compile_st, "nvrtcCompileProgram", log);

    std::size_t image_size = 0;
    check_nvrtc(nvrtcGetCUBINSize(prog, &image_size), "nvrtcGetCUBINSize");
    std::string image(image_size, '\0');
    check_nvrtc(nvrtcGetCUBIN(prog, image.data()), "nvrtcGetCUBIN");
    check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");

    check_cuda(cuModuleLoadDataEx(&module_, image.data(), 0, nullptr, nullptr), "cuModuleLoadDataEx");
    check_cuda(cuModuleGetFunction(&rmsnorm_kernel_, module_, "ds_rmsnorm_kernel"), "cuModuleGetFunction(rmsnorm)");
    check_cuda(cuModuleGetFunction(&add_inplace_kernel_, module_, "ds_add_inplace_kernel"),
               "cuModuleGetFunction(add_inplace)");
    check_cuda(cuModuleGetFunction(&scale_add_kernel_, module_, "ds_scale_add_kernel"),
               "cuModuleGetFunction(scale_add)");
    check_cuda(cuModuleGetFunction(&silu_mul_kernel_, module_, "ds_silu_mul_kernel"), "cuModuleGetFunction(silu_mul)");
    check_cuda(cuModuleGetFunction(&rope_kernel_, module_, "ds_rope_inplace_kernel"), "cuModuleGetFunction(rope)");
    check_cuda(cuModuleGetFunction(&scores_kernel_, module_, "ds_scores_kernel"), "cuModuleGetFunction(scores)");
    check_cuda(cuModuleGetFunction(&softmax_kernel_, module_, "ds_softmax_kernel"), "cuModuleGetFunction(softmax)");
    check_cuda(cuModuleGetFunction(&weighted_sum_kernel_, module_, "ds_weighted_sum_kernel"),
               "cuModuleGetFunction(weighted_sum)");
    check_cuda(cuModuleGetFunction(&select_topk_kernel_, module_, "ds_select_topk_kernel"),
               "cuModuleGetFunction(select_topk)");
  }

  const CachedWeight* maybe_get_cached_weight(const ds::hf::TensorSlice& weight) {
    const void* key = weight.data;
    auto it = weight_cache_.find(key);
    if (it != weight_cache_.end()) return &it->second;
    if (weight.nbytes > kWeightCacheThresholdBytes) return nullptr;

    auto decoded = ds::hf::decode_to_f32(weight);
    CUdeviceptr ptr = 0;
    const std::size_t bytes = decoded.size() * sizeof(float);
    try {
      check_cuda(cuMemAlloc(&ptr, bytes), "cuMemAlloc(weight)");
    } catch (const std::runtime_error&) {
      return nullptr;
    }
    check_cuda(cuMemcpyHtoD(ptr, decoded.data(), bytes), "cuMemcpyHtoD(weight)");
    auto [inserted_it, _] = weight_cache_.emplace(key, CachedWeight{ptr, bytes});
    return &inserted_it->second;
  }

  void gemv_row_major(CUdeviceptr d_weight, std::size_t rows, std::size_t cols, CUdeviceptr d_x, CUdeviceptr d_y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(cublasSgemv(cublas_, CUBLAS_OP_T, static_cast<int>(cols), static_cast<int>(rows), &alpha, dptr_const(d_weight),
                             static_cast<int>(cols), dptr_const(d_x), 1, &beta, dptr(d_y), 1),
                 "cublasSgemv");
  }

  void stream_linear(const ds::hf::TensorSlice& weight, CUdeviceptr d_x, std::size_t in, CUdeviceptr d_y, std::size_t out) {
    const std::size_t rows_per_chunk = std::max<std::size_t>(1, kStreamWeightChunkBytes / (in * sizeof(float)));
    std::size_t row0 = 0;
    while (row0 < out) {
      const std::size_t rows = std::min(rows_per_chunk, out - row0);
      auto decoded = ds::hf::decode_rows_to_f32(weight, row0, rows);
      ensure_buffer(&scratch_weight_, decoded.size() * sizeof(float));
      check_cuda(cuMemcpyHtoD(scratch_weight_.ptr, decoded.data(), decoded.size() * sizeof(float)),
                 "cuMemcpyHtoD(weight chunk)");
      gemv_row_major(scratch_weight_.ptr, rows, in, d_x, d_y + row0 * sizeof(float));
      row0 += rows;
    }
  }

  CUdevice device_ = 0;
  CUcontext ctx_ = nullptr;
  CUmodule module_ = nullptr;
  cublasHandle_t cublas_ = nullptr;

  CUfunction rmsnorm_kernel_ = nullptr;
  CUfunction add_inplace_kernel_ = nullptr;
  CUfunction scale_add_kernel_ = nullptr;
  CUfunction silu_mul_kernel_ = nullptr;
  CUfunction rope_kernel_ = nullptr;
  CUfunction scores_kernel_ = nullptr;
  CUfunction softmax_kernel_ = nullptr;
  CUfunction weighted_sum_kernel_ = nullptr;
  CUfunction select_topk_kernel_ = nullptr;

  DeviceBuffer scratch_w_;
  DeviceBuffer scratch_weight_;
  std::unordered_map<const void*, CachedWeight> weight_cache_;
};

CudaContext& context() {
  static CudaContext ctx;
  return ctx;
}

std::once_flag init_once;
std::string init_error;
bool init_ok = false;
CudaStats g_stats;

DeviceBuffer& slot_buffer(CudaExecutorState* state, DeviceBufferSlot slot);
CUdeviceptr slot_ptr(CudaExecutorState* state, DeviceBufferSlot slot, std::size_t n_floats);

} // namespace

class CudaExecutorState {
 public:
  CudaExecutorState(const ds::hf::DeepSeekConfig& cfg, std::size_t max_seq, std::size_t n_layers)
      : cfg_(cfg), max_seq_(max_seq), n_layers_(n_layers) {
    const std::size_t n_heads = static_cast<std::size_t>(cfg_.num_attention_heads);
    const std::size_t q_head_dim = static_cast<std::size_t>(cfg_.qk_nope_head_dim + cfg_.qk_rope_head_dim);
    const std::size_t v_head_dim = static_cast<std::size_t>(cfg_.v_head_dim);
    caches_.resize(n_layers_);
    for (auto& cache : caches_) {
      cache.max_seq = max_seq_;
      cache.n_heads = n_heads;
      cache.q_head_dim = q_head_dim;
      cache.v_head_dim = v_head_dim;
      context().ensure_buffer(&cache.k, max_seq_ * n_heads * q_head_dim * sizeof(float));
      context().ensure_buffer(&cache.v, max_seq_ * n_heads * v_head_dim * sizeof(float));
      context().zero_buffer(&cache.k, max_seq_ * n_heads * q_head_dim * sizeof(float));
      context().zero_buffer(&cache.v, max_seq_ * n_heads * v_head_dim * sizeof(float));
    }
  }

  ~CudaExecutorState() {
    for (auto& buf : slots_) {
      if (buf.ptr != 0) check_cuda(cuMemFree(buf.ptr), "cuMemFree(slot)");
    }
    for (auto& cache : caches_) {
      if (cache.k.ptr != 0) check_cuda(cuMemFree(cache.k.ptr), "cuMemFree(cache.k)");
      if (cache.v.ptr != 0) check_cuda(cuMemFree(cache.v.ptr), "cuMemFree(cache.v)");
    }
    if (topk_ids_.ptr != 0) check_cuda(cuMemFree(topk_ids_.ptr), "cuMemFree(topk_ids)");
    if (topk_probs_.ptr != 0) check_cuda(cuMemFree(topk_probs_.ptr), "cuMemFree(topk_probs)");
  }

  CudaExecutorState(const CudaExecutorState&) = delete;
  CudaExecutorState& operator=(const CudaExecutorState&) = delete;

  CudaMLACache& cache(std::size_t layer_id) { return caches_.at(layer_id); }
  DeviceBuffer& topk_ids() { return topk_ids_; }
  DeviceBuffer& topk_probs() { return topk_probs_; }

  void reset() {
    for (auto& cache : caches_) {
      context().zero_buffer(&cache.k, cache.max_seq * cache.n_heads * cache.q_head_dim * sizeof(float));
      context().zero_buffer(&cache.v, cache.max_seq * cache.n_heads * cache.v_head_dim * sizeof(float));
    }
  }

 public:
  ds::hf::DeepSeekConfig cfg_;
  std::size_t max_seq_ = 0;
  std::size_t n_layers_ = 0;
  std::array<DeviceBuffer, 12> slots_;
  std::vector<CudaMLACache> caches_;
  DeviceBuffer topk_ids_;
  DeviceBuffer topk_probs_;
};

namespace {

DeviceBuffer& slot_buffer(CudaExecutorState* state, DeviceBufferSlot slot) {
  return state->slots_[static_cast<std::size_t>(slot)];
}

CUdeviceptr slot_ptr(CudaExecutorState* state, DeviceBufferSlot slot, std::size_t n_floats) {
  auto& buf = slot_buffer(state, slot);
  context().ensure_buffer(&buf, n_floats * sizeof(float));
  return buf.ptr;
}

void mark_linear_hit() { ++g_stats.linear_cuda_hits; }
void mark_mla_hit() { ++g_stats.mla_cuda_hits; }
void mark_moe_hit() { ++g_stats.moe_cuda_hits; }

template <typename Fn>
bool call_cuda(FallbackReason fallback_reason, std::size_t* fallback_counter, Fn&& fn) {
  try {
    ensure_initialized();
    fn();
    return true;
  } catch (const std::runtime_error&) {
    ++(*fallback_counter);
    bump_fallback(fallback_reason);
    return false;
  }
}

} // namespace

bool available() {
  std::call_once(init_once, [] {
    try {
      (void)context();
      init_ok = true;
    } catch (const std::exception& e) {
      init_error = e.what();
      init_ok = false;
      bump_fallback(FallbackReason::CudaInitFailed);
    }
  });
  return init_ok;
}

void ensure_initialized() {
  if (!available()) {
    throw std::runtime_error("failed to initialize CUDA backend: " + init_error);
  }
}

void reset_stats() { g_stats = CudaStats{}; }

const CudaStats& stats() { return g_stats; }

CudaExecutorState* create_executor_state(const ds::hf::DeepSeekConfig& cfg, std::size_t max_seq, std::size_t n_layers) {
  ensure_initialized();
  return new CudaExecutorState(cfg, max_seq, n_layers);
}

void destroy_executor_state(CudaExecutorState* state) {
  delete state;
}

void reset_executor_state(CudaExecutorState* state) {
  if (state == nullptr) return;
  state->reset();
}

bool upload_to_slot(CudaExecutorState* state, DeviceBufferSlot slot, const float* src, std::size_t n) {
  if (state == nullptr) return false;
  try {
    const auto ptr = slot_ptr(state, slot, n);
    check_cuda(cuMemcpyHtoD(ptr, src, n * sizeof(float)), "cuMemcpyHtoD(slot)");
    return true;
  } catch (const std::runtime_error&) {
    bump_fallback(FallbackReason::AllocFailed);
    return false;
  }
}

bool download_from_slot(CudaExecutorState* state, DeviceBufferSlot slot, float* dst, std::size_t n) {
  if (state == nullptr) return false;
  try {
    const auto ptr = slot_ptr(state, slot, n);
    check_cuda(cuMemcpyDtoH(dst, ptr, n * sizeof(float)), "cuMemcpyDtoH(slot)");
    return true;
  } catch (const std::runtime_error&) {
    bump_fallback(FallbackReason::KernelError);
    return false;
  }
}

bool zero_slot(CudaExecutorState* state, DeviceBufferSlot slot, std::size_t n) {
  if (state == nullptr) return false;
  try {
    auto& buf = slot_buffer(state, slot);
    context().zero_buffer(&buf, n * sizeof(float));
    return true;
  } catch (const std::runtime_error&) {
    bump_fallback(FallbackReason::AllocFailed);
    return false;
  }
}

bool add_inplace(CudaExecutorState* state, DeviceBufferSlot dst, DeviceBufferSlot src, std::size_t n) {
  if (state == nullptr) return false;
  try {
    context().add_inplace(slot_ptr(state, dst, n), slot_ptr(state, src, n), n);
    return true;
  } catch (const std::runtime_error&) {
    bump_fallback(FallbackReason::KernelError);
    return false;
  }
}

bool linear_to_slot(CudaExecutorState* state, const ds::hf::TensorSlice& weight, DeviceBufferSlot x, std::size_t in,
                    DeviceBufferSlot y, std::size_t out) {
  if (state == nullptr) return false;
  if (weight.shape.size() != 2) {
    ++g_stats.linear_cuda_fallbacks;
    bump_fallback(FallbackReason::UnsupportedShape);
    return false;
  }
  try {
    if (!context().linear_device(weight, slot_ptr(state, x, in), in, slot_ptr(state, y, out), out)) {
      ++g_stats.linear_cuda_fallbacks;
      bump_fallback(FallbackReason::UnsupportedShape);
      return false;
    }
    mark_linear_hit();
    return true;
  } catch (const std::runtime_error& e) {
    ++g_stats.linear_cuda_fallbacks;
    if (std::string_view(e.what()).find("cublas") != std::string_view::npos) {
      bump_fallback(FallbackReason::CublasError);
    } else {
      bump_fallback(FallbackReason::KernelError);
    }
    return false;
  }
}

bool rmsnorm_to_slot(CudaExecutorState* state, DeviceBufferSlot x, const float* w, std::size_t n, float eps,
                     DeviceBufferSlot y) {
  if (state == nullptr) return false;
  try {
    return context().rmsnorm_device(slot_ptr(state, x, n), w, n, eps, slot_ptr(state, y, n));
  } catch (const std::runtime_error&) {
    bump_fallback(FallbackReason::KernelError);
    return false;
  }
}

bool dense_mlp_to_slot(CudaExecutorState* state, const DenseMLPWeights& mlp, std::size_t hidden_size, DeviceBufferSlot x,
                       DeviceBufferSlot y) {
  if (state == nullptr || !mlp.valid()) return false;
  try {
    const auto intermediate = static_cast<std::size_t>(mlp.gate_proj.weight->shape[0]);
    if (!linear_to_slot(state, *mlp.gate_proj.weight, x, hidden_size, DeviceBufferSlot::Tmp0, intermediate)) return false;
    if (!linear_to_slot(state, *mlp.up_proj.weight, x, hidden_size, DeviceBufferSlot::Tmp1, intermediate)) return false;
    context().silu_mul(slot_ptr(state, DeviceBufferSlot::Tmp0, intermediate), slot_ptr(state, DeviceBufferSlot::Tmp1, intermediate),
                       intermediate);
    return linear_to_slot(state, *mlp.down_proj.weight, DeviceBufferSlot::Tmp0, intermediate, y, hidden_size);
  } catch (const std::runtime_error&) {
    bump_fallback(FallbackReason::KernelError);
    return false;
  }
}

bool moe_to_slot(CudaExecutorState* state, const ds::hf::DeepSeekConfig& cfg, const MoEWeights& moe,
                 std::size_t hidden_size, DeviceBufferSlot x, DeviceBufferSlot y) {
  if (state == nullptr || !moe.valid()) return false;
  const auto n_experts = static_cast<std::size_t>(moe.experts.size());
  const std::size_t top_k = std::min<std::size_t>(std::max<std::int32_t>(1, cfg.moe_top_k), n_experts);
  if (top_k == 0 || top_k > 8) {
    ++g_stats.moe_cuda_fallbacks;
    bump_fallback(FallbackReason::UnsupportedShape);
    return false;
  }

  try {
    zero_slot(state, y, hidden_size);
    if (!linear_to_slot(state, *moe.gate.weight, x, hidden_size, DeviceBufferSlot::Tmp0, n_experts)) return false;
    context().softmax(slot_ptr(state, DeviceBufferSlot::Tmp0, n_experts), n_experts);
    context().select_topk(slot_ptr(state, DeviceBufferSlot::Tmp0, n_experts), n_experts, top_k, cfg.norm_topk_prob,
                          cfg.routed_scaling_factor, &state->topk_ids(), &state->topk_probs());

    std::vector<std::int32_t> top_ids(top_k, 0);
    std::vector<float> top_probs(top_k, 0.0f);
    check_cuda(cuMemcpyDtoH(top_ids.data(), state->topk_ids().ptr, top_k * sizeof(std::int32_t)), "cuMemcpyDtoH(topk ids)");
    check_cuda(cuMemcpyDtoH(top_probs.data(), state->topk_probs().ptr, top_k * sizeof(float)), "cuMemcpyDtoH(topk probs)");

    for (std::size_t i = 0; i < top_k; ++i) {
      if (!dense_mlp_to_slot(state, moe.experts[static_cast<std::size_t>(top_ids[i])].ffn, hidden_size, x,
                             DeviceBufferSlot::Tmp1)) {
        return false;
      }
      context().scale_add(slot_ptr(state, y, hidden_size), slot_ptr(state, DeviceBufferSlot::Tmp1, hidden_size), top_probs[i],
                          hidden_size);
    }

    if (moe.shared_experts.valid()) {
      if (!dense_mlp_to_slot(state, moe.shared_experts, hidden_size, x, DeviceBufferSlot::Tmp1)) return false;
      context().add_inplace(slot_ptr(state, y, hidden_size), slot_ptr(state, DeviceBufferSlot::Tmp1, hidden_size), hidden_size);
    }

    mark_moe_hit();
    return true;
  } catch (const std::runtime_error&) {
    ++g_stats.moe_cuda_fallbacks;
    bump_fallback(FallbackReason::KernelError);
    return false;
  }
}

bool mla_decode_to_slot(CudaExecutorState* state, const ds::hf::DeepSeekConfig& cfg, const AttentionWeights& attn,
                        std::size_t layer_id, std::size_t pos, std::size_t hidden_size, DeviceBufferSlot hidden,
                        DeviceBufferSlot out_hidden) {
  if (state == nullptr) return false;

  const std::size_t n_heads = static_cast<std::size_t>(cfg.num_attention_heads);
  const std::size_t qk_nope = static_cast<std::size_t>(cfg.qk_nope_head_dim);
  const std::size_t qk_rope = static_cast<std::size_t>(cfg.qk_rope_head_dim);
  const std::size_t v_head = static_cast<std::size_t>(cfg.v_head_dim);
  const std::size_t q_head = qk_nope + qk_rope;
  const std::size_t kv_rank = static_cast<std::size_t>(cfg.kv_lora_rank);
  const std::size_t seq_len = pos + 1;
  if (seq_len > state->cache(layer_id).max_seq) {
    ++g_stats.mla_cuda_fallbacks;
    bump_fallback(FallbackReason::UnsupportedShape);
    return false;
  }

  try {
    if (attn.use_q_lora) {
      if (!attn.q_a_layernorm.valid()) return false;
      const auto q_rank = static_cast<std::size_t>(attn.q_a_proj.weight->shape[0]);
      if (!linear_to_slot(state, *attn.q_a_proj.weight, hidden, hidden_size, DeviceBufferSlot::Tmp0, q_rank)) return false;
      if (!rmsnorm_to_slot(state, DeviceBufferSlot::Tmp0, attn.q_a_layernorm.data(), q_rank, cfg.rms_norm_eps,
                           DeviceBufferSlot::Tmp1)) {
        return false;
      }
      if (!linear_to_slot(state, *attn.q_b_proj.weight, DeviceBufferSlot::Tmp1, q_rank, DeviceBufferSlot::Tmp2,
                          n_heads * q_head)) {
        return false;
      }
    } else {
      if (!linear_to_slot(state, *attn.q_proj.weight, hidden, hidden_size, DeviceBufferSlot::Tmp2, n_heads * q_head)) {
        return false;
      }
    }

    if (!linear_to_slot(state, *attn.kv_a_proj_with_mqa.weight, hidden, hidden_size, DeviceBufferSlot::Tmp3,
                        kv_rank + qk_rope)) {
      return false;
    }
    check_cuda(cuMemcpyDtoD(slot_ptr(state, DeviceBufferSlot::Tmp4, kv_rank), slot_ptr(state, DeviceBufferSlot::Tmp3, kv_rank + qk_rope),
                            kv_rank * sizeof(float)),
               "cuMemcpyDtoD(kv compressed)");
    if (!rmsnorm_to_slot(state, DeviceBufferSlot::Tmp4, attn.kv_a_layernorm.data(), kv_rank, cfg.rms_norm_eps,
                         DeviceBufferSlot::Tmp4)) {
      return false;
    }
    if (!linear_to_slot(state, *attn.kv_b_proj.weight, DeviceBufferSlot::Tmp4, kv_rank, DeviceBufferSlot::Tmp5,
                        n_heads * (qk_nope + v_head))) {
      return false;
    }

    if (qk_rope > 0) {
      check_cuda(cuMemcpyDtoD(slot_ptr(state, DeviceBufferSlot::Tmp6, qk_rope),
                              slot_ptr(state, DeviceBufferSlot::Tmp3, kv_rank + qk_rope) + kv_rank * sizeof(float),
                              qk_rope * sizeof(float)),
                 "cuMemcpyDtoD(shared_k_rope)");
      context().rope_inplace(slot_ptr(state, DeviceBufferSlot::Tmp6, qk_rope), qk_rope, pos, cfg.rope_theta);
    }

    zero_slot(state, DeviceBufferSlot::Tmp7, n_heads * v_head);
    auto& cache = state->cache(layer_id);
    const std::size_t key_stride = cache.n_heads * cache.q_head_dim;
    const std::size_t val_stride = cache.n_heads * cache.v_head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(q_head));

    for (std::size_t head = 0; head < n_heads; ++head) {
      check_cuda(cuMemcpyDtoD(slot_ptr(state, DeviceBufferSlot::Tmp0, q_head),
                              slot_ptr(state, DeviceBufferSlot::Tmp2, n_heads * q_head) + head * q_head * sizeof(float),
                              q_head * sizeof(float)),
                 "cuMemcpyDtoD(q head)");
      if (qk_rope > 0) {
        context().rope_inplace(slot_ptr(state, DeviceBufferSlot::Tmp0, q_head) + qk_nope * sizeof(float), qk_rope, pos, cfg.rope_theta);
      }

      const std::size_t key_offset = ((pos * n_heads + head) * q_head);
      const std::size_t val_offset = ((pos * n_heads + head) * v_head);
      const std::size_t kv_head_offset = head * (qk_nope + v_head);
      if (qk_nope > 0) {
        check_cuda(cuMemcpyDtoD(cache.k.ptr + key_offset * sizeof(float),
                                slot_ptr(state, DeviceBufferSlot::Tmp5, n_heads * (qk_nope + v_head)) + kv_head_offset * sizeof(float),
                                qk_nope * sizeof(float)),
                   "cuMemcpyDtoD(cache key)");
      }
      if (qk_rope > 0) {
        check_cuda(cuMemcpyDtoD(cache.k.ptr + (key_offset + qk_nope) * sizeof(float), slot_ptr(state, DeviceBufferSlot::Tmp6, qk_rope),
                                qk_rope * sizeof(float)),
                   "cuMemcpyDtoD(cache rope key)");
      }
      check_cuda(cuMemcpyDtoD(cache.v.ptr + val_offset * sizeof(float),
                              slot_ptr(state, DeviceBufferSlot::Tmp5, n_heads * (qk_nope + v_head)) +
                                  (kv_head_offset + qk_nope) * sizeof(float),
                              v_head * sizeof(float)),
                 "cuMemcpyDtoD(cache value)");

      const CUdeviceptr head_k_base = cache.k.ptr + head * q_head * sizeof(float);
      const CUdeviceptr head_v_base = cache.v.ptr + head * v_head * sizeof(float);
      context().scores(slot_ptr(state, DeviceBufferSlot::Tmp0, q_head), head_k_base, q_head, seq_len, key_stride, scale,
                       slot_ptr(state, DeviceBufferSlot::Tmp1, seq_len));
      context().softmax(slot_ptr(state, DeviceBufferSlot::Tmp1, seq_len), seq_len);
      context().weighted_sum(slot_ptr(state, DeviceBufferSlot::Tmp1, seq_len), head_v_base, seq_len, v_head, val_stride,
                             slot_ptr(state, DeviceBufferSlot::Tmp7, n_heads * v_head) + head * v_head * sizeof(float));
    }

    if (!linear_to_slot(state, *attn.o_proj.weight, DeviceBufferSlot::Tmp7, n_heads * v_head, out_hidden, hidden_size)) return false;
    mark_mla_hit();
    return true;
  } catch (const std::runtime_error&) {
    ++g_stats.mla_cuda_fallbacks;
    bump_fallback(FallbackReason::KernelError);
    return false;
  }
}

bool linear_try(const ds::hf::TensorSlice& weight, const float* x, std::size_t in, float* y, std::size_t out) {
  static CudaExecutorState* host_state = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    ds::hf::DeepSeekConfig dummy;
    host_state = create_executor_state(dummy, 1, 0);
  });

  if (!upload_to_slot(host_state, DeviceBufferSlot::Hidden, x, in)) return false;
  if (!linear_to_slot(host_state, weight, DeviceBufferSlot::Hidden, in, DeviceBufferSlot::Delta, out)) return false;
  return download_from_slot(host_state, DeviceBufferSlot::Delta, y, out);
}

bool rmsnorm_try(const float* x, const float* w, std::size_t n, float eps, float* y) {
  static CudaExecutorState* host_state = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    ds::hf::DeepSeekConfig dummy;
    host_state = create_executor_state(dummy, 1, 0);
  });

  if (!upload_to_slot(host_state, DeviceBufferSlot::Hidden, x, n)) return false;
  if (!rmsnorm_to_slot(host_state, DeviceBufferSlot::Hidden, w, n, eps, DeviceBufferSlot::Delta)) return false;
  return download_from_slot(host_state, DeviceBufferSlot::Delta, y, n);
}

} // namespace ds::rt::cuda

#else

namespace ds::rt::cuda {

class CudaExecutorState {};

bool available() { return false; }
void ensure_initialized() { throw std::runtime_error("CUDA support is not built in"); }
void reset_stats() {}
const CudaStats& stats() {
  static CudaStats s;
  return s;
}
CudaExecutorState* create_executor_state(const ds::hf::DeepSeekConfig&, std::size_t, std::size_t) {
  throw std::runtime_error("CUDA support is not built in");
}
void destroy_executor_state(CudaExecutorState*) {}
void reset_executor_state(CudaExecutorState*) {}
bool upload_to_slot(CudaExecutorState*, DeviceBufferSlot, const float*, std::size_t) { return false; }
bool download_from_slot(CudaExecutorState*, DeviceBufferSlot, float*, std::size_t) { return false; }
bool zero_slot(CudaExecutorState*, DeviceBufferSlot, std::size_t) { return false; }
bool add_inplace(CudaExecutorState*, DeviceBufferSlot, DeviceBufferSlot, std::size_t) { return false; }
bool linear_to_slot(CudaExecutorState*, const ds::hf::TensorSlice&, DeviceBufferSlot, std::size_t, DeviceBufferSlot,
                    std::size_t) {
  return false;
}
bool rmsnorm_to_slot(CudaExecutorState*, DeviceBufferSlot, const float*, std::size_t, float, DeviceBufferSlot) { return false; }
bool dense_mlp_to_slot(CudaExecutorState*, const DenseMLPWeights&, std::size_t, DeviceBufferSlot, DeviceBufferSlot) {
  return false;
}
bool moe_to_slot(CudaExecutorState*, const ds::hf::DeepSeekConfig&, const MoEWeights&, std::size_t, DeviceBufferSlot,
                 DeviceBufferSlot) {
  return false;
}
bool mla_decode_to_slot(CudaExecutorState*, const ds::hf::DeepSeekConfig&, const AttentionWeights&, std::size_t, std::size_t,
                        std::size_t, DeviceBufferSlot, DeviceBufferSlot) {
  return false;
}
bool linear_try(const ds::hf::TensorSlice&, const float*, std::size_t, float*, std::size_t) { return false; }
bool rmsnorm_try(const float*, const float*, std::size_t, float, float*) { return false; }

} // namespace ds::rt::cuda

#endif
