#include "ds/models/deepseek/cuda_backend.h"
#include "ds/models/deepseek/cuda_kernels.h"

#include <stdexcept>

#if DS_USE_CUDA

#include "ds/hf/decode.h"
#include "ds/hf/weight_ops.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

constexpr std::size_t kStreamWeightChunkBytes = 16ull << 20;
constexpr int kThreads = 256;

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

enum class CachedWeightKind {
  RawF32,
  RawF16,
  RawBF16,
  DecodedF32,
};

struct CachedWeight {
  CUdeviceptr ptr = 0;
  std::size_t bytes = 0;
  CachedWeightKind kind = CachedWeightKind::DecodedF32;
};

struct CachedWeightKey {
  const void* data = nullptr;
  std::size_t nbytes = 0;
  ds::hf::DType dtype = ds::hf::DType::F32;
  std::string name;

  bool operator==(const CachedWeightKey& other) const {
    return data == other.data && nbytes == other.nbytes && dtype == other.dtype && name == other.name;
  }
};

struct CachedWeightKeyHash {
  std::size_t operator()(const CachedWeightKey& key) const {
    std::size_t h = std::hash<const void*>{}(key.data);
    h ^= std::hash<std::size_t>{}(key.nbytes) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(key.dtype)) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<std::string>{}(key.name) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
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

[[noreturn]] void fail_cublas(cublasStatus_t st, const char* where) {
  std::ostringstream oss;
  oss << where << " failed: cublas status " << static_cast<int>(st);
  throw std::runtime_error(oss.str());
}

[[noreturn]] void fail_runtime(cudaError_t st, const char* where) {
  std::ostringstream oss;
  oss << where << " failed: " << cudaGetErrorString(st);
  throw std::runtime_error(oss.str());
}

void check_cuda(CUresult st, const char* where) {
  if (st != CUDA_SUCCESS) fail_cuda(st, where);
}

void check_cublas(cublasStatus_t st, const char* where) {
  if (st != CUBLAS_STATUS_SUCCESS) fail_cublas(st, where);
}

void check_runtime(cudaError_t st, const char* where) {
  if (st != cudaSuccess) fail_runtime(st, where);
}

[[noreturn]] void fail_kernel_launch(const char* where) {
  fail_runtime(cudaPeekAtLastError(), where);
}

inline float* dptr(CUdeviceptr ptr) { return reinterpret_cast<float*>(static_cast<uintptr_t>(ptr)); }
inline const float* dptr_const(CUdeviceptr ptr) { return reinterpret_cast<const float*>(static_cast<uintptr_t>(ptr)); }
inline int* iptr(CUdeviceptr ptr) { return reinterpret_cast<int*>(static_cast<uintptr_t>(ptr)); }
inline std::uint16_t* u16ptr(CUdeviceptr ptr) { return reinterpret_cast<std::uint16_t*>(static_cast<uintptr_t>(ptr)); }
inline const std::uint16_t* u16ptr_const(CUdeviceptr ptr) {
  return reinterpret_cast<const std::uint16_t*>(static_cast<uintptr_t>(ptr));
}

extern CudaStats g_stats;

class CudaContext {
 public:
  CudaContext() {
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_cuda(cuDevicePrimaryCtxRetain(&ctx_, device_), "cuDevicePrimaryCtxRetain");
    activate();
    check_runtime(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cuStreamCreate(&stream_, CU_STREAM_DEFAULT), "cuStreamCreate");
    check_cublas(cublasCreate(&cublas_), "cublasCreate");
    check_cublas(cublasSetStream(cublas_, reinterpret_cast<cudaStream_t>(stream_)), "cublasSetStream");
    int cc_major = 0;
    check_cuda(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
               "cuDeviceGetAttribute(cc_major)");
    bf16_supported_ = cc_major >= 8;
  }

  ~CudaContext() {
    synchronize();
    for (auto& kv : weight_cache_) {
      if (kv.second.ptr != 0) cuMemFree(kv.second.ptr);
    }
    if (cublas_ != nullptr) cublasDestroy(cublas_);
    if (stream_ != nullptr) cuStreamDestroy(stream_);
    if (ctx_ != nullptr) {
      cuCtxSetCurrent(nullptr);
      cuDevicePrimaryCtxRelease(device_);
    }
  }

  CUstream stream() const { return stream_; }

  void activate() { check_cuda(cuCtxSetCurrent(ctx_), "cuCtxSetCurrent"); }

  void ensure_buffer(DeviceBuffer* buf, std::size_t bytes) {
    activate();
    if (buf->bytes >= bytes) return;
    synchronize();
    if (buf->ptr != 0) check_cuda(cuMemFree(buf->ptr), "cuMemFree(buffer)");
    buf->ptr = 0;
    buf->bytes = 0;
    check_cuda(cuMemAlloc(&buf->ptr, bytes), "cuMemAlloc(buffer)");
    buf->bytes = bytes;
  }

  void zero_buffer(DeviceBuffer* buf, std::size_t bytes) {
    activate();
    ensure_buffer(buf, bytes);
    check_cuda(cuMemsetD8Async(buf->ptr, 0, bytes, stream_), "cuMemsetD8Async");
  }

  void synchronize() {
    activate();
    if (stream_ != nullptr) check_cuda(cuStreamSynchronize(stream_), "cuStreamSynchronize");
  }

  bool preload_weight(const ds::hf::TensorSlice& weight) {
    activate();
    try {
      return cache_weight(weight) != nullptr;
    } catch (const std::runtime_error&) {
      return false;
    }
  }

  bool linear_device(const ds::hf::TensorSlice& weight, CUdeviceptr x, std::size_t in, CUdeviceptr y, std::size_t out) {
    activate();
    if (weight.shape.size() != 2) return false;
    if (static_cast<std::size_t>(weight.shape[0]) != out || static_cast<std::size_t>(weight.shape[1]) != in) return false;

    try {
      const CachedWeight* cached = cache_weight(weight);
      if (cached != nullptr) {
        gemv_row_major_mixed(*cached, out, in, x, y);
        return true;
      }

      ++g_stats.stream_linear_fallbacks;
      stream_linear(weight, x, in, y, out);
      return true;
    } catch (const std::runtime_error&) {
      throw;
    }
  }

  bool embedding_to_device(CUdeviceptr y, const ds::hf::TensorSlice& embedding, std::int32_t token_id, std::size_t hidden_size) {
    activate();
    if (embedding.shape.size() != 2) return false;
    if (token_id < 0 || token_id >= embedding.shape[0]) return false;
    if (static_cast<std::size_t>(embedding.shape[1]) != hidden_size) return false;
    const std::size_t row = static_cast<std::size_t>(token_id);

    try {
      const CachedWeight* cached = cache_weight(embedding);
      if (cached != nullptr) {
        switch (cached->kind) {
          case CachedWeightKind::RawF32:
          case CachedWeightKind::DecodedF32: {
            const std::size_t offset = row * hidden_size * sizeof(float);
            check_cuda(cuMemcpyDtoDAsync(y, cached->ptr + offset, hidden_size * sizeof(float), stream_),
                       "cuMemcpyDtoDAsync(embedding)");
            return true;
          }
          case CachedWeightKind::RawF16: {
            const auto src = u16ptr_const(cached->ptr) + row * hidden_size;
            if (!kernels::launch_f16_row_to_f32(reinterpret_cast<cudaStream_t>(stream_), src, static_cast<int>(hidden_size),
                                                dptr(y))) {
              fail_kernel_launch("launch_f16_row_to_f32");
            }
            return true;
          }
          case CachedWeightKind::RawBF16: {
            const auto src = u16ptr_const(cached->ptr) + row * hidden_size;
            if (!kernels::launch_bf16_row_to_f32(reinterpret_cast<cudaStream_t>(stream_), src,
                                                 static_cast<int>(hidden_size), dptr(y))) {
              fail_kernel_launch("launch_bf16_row_to_f32");
            }
            return true;
          }
        }
        throw std::runtime_error("embedding: unsupported cached weight kind");
      }

      switch (embedding.dtype) {
        case ds::hf::DType::F32: {
          auto decoded = ds::hf::decode_rows_to_f32(embedding, row, 1);
          check_cuda(cuMemcpyHtoD(y, decoded.data(), hidden_size * sizeof(float)), "cuMemcpyHtoD(embedding)");
          return true;
        }
        case ds::hf::DType::F16:
        case ds::hf::DType::BF16: {
          const std::size_t elem = ds::hf::dtype_nbytes(embedding.dtype);
          ensure_buffer(&scratch_weight_, hidden_size * elem);
          const auto* src = embedding.data + row * hidden_size * elem;
          check_cuda(cuMemcpyHtoDAsync(scratch_weight_.ptr, src, hidden_size * elem, stream_),
                     "cuMemcpyHtoDAsync(embedding row)");
          if (embedding.dtype == ds::hf::DType::F16) {
            if (!kernels::launch_f16_row_to_f32(reinterpret_cast<cudaStream_t>(stream_), u16ptr_const(scratch_weight_.ptr),
                                                static_cast<int>(hidden_size), dptr(y))) {
              fail_kernel_launch("launch_f16_row_to_f32");
            }
          } else {
            if (!kernels::launch_bf16_row_to_f32(reinterpret_cast<cudaStream_t>(stream_),
                                                 u16ptr_const(scratch_weight_.ptr), static_cast<int>(hidden_size), dptr(y))) {
              fail_kernel_launch("launch_bf16_row_to_f32");
            }
          }
          return true;
        }
        default:
          return false;
      }
    } catch (const std::runtime_error&) {
      return false;
    }
  }

  bool supports_bf16_gemm() const { return bf16_supported_; }

  bool rmsnorm_device(CUdeviceptr x, const NormWeights& norm, std::size_t n, float eps, CUdeviceptr y) {
    activate();
    if (!norm.valid()) return false;
    CUdeviceptr w_ptr = 0;
    const CachedWeight* cached = cache_weight(*norm.weight);
    if (cached != nullptr) {
      w_ptr = cached->ptr;
    } else {
      ensure_buffer(&scratch_w_, n * sizeof(float));
      check_cuda(cuMemcpyHtoDAsync(scratch_w_.ptr, norm.data(), n * sizeof(float), stream_), "cuMemcpyHtoDAsync(rmsnorm w)");
      w_ptr = scratch_w_.ptr;
    }
    if (!kernels::launch_rmsnorm(reinterpret_cast<cudaStream_t>(stream_), dptr_const(x), dptr_const(w_ptr),
                                 static_cast<int>(n), eps, dptr(y))) {
      fail_kernel_launch("launch_rmsnorm");
    }
    return true;
  }

  void add_inplace(CUdeviceptr dst, CUdeviceptr src, std::size_t n) {
    activate();
    if (!kernels::launch_add_inplace(reinterpret_cast<cudaStream_t>(stream_), dptr(dst), dptr_const(src),
                                     static_cast<int>(n))) {
      fail_kernel_launch("launch_add_inplace");
    }
  }

  void scale_add(CUdeviceptr dst, CUdeviceptr src, float scale, std::size_t n) {
    activate();
    if (!kernels::launch_scale_add(reinterpret_cast<cudaStream_t>(stream_), dptr(dst), dptr_const(src), scale,
                                   static_cast<int>(n))) {
      fail_kernel_launch("launch_scale_add");
    }
  }

  void silu_mul(CUdeviceptr gate, CUdeviceptr up, std::size_t n) {
    activate();
    if (!kernels::launch_silu_mul(reinterpret_cast<cudaStream_t>(stream_), dptr(gate), dptr_const(up),
                                  static_cast<int>(n))) {
      fail_kernel_launch("launch_silu_mul");
    }
  }

  void rope_inplace(CUdeviceptr x, std::size_t dim, std::size_t pos, float theta) {
    activate();
    if (dim == 0) return;
    if (!kernels::launch_rope_inplace(reinterpret_cast<cudaStream_t>(stream_), dptr(x), static_cast<int>(dim),
                                      static_cast<int>(pos), theta)) {
      fail_kernel_launch("launch_rope_inplace");
    }
  }

  void scores(CUdeviceptr q, CUdeviceptr k_base, std::size_t q_head, std::size_t seq_len, std::size_t key_stride,
              float scale, CUdeviceptr out_scores) {
    activate();
    if (!kernels::launch_scores(reinterpret_cast<cudaStream_t>(stream_), dptr_const(q), dptr_const(k_base),
                                static_cast<int>(q_head), static_cast<int>(seq_len), static_cast<int>(key_stride), scale,
                                dptr(out_scores))) {
      fail_kernel_launch("launch_scores");
    }
  }

  void softmax(CUdeviceptr x, std::size_t n) {
    activate();
    if (!kernels::launch_softmax(reinterpret_cast<cudaStream_t>(stream_), dptr(x), static_cast<int>(n))) {
      fail_kernel_launch("launch_softmax");
    }
  }

  void weighted_sum(CUdeviceptr scores_ptr, CUdeviceptr v_base, std::size_t seq_len, std::size_t v_head,
                    std::size_t v_stride, CUdeviceptr out) {
    activate();
    if (!kernels::launch_weighted_sum(reinterpret_cast<cudaStream_t>(stream_), dptr_const(scores_ptr), dptr_const(v_base),
                                      static_cast<int>(seq_len), static_cast<int>(v_head), static_cast<int>(v_stride),
                                      dptr(out))) {
      fail_kernel_launch("launch_weighted_sum");
    }
  }

  void select_topk(CUdeviceptr scores_ptr, std::size_t n, std::size_t top_k, bool normalize, float routed_scale,
                   DeviceBuffer* ids, DeviceBuffer* probs) {
    activate();
    ensure_buffer(ids, top_k * sizeof(std::int32_t));
    ensure_buffer(probs, top_k * sizeof(float));
    if (!kernels::launch_select_topk(reinterpret_cast<cudaStream_t>(stream_), dptr_const(scores_ptr), static_cast<int>(n),
                                     static_cast<int>(top_k), normalize, routed_scale, iptr(ids->ptr), dptr(probs->ptr))) {
      fail_kernel_launch("launch_select_topk");
    }
  }

  const CachedWeight* cache_weight(const ds::hf::TensorSlice& weight) {
    const CachedWeightKey key{
        .data = weight.data,
        .nbytes = weight.nbytes,
        .dtype = weight.dtype,
        .name = weight.name,
    };
    auto it = weight_cache_.find(key);
    if (it != weight_cache_.end()) {
      ++g_stats.cached_weight_hits;
      return &it->second;
    }

    CachedWeight cached;
    const void* host_data = weight.data;
    std::vector<float> decoded;
    switch (weight.dtype) {
      case ds::hf::DType::F32:
        cached.kind = CachedWeightKind::RawF32;
        cached.bytes = weight.nbytes;
        break;
      case ds::hf::DType::F16:
        cached.kind = CachedWeightKind::RawF16;
        cached.bytes = weight.nbytes;
        break;
      case ds::hf::DType::BF16:
        if (supports_bf16_gemm()) {
          cached.kind = CachedWeightKind::RawBF16;
          cached.bytes = weight.nbytes;
        } else {
          decoded = ds::hf::decode_to_f32(weight);
          cached.kind = CachedWeightKind::DecodedF32;
          cached.bytes = decoded.size() * sizeof(float);
          host_data = decoded.data();
        }
        break;
      default:
        throw std::runtime_error("cache_weight: unsupported dtype");
    }

    if (cached.kind != CachedWeightKind::DecodedF32) {
      host_data = weight.data;
    }

    try {
      check_cuda(cuMemAlloc(&cached.ptr, cached.bytes), "cuMemAlloc(weight)");
    } catch (const std::runtime_error&) {
      return nullptr;
    }
    check_cuda(cuMemcpyHtoDAsync(cached.ptr, host_data, cached.bytes, stream_), "cuMemcpyHtoDAsync(weight)");
    synchronize();
    auto [inserted_it, _] = weight_cache_.emplace(key, cached);
    ++g_stats.cached_weight_uploads;
    g_stats.cached_weight_bytes += cached.bytes;
    return &inserted_it->second;
  }

  void gemv_row_major_mixed(const CachedWeight& weight, std::size_t rows, std::size_t cols, CUdeviceptr d_x, CUdeviceptr d_y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    switch (weight.kind) {
      case CachedWeightKind::RawF32:
      case CachedWeightKind::DecodedF32:
        check_cublas(cublasSgemv(cublas_, CUBLAS_OP_T, static_cast<int>(cols), static_cast<int>(rows), &alpha,
                                 dptr_const(weight.ptr), static_cast<int>(cols), dptr_const(d_x), 1, &beta, dptr(d_y), 1),
                     "cublasSgemv");
        return;
      case CachedWeightKind::RawF16:
        ensure_buffer(&scratch_x_, cols * sizeof(std::uint16_t));
        if (!kernels::launch_f32_to_f16(reinterpret_cast<cudaStream_t>(stream_), dptr_const(d_x), static_cast<int>(cols),
                                        u16ptr(scratch_x_.ptr))) {
          fail_kernel_launch("launch_f32_to_f16");
        }
        check_cublas(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows), 1, static_cast<int>(cols),
                                  &alpha, reinterpret_cast<const __half*>(static_cast<uintptr_t>(weight.ptr)), CUDA_R_16F,
                                  static_cast<int>(cols), reinterpret_cast<const __half*>(static_cast<uintptr_t>(scratch_x_.ptr)),
                                  CUDA_R_16F, static_cast<int>(cols), &beta,
                                  dptr(d_y), CUDA_R_32F, static_cast<int>(rows), CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx(f16)");
        return;
      case CachedWeightKind::RawBF16:
        ensure_buffer(&scratch_x_, cols * sizeof(std::uint16_t));
        if (!kernels::launch_f32_to_bf16(reinterpret_cast<cudaStream_t>(stream_), dptr_const(d_x), static_cast<int>(cols),
                                         u16ptr(scratch_x_.ptr))) {
          fail_kernel_launch("launch_f32_to_bf16");
        }
        check_cublas(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows), 1, static_cast<int>(cols),
                                  &alpha, reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(weight.ptr)),
                                  CUDA_R_16BF, static_cast<int>(cols),
                                  reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(scratch_x_.ptr)),
                                  CUDA_R_16BF, static_cast<int>(cols), &beta, dptr(d_y), CUDA_R_32F,
                                  static_cast<int>(rows), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx(bf16)");
        return;
    }
  }

  void stream_linear(const ds::hf::TensorSlice& weight, CUdeviceptr d_x, std::size_t in, CUdeviceptr d_y, std::size_t out) {
    const std::size_t elem_bytes = ds::hf::dtype_nbytes(weight.dtype);
    if (elem_bytes == 0) throw std::runtime_error("stream_linear: unknown dtype");
    const std::size_t rows_per_chunk = std::max<std::size_t>(1, kStreamWeightChunkBytes / (in * elem_bytes));
    std::size_t row0 = 0;
    while (row0 < out) {
      const std::size_t rows = std::min(rows_per_chunk, out - row0);
      CachedWeight chunk;
      switch (weight.dtype) {
        case ds::hf::DType::F32: {
          auto decoded = ds::hf::decode_rows_to_f32(weight, row0, rows);
          chunk.kind = CachedWeightKind::RawF32;
          chunk.bytes = decoded.size() * sizeof(float);
          ensure_buffer(&scratch_weight_, chunk.bytes);
          check_cuda(cuMemcpyHtoD(scratch_weight_.ptr, decoded.data(), chunk.bytes), "cuMemcpyHtoD(weight chunk)");
          chunk.ptr = scratch_weight_.ptr;
          break;
        }
        case ds::hf::DType::F16: {
          chunk.kind = CachedWeightKind::RawF16;
          chunk.bytes = rows * in * elem_bytes;
          ensure_buffer(&scratch_weight_, chunk.bytes);
          const auto* src = weight.data + row0 * in * elem_bytes;
          check_cuda(cuMemcpyHtoD(scratch_weight_.ptr, src, chunk.bytes), "cuMemcpyHtoD(weight chunk)");
          chunk.ptr = scratch_weight_.ptr;
          break;
        }
        case ds::hf::DType::BF16: {
          if (supports_bf16_gemm()) {
            chunk.kind = CachedWeightKind::RawBF16;
            chunk.bytes = rows * in * elem_bytes;
            ensure_buffer(&scratch_weight_, chunk.bytes);
            const auto* src = weight.data + row0 * in * elem_bytes;
            check_cuda(cuMemcpyHtoD(scratch_weight_.ptr, src, chunk.bytes), "cuMemcpyHtoD(weight chunk)");
            chunk.ptr = scratch_weight_.ptr;
          } else {
            auto decoded = ds::hf::decode_rows_to_f32(weight, row0, rows);
            chunk.kind = CachedWeightKind::DecodedF32;
            chunk.bytes = decoded.size() * sizeof(float);
            ensure_buffer(&scratch_weight_, chunk.bytes);
            check_cuda(cuMemcpyHtoD(scratch_weight_.ptr, decoded.data(), chunk.bytes), "cuMemcpyHtoD(weight chunk)");
            chunk.ptr = scratch_weight_.ptr;
          }
          break;
        }
        default:
          throw std::runtime_error("stream_linear: unsupported dtype");
      }
      gemv_row_major_mixed(chunk, rows, in, d_x, d_y + row0 * sizeof(float));
      row0 += rows;
    }
  }

  CUdevice device_ = 0;
  CUcontext ctx_ = nullptr;
  CUstream stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  bool bf16_supported_ = false;

  DeviceBuffer scratch_w_;
  DeviceBuffer scratch_weight_;
  DeviceBuffer scratch_x_;
    std::unordered_map<CachedWeightKey, CachedWeight, CachedWeightKeyHash> weight_cache_;
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
    context().synchronize();
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

bool preload_tensor(const ds::hf::TensorSlice& weight) {
  try {
    ensure_initialized();
    return context().preload_weight(weight);
  } catch (const std::runtime_error&) {
    bump_fallback(FallbackReason::AllocFailed);
    return false;
  }
}

bool embedding_to_slot(CudaExecutorState* state, const ds::hf::TensorSlice& embedding, std::int32_t token_id,
                       DeviceBufferSlot slot, std::size_t hidden_size) {
  if (state == nullptr) return false;
  try {
    if (!context().embedding_to_device(slot_ptr(state, slot, hidden_size), embedding, token_id, hidden_size)) {
      ++g_stats.embedding_cuda_fallbacks;
      bump_fallback(FallbackReason::UnsupportedShape);
      return false;
    }
    ++g_stats.embedding_cuda_hits;
    return true;
  } catch (const std::runtime_error&) {
    ++g_stats.embedding_cuda_fallbacks;
    bump_fallback(FallbackReason::KernelError);
    return false;
  }
}

bool upload_to_slot(CudaExecutorState* state, DeviceBufferSlot slot, const float* src, std::size_t n) {
  if (state == nullptr) return false;
  try {
    const auto ptr = slot_ptr(state, slot, n);
    check_cuda(cuMemcpyHtoDAsync(ptr, src, n * sizeof(float), context().stream()), "cuMemcpyHtoDAsync(slot)");
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
    check_cuda(cuMemcpyDtoHAsync(dst, ptr, n * sizeof(float), context().stream()), "cuMemcpyDtoHAsync(slot)");
    context().synchronize();
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

bool rmsnorm_to_slot(CudaExecutorState* state, DeviceBufferSlot x, const NormWeights& norm, std::size_t n, float eps,
                     DeviceBufferSlot y) {
  if (state == nullptr) return false;
  try {
    return context().rmsnorm_device(slot_ptr(state, x, n), norm, n, eps, slot_ptr(state, y, n));
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
    check_cuda(cuMemcpyDtoHAsync(top_ids.data(), state->topk_ids().ptr, top_k * sizeof(std::int32_t), context().stream()),
               "cuMemcpyDtoHAsync(topk ids)");
    check_cuda(cuMemcpyDtoHAsync(top_probs.data(), state->topk_probs().ptr, top_k * sizeof(float), context().stream()),
               "cuMemcpyDtoHAsync(topk probs)");
    context().synchronize();

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
      if (!rmsnorm_to_slot(state, DeviceBufferSlot::Tmp0, attn.q_a_layernorm, q_rank, cfg.rms_norm_eps,
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
    check_cuda(cuMemcpyDtoDAsync(slot_ptr(state, DeviceBufferSlot::Tmp4, kv_rank),
                                 slot_ptr(state, DeviceBufferSlot::Tmp3, kv_rank + qk_rope), kv_rank * sizeof(float),
                                 context().stream()),
               "cuMemcpyDtoDAsync(kv compressed)");
    if (!rmsnorm_to_slot(state, DeviceBufferSlot::Tmp4, attn.kv_a_layernorm, kv_rank, cfg.rms_norm_eps,
                         DeviceBufferSlot::Tmp4)) {
      return false;
    }
    if (!linear_to_slot(state, *attn.kv_b_proj.weight, DeviceBufferSlot::Tmp4, kv_rank, DeviceBufferSlot::Tmp5,
                        n_heads * (qk_nope + v_head))) {
      return false;
    }

    if (qk_rope > 0) {
      check_cuda(cuMemcpyDtoDAsync(slot_ptr(state, DeviceBufferSlot::Tmp6, qk_rope),
                                   slot_ptr(state, DeviceBufferSlot::Tmp3, kv_rank + qk_rope) + kv_rank * sizeof(float),
                                   qk_rope * sizeof(float), context().stream()),
                 "cuMemcpyDtoDAsync(shared_k_rope)");
      context().rope_inplace(slot_ptr(state, DeviceBufferSlot::Tmp6, qk_rope), qk_rope, pos, cfg.rope_theta);
    }

    zero_slot(state, DeviceBufferSlot::Tmp7, n_heads * v_head);
    auto& cache = state->cache(layer_id);
    const std::size_t key_stride = cache.n_heads * cache.q_head_dim;
    const std::size_t val_stride = cache.n_heads * cache.v_head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(q_head));

    for (std::size_t head = 0; head < n_heads; ++head) {
      check_cuda(cuMemcpyDtoDAsync(slot_ptr(state, DeviceBufferSlot::Tmp0, q_head),
                                   slot_ptr(state, DeviceBufferSlot::Tmp2, n_heads * q_head) + head * q_head * sizeof(float),
                                   q_head * sizeof(float), context().stream()),
                 "cuMemcpyDtoDAsync(q head)");
      if (qk_rope > 0) {
        context().rope_inplace(slot_ptr(state, DeviceBufferSlot::Tmp0, q_head) + qk_nope * sizeof(float), qk_rope, pos, cfg.rope_theta);
      }

      const std::size_t key_offset = ((pos * n_heads + head) * q_head);
      const std::size_t val_offset = ((pos * n_heads + head) * v_head);
      const std::size_t kv_head_offset = head * (qk_nope + v_head);
      if (qk_nope > 0) {
        check_cuda(cuMemcpyDtoDAsync(cache.k.ptr + key_offset * sizeof(float),
                                     slot_ptr(state, DeviceBufferSlot::Tmp5, n_heads * (qk_nope + v_head)) +
                                         kv_head_offset * sizeof(float),
                                     qk_nope * sizeof(float), context().stream()),
                   "cuMemcpyDtoDAsync(cache key)");
      }
      if (qk_rope > 0) {
        check_cuda(cuMemcpyDtoDAsync(cache.k.ptr + (key_offset + qk_nope) * sizeof(float),
                                     slot_ptr(state, DeviceBufferSlot::Tmp6, qk_rope), qk_rope * sizeof(float),
                                     context().stream()),
                   "cuMemcpyDtoDAsync(cache rope key)");
      }
      check_cuda(cuMemcpyDtoDAsync(cache.v.ptr + val_offset * sizeof(float),
                                   slot_ptr(state, DeviceBufferSlot::Tmp5, n_heads * (qk_nope + v_head)) +
                                       (kv_head_offset + qk_nope) * sizeof(float),
                                   v_head * sizeof(float), context().stream()),
                 "cuMemcpyDtoDAsync(cache value)");

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

  ds::hf::TensorSlice weight;
  weight.name = "runtime_rmsnorm";
  weight.dtype = ds::hf::DType::F32;
  weight.shape = {static_cast<std::int64_t>(n)};
  weight.data = reinterpret_cast<const std::uint8_t*>(w);
  weight.nbytes = n * sizeof(float);
  weight.shard_path = "runtime";
  NormWeights norm;
  norm.weight = &weight;
  norm.f32.assign(w, w + n);

  if (!upload_to_slot(host_state, DeviceBufferSlot::Hidden, x, n)) return false;
  if (!rmsnorm_to_slot(host_state, DeviceBufferSlot::Hidden, norm, n, eps, DeviceBufferSlot::Delta)) return false;
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
bool preload_tensor(const ds::hf::TensorSlice&) { return false; }
bool embedding_to_slot(CudaExecutorState*, const ds::hf::TensorSlice&, std::int32_t, DeviceBufferSlot, std::size_t) { return false; }
bool rmsnorm_to_slot(CudaExecutorState*, DeviceBufferSlot, const NormWeights&, std::size_t, float, DeviceBufferSlot) {
  return false;
}
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
