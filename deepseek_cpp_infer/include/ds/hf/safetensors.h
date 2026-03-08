#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ds::hf {

enum class DType {
  F16,
  BF16,
  F32,
  I32,
  I64,
  U8,
  UNKNOWN,
};

struct TensorView {
  DType dtype = DType::UNKNOWN;
  std::vector<std::int64_t> shape;
  const std::uint8_t* data = nullptr;
  std::size_t nbytes = 0;
};

// Header-only metadata, safe to load for multi-GB shards.
struct TensorMeta {
  DType dtype = DType::UNKNOWN;
  std::vector<std::int64_t> shape;
  std::uint64_t data_start = 0; // offset into the data section
  std::uint64_t data_end = 0;   // offset into the data section
};

struct SafeTensorsHeader {
  std::string path;
  std::uint64_t header_len = 0;
  std::uint64_t data_offset = 0; // absolute file offset where tensor data begins
  std::unordered_map<std::string, TensorMeta> tensors;
};

std::size_t dtype_nbytes(DType t);

struct SafeTensorsFile {
  std::string path;
  std::vector<std::uint8_t> raw;  // simple loader (no mmap yet)
  std::unordered_map<std::string, TensorView> tensors;
};

SafeTensorsFile load_safetensors(const std::string& path);
SafeTensorsHeader load_safetensors_header(const std::string& path);

} // namespace ds::hf
