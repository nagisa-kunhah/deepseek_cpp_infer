#include "ds/hf/safetensors.h"

#include "ds/util/fs.h"
#include "ds/util/json.h"

#include <fstream>
#include <stdexcept>

namespace ds::hf {
namespace {

DType parse_dtype(const std::string& s) {
  if (s == "F16") return DType::F16;
  if (s == "BF16") return DType::BF16;
  if (s == "F32") return DType::F32;
  if (s == "I32") return DType::I32;
  if (s == "I64") return DType::I64;
  if (s == "U8") return DType::U8;
  return DType::UNKNOWN;
}

std::uint64_t read_u64_le(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= (static_cast<std::uint64_t>(p[i]) << (8 * i));
  return v;
}

} // namespace

std::size_t dtype_nbytes(DType t) {
  switch (t) {
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::F32: return 4;
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::U8: return 1;
    default: return 0;
  }
}

SafeTensorsHeader load_safetensors_header(const std::string& path) {
  SafeTensorsHeader h;
  h.path = path;

  // Read only the safetensors header; weights can be multi-GB.
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) throw std::runtime_error("safetensors: failed to open: " + path);

  std::uint8_t lenbuf[8];
  f.read(reinterpret_cast<char*>(lenbuf), 8);
  if (f.gcount() != 8) throw std::runtime_error("safetensors: file too small: " + path);

  const auto header_len = read_u64_le(lenbuf);
  h.header_len = header_len;
  h.data_offset = 8ull + header_len;

  std::string header_json;
  header_json.resize(static_cast<std::size_t>(header_len));
  f.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (f.gcount() != static_cast<std::streamsize>(header_len)) {
    throw std::runtime_error("safetensors: truncated header: " + path);
  }

  const auto j = ds::util::parse_json(header_json);
  if (!j.is_object()) throw std::runtime_error("safetensors: header is not an object: " + path);

  for (const auto& kv : j.as_object().v) {
    const auto& name = kv.first;
    if (name == "__metadata__") continue;

    const auto& tj = *kv.second;
    if (!tj.is_object()) throw std::runtime_error("safetensors: tensor entry not object: " + name);

    const auto* dtype_j = tj.find("dtype");
    const auto* shape_j = tj.find("shape");
    const auto* off_j = tj.find("data_offsets");
    if (!dtype_j || !shape_j || !off_j) throw std::runtime_error("safetensors: missing fields: " + name);

    const auto dtype = parse_dtype(dtype_j->as_string());

    std::vector<std::int64_t> shape;
    for (const auto& dimp : shape_j->as_array().v) shape.push_back(dimp->as_int());

    std::vector<std::uint64_t> offsets;
    for (const auto& vp : off_j->as_array().v) offsets.push_back(static_cast<std::uint64_t>(vp->as_int()));
    if (offsets.size() != 2) throw std::runtime_error("safetensors: bad data_offsets for " + name);

    TensorMeta tm;
    tm.dtype = dtype;
    tm.shape = std::move(shape);
    tm.data_start = offsets[0];
    tm.data_end = offsets[1];
    if (tm.data_end < tm.data_start) throw std::runtime_error("safetensors: bad offsets for " + name);

    h.tensors.emplace(name, std::move(tm));
  }

  return h;
}

SafeTensorsFile load_safetensors(const std::string& path) {
  SafeTensorsFile f;
  f.path = path;
  f.raw = ds::util::read_binary_file(path);
  if (f.raw.size() < 8) throw std::runtime_error("safetensors: file too small: " + path);

  const auto header_len = read_u64_le(f.raw.data());
  const auto header_off = 8ull;
  const auto header_end = header_off + header_len;
  if (header_end > f.raw.size()) throw std::runtime_error("safetensors: bad header length: " + path);

  const auto* header_ptr = reinterpret_cast<const char*>(f.raw.data() + header_off);
  const std::string header_json(header_ptr, header_ptr + header_len);
  const auto j = ds::util::parse_json(header_json);
  if (!j.is_object()) throw std::runtime_error("safetensors: header is not an object: " + path);

  const std::uint8_t* data_base = f.raw.data() + header_end;
  const std::size_t data_size = f.raw.size() - header_end;

  for (const auto& kv : j.as_object().v) {
    const auto& name = kv.first;
    if (name == "__metadata__") continue;

    const auto& tj = *kv.second;
    if (!tj.is_object()) throw std::runtime_error("safetensors: tensor entry not object: " + name);

    const auto* dtype_j = tj.find("dtype");
    const auto* shape_j = tj.find("shape");
    const auto* off_j = tj.find("data_offsets");
    if (!dtype_j || !shape_j || !off_j) throw std::runtime_error("safetensors: missing fields: " + name);

    const auto dtype_s = dtype_j->as_string();
    const auto dtype = parse_dtype(dtype_s);

    std::vector<std::int64_t> shape;
    for (const auto& dimp : shape_j->as_array().v) shape.push_back(dimp->as_int());

    std::vector<std::uint64_t> offsets;
    for (const auto& vp : off_j->as_array().v) offsets.push_back(static_cast<std::uint64_t>(vp->as_int()));
    if (offsets.size() != 2) throw std::runtime_error("safetensors: bad data_offsets for " + name);

    const auto start = offsets[0];
    const auto end = offsets[1];
    if (end < start) throw std::runtime_error("safetensors: bad offsets for " + name);
    if (end > data_size) throw std::runtime_error("safetensors: tensor out of range for " + name);

    TensorView tv;
    tv.dtype = dtype;
    tv.shape = std::move(shape);
    tv.data = data_base + start;
    tv.nbytes = static_cast<std::size_t>(end - start);

    f.tensors.emplace(name, tv);
  }

  return f;
}

} // namespace ds::hf
