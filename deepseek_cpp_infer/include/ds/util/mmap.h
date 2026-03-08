#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ds::util {

// Minimal read-only mmap wrapper (Linux/WSL). Used to access multi-GB safetensors
// shards without reading them into RAM.
class MMapFile {
 public:
  MMapFile() = default;
  ~MMapFile();

  MMapFile(const MMapFile&) = delete;
  MMapFile& operator=(const MMapFile&) = delete;

  MMapFile(MMapFile&& o) noexcept;
  MMapFile& operator=(MMapFile&& o) noexcept;

  void open_readonly(const std::string& path);
  void close();

  const std::uint8_t* data() const { return data_; }
  std::size_t size() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  int fd_ = -1;
  std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
};

} // namespace ds::util
