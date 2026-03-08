#include "ds/util/mmap.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ds::util {

MMapFile::~MMapFile() { close(); }

MMapFile::MMapFile(MMapFile&& o) noexcept {
  *this = std::move(o);
}

MMapFile& MMapFile::operator=(MMapFile&& o) noexcept {
  if (this == &o) return *this;
  close();
  path_ = std::move(o.path_);
  fd_ = o.fd_;
  data_ = o.data_;
  size_ = o.size_;
  o.fd_ = -1;
  o.data_ = nullptr;
  o.size_ = 0;
  return *this;
}

void MMapFile::open_readonly(const std::string& path) {
#if !defined(__linux__)
  (void)path;
  throw std::runtime_error("mmap not supported on this platform build");
#else
  close();
  path_ = path;

  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error("open failed: " + path + ": " + std::strerror(errno));
  }

  struct stat st;
  if (::fstat(fd_, &st) != 0) {
    const auto e = std::strerror(errno);
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("fstat failed: " + path + ": " + e);
  }
  if (st.st_size <= 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("file is empty: " + path);
  }
  size_ = static_cast<std::size_t>(st.st_size);

  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) {
    const auto e = std::strerror(errno);
    ::close(fd_);
    fd_ = -1;
    size_ = 0;
    throw std::runtime_error("mmap failed: " + path + ": " + e);
  }
  data_ = static_cast<std::uint8_t*>(p);
#endif
}

void MMapFile::close() {
#if defined(__linux__)
  if (data_) {
    ::munmap(data_, size_);
    data_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
#endif
  size_ = 0;
  path_.clear();
}

} // namespace ds::util
