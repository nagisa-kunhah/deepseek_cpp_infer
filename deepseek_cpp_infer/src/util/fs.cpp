#include "ds/util/fs.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ds::util {

std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) throw std::runtime_error("failed to open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<std::uint8_t> read_binary_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) throw std::runtime_error("failed to open file: " + path);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<std::size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> buf(n);
  if (n > 0) f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  return buf;
}

} // namespace ds::util
