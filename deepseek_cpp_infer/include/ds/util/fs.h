#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ds::util {

std::string read_text_file(const std::string& path);
std::vector<std::uint8_t> read_binary_file(const std::string& path);

} // namespace ds::util
