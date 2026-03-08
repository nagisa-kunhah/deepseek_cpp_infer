#pragma once

#include <string>
#include <vector>

namespace ds::util {

bool file_exists(const std::string& path);
std::vector<std::string> list_files_with_suffix(const std::string& dir, const std::string& suffix);

} // namespace ds::util
