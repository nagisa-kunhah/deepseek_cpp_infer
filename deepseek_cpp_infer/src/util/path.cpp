#include "ds/util/path.h"

#include <filesystem>
#include <system_error>

namespace ds::util {

bool file_exists(const std::string& path) {
  std::error_code ec;
  return std::filesystem::exists(std::filesystem::u8path(path), ec);
}

std::vector<std::string> list_files_with_suffix(const std::string& dir, const std::string& suffix) {
  std::vector<std::string> out;
  std::error_code ec;
  const auto p = std::filesystem::u8path(dir);
  if (!std::filesystem::exists(p, ec)) return out;

  for (const auto& ent : std::filesystem::directory_iterator(p, ec)) {
    if (ec) break;
    if (!ent.is_regular_file()) continue;
    const auto name = ent.path().filename().u8string();
    if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      out.push_back(ent.path().u8string());
    }
  }
  return out;
}

} // namespace ds::util
