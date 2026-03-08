#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ds::util {

// Minimal JSON value for HF config + safetensors header.
// Arrays/objects store shared_ptr<Json> so the type can be recursive.
struct Json {
  struct Array {
    std::vector<std::shared_ptr<Json>> v;
  };
  struct Object {
    std::unordered_map<std::string, std::shared_ptr<Json>> v;
  };

  using array_ptr = std::shared_ptr<Array>;
  using object_ptr = std::shared_ptr<Object>;

  enum class Kind { Null, Bool, Int, Double, String, Array, Object };

  std::variant<std::monostate, bool, std::int64_t, double, std::string, array_ptr, object_ptr> v;

  Kind kind() const;
  bool is_null() const;
  bool is_bool() const;
  bool is_int() const;
  bool is_double() const;
  bool is_string() const;
  bool is_array() const;
  bool is_object() const;

  bool as_bool() const;
  std::int64_t as_int() const;
  double as_double() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  const Json* find(const std::string& key) const;
};

Json parse_json(const std::string& s);

std::string get_string_or(const Json& j, const std::string& key, const std::string& def);
std::int64_t get_int_or(const Json& j, const std::string& key, std::int64_t def);
double get_double_or(const Json& j, const std::string& key, double def);

} // namespace ds::util
