#include "ds/util/json.h"

#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace ds::util {

Json::Kind Json::kind() const {
  if (std::holds_alternative<std::monostate>(v)) return Kind::Null;
  if (std::holds_alternative<bool>(v)) return Kind::Bool;
  if (std::holds_alternative<std::int64_t>(v)) return Kind::Int;
  if (std::holds_alternative<double>(v)) return Kind::Double;
  if (std::holds_alternative<std::string>(v)) return Kind::String;
  if (std::holds_alternative<array_ptr>(v)) return Kind::Array;
  return Kind::Object;
}

bool Json::is_null() const { return std::holds_alternative<std::monostate>(v); }
bool Json::is_bool() const { return std::holds_alternative<bool>(v); }
bool Json::is_int() const { return std::holds_alternative<std::int64_t>(v); }
bool Json::is_double() const { return std::holds_alternative<double>(v); }
bool Json::is_string() const { return std::holds_alternative<std::string>(v); }
bool Json::is_array() const { return std::holds_alternative<array_ptr>(v); }
bool Json::is_object() const { return std::holds_alternative<object_ptr>(v); }

bool Json::as_bool() const {
  if (!is_bool()) throw std::runtime_error("json: not a bool");
  return std::get<bool>(v);
}

std::int64_t Json::as_int() const {
  if (is_int()) return std::get<std::int64_t>(v);
  if (is_double()) return static_cast<std::int64_t>(std::get<double>(v));
  throw std::runtime_error("json: not a number");
}

double Json::as_double() const {
  if (is_double()) return std::get<double>(v);
  if (is_int()) return static_cast<double>(std::get<std::int64_t>(v));
  throw std::runtime_error("json: not a number");
}

const std::string& Json::as_string() const {
  if (!is_string()) throw std::runtime_error("json: not a string");
  return std::get<std::string>(v);
}

const Json::Array& Json::as_array() const {
  if (!is_array()) throw std::runtime_error("json: not an array");
  return *std::get<array_ptr>(v);
}

const Json::Object& Json::as_object() const {
  if (!is_object()) throw std::runtime_error("json: not an object");
  return *std::get<object_ptr>(v);
}

const Json* Json::find(const std::string& key) const {
  if (!is_object()) return nullptr;
  const auto& o = std::get<object_ptr>(v)->v;
  auto it = o.find(key);
  if (it == o.end()) return nullptr;
  return it->second.get();
}

namespace {

struct Parser {
  const char* p;
  const char* end;

  void skip_ws() {
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;
  }

  [[noreturn]] void fail(const char* msg) const {
    throw std::runtime_error(std::string("json parse error: ") + msg);
  }

  bool consume(char c) {
    skip_ws();
    if (p < end && *p == c) {
      ++p;
      return true;
    }
    return false;
  }

  void expect(char c) {
    if (!consume(c)) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "expected '%c'", c);
      fail(buf);
    }
  }

  Json parse_value() {
    skip_ws();
    if (p >= end) fail("unexpected end");

    switch (*p) {
      case 'n': return parse_null();
      case 't': return parse_true();
      case 'f': return parse_false();
      case '"': {
        Json j;
        j.v = parse_string();
        return j;
      }
      case '[': return parse_array();
      case '{': return parse_object();
      default:
        if (*p == '-' || std::isdigit(static_cast<unsigned char>(*p))) return parse_number();
        fail("unexpected token");
    }
  }

  Json parse_null() {
    if (end - p >= 4 && std::memcmp(p, "null", 4) == 0) {
      p += 4;
      return Json{};
    }
    fail("bad null");
  }

  Json parse_true() {
    if (end - p >= 4 && std::memcmp(p, "true", 4) == 0) {
      p += 4;
      Json j;
      j.v = true;
      return j;
    }
    fail("bad true");
  }

  Json parse_false() {
    if (end - p >= 5 && std::memcmp(p, "false", 5) == 0) {
      p += 5;
      Json j;
      j.v = false;
      return j;
    }
    fail("bad false");
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (p < end) {
      char c = *p++;
      if (c == '"') return out;
      if (c == '\\') {
        if (p >= end) fail("bad escape");
        char e = *p++;
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            if (end - p < 4) fail("bad unicode escape");
            unsigned code = 0;
            for (int i = 0; i < 4; ++i) {
              char h = p[i];
              code <<= 4;
              if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
              else fail("bad hex");
            }
            p += 4;
            if (code <= 0x7F) {
              out.push_back(static_cast<char>(code));
            } else if (code <= 0x7FF) {
              out.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
              out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
              out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            break;
          }
          default: fail("unsupported escape");
        }
      } else {
        out.push_back(c);
      }
    }
    fail("unterminated string");
  }

  Json parse_number() {
    skip_ws();
    const char* start = p;
    if (*p == '-') ++p;
    while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
    bool is_float = false;
    if (p < end && *p == '.') {
      is_float = true;
      ++p;
      while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
      is_float = true;
      ++p;
      if (p < end && (*p == '+' || *p == '-')) ++p;
      while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
    }

    const std::string_view sv(start, static_cast<std::size_t>(p - start));
    Json j;
    if (!is_float) {
      std::int64_t v = 0;
      auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
      if (ec != std::errc{} || ptr != sv.data() + sv.size()) fail("bad int");
      j.v = v;
      return j;
    }

    std::string tmp(sv);
    char* ep = nullptr;
    const double d = std::strtod(tmp.c_str(), &ep);
    if (!ep || *ep != '\0') fail("bad float");
    j.v = d;
    return j;
  }

  Json parse_array() {
    expect('[');
    auto a = std::make_shared<Json::Array>();
    skip_ws();
    if (consume(']')) {
      Json j;
      j.v = a;
      return j;
    }
    while (true) {
      a->v.push_back(std::make_shared<Json>(parse_value()));
      skip_ws();
      if (consume(']')) break;
      expect(',');
    }
    Json j;
    j.v = a;
    return j;
  }

  Json parse_object() {
    expect('{');
    auto o = std::make_shared<Json::Object>();
    skip_ws();
    if (consume('}')) {
      Json j;
      j.v = o;
      return j;
    }
    while (true) {
      skip_ws();
      if (p >= end || *p != '"') fail("object key must be string");
      auto key = parse_string();
      skip_ws();
      expect(':');
      auto val = parse_value();
      o->v.emplace(std::move(key), std::make_shared<Json>(std::move(val)));
      skip_ws();
      if (consume('}')) break;
      expect(',');
    }
    Json j;
    j.v = o;
    return j;
  }
};

} // namespace

Json parse_json(const std::string& s) {
  Parser ps{.p = s.data(), .end = s.data() + s.size()};
  auto v = ps.parse_value();
  ps.skip_ws();
  if (ps.p != ps.end) throw std::runtime_error("json parse error: trailing garbage");
  return v;
}

std::string get_string_or(const Json& j, const std::string& key, const std::string& def) {
  const auto* v = j.find(key);
  if (!v || !v->is_string()) return def;
  return v->as_string();
}

std::int64_t get_int_or(const Json& j, const std::string& key, std::int64_t def) {
  const auto* v = j.find(key);
  if (!v) return def;
  if (!(v->is_int() || v->is_double())) return def;
  return v->as_int();
}

double get_double_or(const Json& j, const std::string& key, double def) {
  const auto* v = j.find(key);
  if (!v) return def;
  if (!(v->is_int() || v->is_double())) return def;
  return v->as_double();
}

} // namespace ds::util
