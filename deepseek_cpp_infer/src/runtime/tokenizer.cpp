#include "ds/runtime/tokenizer.h"

#include "ds/util/fs.h"
#include "ds/util/json.h"

#include <tokenizers_cpp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ds::rt {
class TokenizerImpl {
 public:
  virtual ~TokenizerImpl() = default;

  virtual bool can_encode_text() const = 0;
  virtual std::vector<std::int32_t> encode(const std::string& text) const = 0;
  virtual std::string decode(const std::vector<std::int32_t>& ids) const = 0;
};

namespace {

struct ParsedTokenizerMetadata {
  TokenizerMetadata metadata;
  std::unordered_set<std::int32_t> special_ids;
};

std::string trim_piece(std::string token) {
  if (token.rfind("##", 0) == 0) token.erase(0, 2);
  if (token.size() >= 4 && token.compare(token.size() - 4, 4, "</w>") == 0) {
    token.erase(token.size() - 4);
    token.push_back(' ');
  }
  if (token.rfind("\xE2\x96\x81", 0) == 0) {
    token.erase(0, 3);
    token.insert(token.begin(), ' ');
  } else if (token.rfind("\xC4\xA0", 0) == 0) {
    token.erase(0, 2);
    token.insert(token.begin(), ' ');
  }
  return token;
}

const ds::util::Json* find_object_field(const ds::util::Json& j, const std::string& key) {
  const auto* value = j.find(key);
  if (!value || !value->is_object()) return nullptr;
  return value;
}

std::int32_t maybe_get_token_id(const ds::util::Json& j, const char* key) {
  const auto* v = j.find(key);
  if (!v || v->is_null()) return -1;
  return static_cast<std::int32_t>(v->as_int());
}

std::int32_t find_vocab_token_id(const ds::util::Json& model, const std::string& token) {
  const auto* vocab = model.find("vocab");
  if (!vocab || !vocab->is_object()) return -1;
  const auto* id = vocab->find(token);
  if (!id || (!id->is_int() && !id->is_double())) return -1;
  return static_cast<std::int32_t>(id->as_int());
}

ParsedTokenizerMetadata parse_metadata(const ds::util::Json& root) {
  if (!root.is_object()) throw std::runtime_error("tokenizer.json root is not an object");

  ParsedTokenizerMetadata parsed;
  const auto* model = find_object_field(root, "model");
  if (!model) throw std::runtime_error("tokenizer.json missing model object");
  parsed.metadata.model_type = ds::util::get_string_or(*model, "type", "");
  parsed.metadata.unk_token_id = maybe_get_token_id(*model, "unk_token_id");
  if (parsed.metadata.unk_token_id < 0) {
    const auto unk_token = ds::util::get_string_or(*model, "unk_token", "");
    if (!unk_token.empty()) parsed.metadata.unk_token_id = find_vocab_token_id(*model, unk_token);
  }

  const auto* added_tokens = root.find("added_tokens");
  if (added_tokens && added_tokens->is_array()) {
    for (const auto& item_ptr : added_tokens->as_array().v) {
      const auto& item = *item_ptr;
      const auto id = maybe_get_token_id(item, "id");
      const bool special = ds::util::get_int_or(item, "special", 0) != 0;
      if (special && id >= 0) parsed.special_ids.insert(id);
    }
  }

  parsed.metadata.bos_token_id = maybe_get_token_id(root, "bos_token_id");
  parsed.metadata.eos_token_id = maybe_get_token_id(root, "eos_token_id");
  if (parsed.metadata.bos_token_id >= 0) parsed.special_ids.insert(parsed.metadata.bos_token_id);
  if (parsed.metadata.eos_token_id >= 0) parsed.special_ids.insert(parsed.metadata.eos_token_id);
  if (parsed.metadata.unk_token_id >= 0) parsed.special_ids.insert(parsed.metadata.unk_token_id);
  return parsed;
}

void append_json_escaped_string(const std::string& value, std::string* out) {
  out->push_back('"');
  for (unsigned char c : value) {
    switch (c) {
      case '"': out->append("\\\""); break;
      case '\\': out->append("\\\\"); break;
      case '\b': out->append("\\b"); break;
      case '\f': out->append("\\f"); break;
      case '\n': out->append("\\n"); break;
      case '\r': out->append("\\r"); break;
      case '\t': out->append("\\t"); break;
      default:
        if (c < 0x20) {
          static constexpr char kHex[] = "0123456789abcdef";
          out->append("\\u00");
          out->push_back(kHex[(c >> 4) & 0x0F]);
          out->push_back(kHex[c & 0x0F]);
        } else {
          out->push_back(static_cast<char>(c));
        }
        break;
    }
  }
  out->push_back('"');
}

void append_json_value(const ds::util::Json& value, std::string* out) {
  switch (value.kind()) {
    case ds::util::Json::Kind::Null:
      out->append("null");
      return;
    case ds::util::Json::Kind::Bool:
      out->append(value.as_bool() ? "true" : "false");
      return;
    case ds::util::Json::Kind::Int:
      out->append(std::to_string(value.as_int()));
      return;
    case ds::util::Json::Kind::Double:
      out->append(std::to_string(value.as_double()));
      return;
    case ds::util::Json::Kind::String:
      append_json_escaped_string(value.as_string(), out);
      return;
    case ds::util::Json::Kind::Array: {
      out->push_back('[');
      bool first = true;
      for (const auto& item : value.as_array().v) {
        if (!first) out->push_back(',');
        first = false;
        append_json_value(*item, out);
      }
      out->push_back(']');
      return;
    }
    case ds::util::Json::Kind::Object: {
      std::vector<std::string> keys;
      keys.reserve(value.as_object().v.size());
      for (const auto& kv : value.as_object().v) keys.push_back(kv.first);
      std::sort(keys.begin(), keys.end());

      out->push_back('{');
      bool first = true;
      for (const auto& key : keys) {
        if (!first) out->push_back(',');
        first = false;
        append_json_escaped_string(key, out);
        out->push_back(':');
        append_json_value(*value.as_object().v.at(key), out);
      }
      out->push_back('}');
      return;
    }
  }
  throw std::runtime_error("unsupported json kind while serializing tokenizer config");
}

std::string build_external_tokenizer_blob(const ds::util::Json& root) {
  if (!root.is_object()) throw std::runtime_error("tokenizer.json root is not an object");

  static constexpr std::array<const char*, 9> kRecognizedFields = {
      "version",
      "truncation",
      "padding",
      "added_tokens",
      "normalizer",
      "pre_tokenizer",
      "post_processor",
      "decoder",
      "model",
  };

  std::string out;
  out.push_back('{');
  bool first = true;
  for (const char* field : kRecognizedFields) {
    const ds::util::Json* value = root.find(field);
    const std::string field_name(field);
    if (value == nullptr) {
      if (field_name == "model") {
        throw std::runtime_error("tokenizer.json missing model object");
      }
      if (field_name == "added_tokens") {
        static const ds::util::Json kEmptyArray = [] {
          ds::util::Json j;
          j.v = std::make_shared<ds::util::Json::Array>();
          return j;
        }();
        value = &kEmptyArray;
      } else if (field_name == "version") {
        static const ds::util::Json kVersion = [] {
          ds::util::Json j;
          j.v = std::string("1.0");
          return j;
        }();
        value = &kVersion;
      } else {
        static const ds::util::Json kNull;
        value = &kNull;
      }
    }

    if (!first) out.push_back(',');
    first = false;
    append_json_escaped_string(field, &out);
    out.push_back(':');
    append_json_value(*value, &out);
  }
  out.push_back('}');
  return out;
}

class MinimalTokenizerImpl final : public TokenizerImpl {
 public:
  explicit MinimalTokenizerImpl(const ds::util::Json& root) {
    const auto* model = find_object_field(root, "model");
    if (!model) throw std::runtime_error("tokenizer.json missing model object");

    const auto* vocab = model->find("vocab");
    if (!vocab || !vocab->is_object()) throw std::runtime_error("tokenizer.json missing model.vocab");

    std::size_t max_id = 0;
    for (const auto& kv : vocab->as_object().v) {
      const auto id = static_cast<std::size_t>(kv.second->as_int());
      max_id = std::max(max_id, id);
    }

    id_to_token_.assign(max_id + 1, std::string());
    for (const auto& kv : vocab->as_object().v) {
      const auto id = static_cast<std::size_t>(kv.second->as_int());
      token_to_id_.emplace(kv.first, static_cast<std::int32_t>(id));
      id_to_token_[id] = kv.first;
    }

    const auto* added_tokens = root.find("added_tokens");
    if (added_tokens && added_tokens->is_array()) {
      for (const auto& item_ptr : added_tokens->as_array().v) {
        const auto& item = *item_ptr;
        const auto id = maybe_get_token_id(item, "id");
        const auto content = ds::util::get_string_or(item, "content", "");
        if (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size() && !content.empty()) {
          id_to_token_[static_cast<std::size_t>(id)] = content;
          token_to_id_[content] = id;
        }
      }
    }
  }

  bool can_encode_text() const override { return !token_to_id_.empty(); }

  std::vector<std::int32_t> encode(const std::string& text) const override {
    if (text.empty()) return {};
    if (!can_encode_text()) throw std::runtime_error("tokenizer has no vocabulary");

    std::vector<std::int32_t> ids;
    std::size_t pos = 0;
    while (pos < text.size()) {
      const bool at_word_start = (pos == 0) || std::isspace(static_cast<unsigned char>(text[pos - 1]));

      std::int32_t best_id = -1;
      std::size_t best_len = 0;
      for (std::size_t len = text.size() - pos; len > 0; --len) {
        std::int32_t candidate = -1;
        if (lookup_piece_with_variants(text.substr(pos, len), at_word_start, &candidate)) {
          best_id = candidate;
          best_len = len;
          break;
        }
      }

      if (best_id < 0) {
        if (std::isspace(static_cast<unsigned char>(text[pos]))) {
          do {
            ++pos;
          } while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])));
          continue;
        }

        std::int32_t candidate = -1;
        if (!lookup_piece_with_variants(text.substr(pos, 1), at_word_start, &candidate)) {
          throw std::runtime_error("tokenizer could not encode input near byte offset " + std::to_string(pos));
        }
        best_id = candidate;
        best_len = 1;
      }

      ids.push_back(best_id);
      pos += best_len;
    }

    return ids;
  }

  std::string decode(const std::vector<std::int32_t>& ids) const override {
    std::string out;
    for (auto id : ids) {
      if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
        throw std::runtime_error("token id out of range");
      }
      out += trim_piece(id_to_token_[static_cast<std::size_t>(id)]);
    }
    return out;
  }

 private:
  bool lookup_token(const std::string& piece, std::int32_t* id) const {
    auto it = token_to_id_.find(piece);
    if (it == token_to_id_.end()) return false;
    *id = it->second;
    return true;
  }

  bool lookup_piece_with_variants(const std::string& piece, bool at_word_start, std::int32_t* id) const {
    if (lookup_token(piece, id)) return true;
    if (at_word_start) {
      if (lookup_token(std::string("\xE2\x96\x81") + piece, id)) return true;
      if (lookup_token(std::string("\xC4\xA0") + piece, id)) return true;
    }
    if (lookup_token(piece + "</w>", id)) return true;
    return false;
  }

  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, std::int32_t> token_to_id_;
};

class ExternalTokenizerImpl final : public TokenizerImpl {
 public:
  explicit ExternalTokenizerImpl(std::string blob) : blob_(std::move(blob)), tokenizer_(tokenizers::Tokenizer::FromBlobJSON(blob_)) {
    if (tokenizer_ == nullptr) throw std::runtime_error("failed to create external tokenizer from tokenizer.json");
  }

  bool can_encode_text() const override { return tokenizer_ != nullptr; }

  std::vector<std::int32_t> encode(const std::string& text) const override {
    return tokenizer_->Encode(text);
  }

  std::string decode(const std::vector<std::int32_t>& ids) const override {
    return tokenizer_->Decode(ids);
  }

 private:
  std::string blob_;
  mutable std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

} // namespace

Tokenizer::Tokenizer() = default;

Tokenizer::Tokenizer(std::shared_ptr<TokenizerImpl> impl, TokenizerMetadata metadata,
                     std::unordered_set<std::int32_t> special_ids)
    : impl_(std::move(impl)), metadata_(std::move(metadata)), special_ids_(std::move(special_ids)) {}

Tokenizer Tokenizer::load_from_file(const std::string& path) {
  const auto text = ds::util::read_text_file(path);
  const auto root = ds::util::parse_json(text);
  auto parsed = parse_metadata(root);
  const auto external_blob = build_external_tokenizer_blob(root);
  return Tokenizer(std::make_shared<ExternalTokenizerImpl>(external_blob), std::move(parsed.metadata),
                   std::move(parsed.special_ids));
}

Tokenizer Tokenizer::load_minimal_from_file(const std::string& path) {
  const auto text = ds::util::read_text_file(path);
  const auto root = ds::util::parse_json(text);
  auto parsed = parse_metadata(root);
  return Tokenizer(std::make_shared<MinimalTokenizerImpl>(root), std::move(parsed.metadata), std::move(parsed.special_ids));
}

bool Tokenizer::can_encode_text() const {
  return impl_ != nullptr && impl_->can_encode_text();
}

std::vector<std::int32_t> Tokenizer::encode(const std::string& text) const {
  if (impl_ == nullptr) throw std::runtime_error("tokenizer is empty");
  return impl_->encode(text);
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids, bool skip_special_tokens) const {
  if (impl_ == nullptr) throw std::runtime_error("tokenizer is empty");

  if (!skip_special_tokens) return impl_->decode(ids);

  std::vector<std::int32_t> filtered;
  filtered.reserve(ids.size());
  for (auto id : ids) {
    if (special_ids_.find(id) == special_ids_.end()) filtered.push_back(id);
  }
  if (filtered.empty()) return {};
  return impl_->decode(filtered);
}

} // namespace ds::rt
