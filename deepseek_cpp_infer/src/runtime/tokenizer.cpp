#include "ds/runtime/tokenizer.h"

#include "ds/util/fs.h"
#include "ds/util/json.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace ds::rt {
namespace {

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

} // namespace

Tokenizer Tokenizer::load_from_file(const std::string& path) {
  const auto text = ds::util::read_text_file(path);
  const auto root = ds::util::parse_json(text);
  if (!root.is_object()) throw std::runtime_error("tokenizer.json root is not an object");

  Tokenizer tok;

  const auto* model = find_object_field(root, "model");
  if (!model) throw std::runtime_error("tokenizer.json missing model object");
  tok.metadata_.model_type = ds::util::get_string_or(*model, "type", "");

  const auto* vocab = model->find("vocab");
  if (!vocab || !vocab->is_object()) throw std::runtime_error("tokenizer.json missing model.vocab");

  std::size_t max_id = 0;
  for (const auto& kv : vocab->as_object().v) {
    const auto id = static_cast<std::size_t>(kv.second->as_int());
    max_id = std::max(max_id, id);
  }

  tok.id_to_token_.assign(max_id + 1, std::string());
  for (const auto& kv : vocab->as_object().v) {
    const auto id = static_cast<std::size_t>(kv.second->as_int());
    tok.token_to_id_.emplace(kv.first, static_cast<std::int32_t>(id));
    tok.id_to_token_[id] = kv.first;
  }

  tok.metadata_.unk_token_id = maybe_get_token_id(*model, "unk_token_id");

  const auto* added_tokens = root.find("added_tokens");
  if (added_tokens && added_tokens->is_array()) {
    for (const auto& item_ptr : added_tokens->as_array().v) {
      const auto& item = *item_ptr;
      const auto id = maybe_get_token_id(item, "id");
      const auto content = ds::util::get_string_or(item, "content", "");
      const bool special = ds::util::get_int_or(item, "special", 0) != 0;
      if (id >= 0 && static_cast<std::size_t>(id) < tok.id_to_token_.size() && !content.empty()) {
        tok.id_to_token_[static_cast<std::size_t>(id)] = content;
        tok.token_to_id_[content] = id;
      }
      if (special && id >= 0) tok.special_ids_.insert(id);
    }
  }

  tok.metadata_.bos_token_id = maybe_get_token_id(root, "bos_token_id");
  tok.metadata_.eos_token_id = maybe_get_token_id(root, "eos_token_id");
  if (tok.metadata_.bos_token_id >= 0) tok.special_ids_.insert(tok.metadata_.bos_token_id);
  if (tok.metadata_.eos_token_id >= 0) tok.special_ids_.insert(tok.metadata_.eos_token_id);

  return tok;
}

bool Tokenizer::lookup_token(const std::string& piece, std::int32_t* id) const {
  auto it = token_to_id_.find(piece);
  if (it == token_to_id_.end()) return false;
  *id = it->second;
  return true;
}

bool Tokenizer::lookup_piece_with_variants(const std::string& piece, bool at_word_start, std::int32_t* id) const {
  if (lookup_token(piece, id)) return true;
  if (at_word_start) {
    if (lookup_token(std::string("\xE2\x96\x81") + piece, id)) return true;
    if (lookup_token(std::string("\xC4\xA0") + piece, id)) return true;
  }
  if (lookup_token(piece + "</w>", id)) return true;
  return false;
}

std::vector<std::int32_t> Tokenizer::encode(const std::string& text) const {
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

std::string Tokenizer::decode_token(std::int32_t id, bool skip_special_tokens) const {
  if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
    throw std::runtime_error("token id out of range");
  }
  if (skip_special_tokens && special_ids_.find(id) != special_ids_.end()) return {};
  return trim_piece(id_to_token_[static_cast<std::size_t>(id)]);
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids, bool skip_special_tokens) const {
  std::string out;
  for (auto id : ids) out += decode_token(id, skip_special_tokens);
  return out;
}

} // namespace ds::rt
