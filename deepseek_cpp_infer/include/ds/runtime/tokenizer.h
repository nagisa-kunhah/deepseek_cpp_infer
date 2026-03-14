#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ds::rt {

struct TokenizerMetadata {
  std::string model_type;
  std::int32_t bos_token_id = -1;
  std::int32_t eos_token_id = -1;
  std::int32_t unk_token_id = -1;
};

class Tokenizer {
 public:
  static Tokenizer load_from_file(const std::string& path);

  bool empty() const { return id_to_token_.empty(); }
  bool can_encode_text() const { return !token_to_id_.empty(); }

  const TokenizerMetadata& metadata() const { return metadata_; }

  std::vector<std::int32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<std::int32_t>& ids, bool skip_special_tokens = true) const;

 private:
  bool lookup_token(const std::string& piece, std::int32_t* id) const;
  bool lookup_piece_with_variants(const std::string& piece, bool at_word_start, std::int32_t* id) const;
  std::string decode_token(std::int32_t id, bool skip_special_tokens) const;

  TokenizerMetadata metadata_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, std::int32_t> token_to_id_;
  std::unordered_set<std::int32_t> special_ids_;
};

} // namespace ds::rt
