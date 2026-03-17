#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace ds::rt {

class TokenizerImpl;

struct TokenizerMetadata {
  std::string model_type;
  std::int32_t bos_token_id = -1;
  std::int32_t eos_token_id = -1;
  std::int32_t unk_token_id = -1;
};

class Tokenizer {
 public:
  Tokenizer();

  static Tokenizer load_from_file(const std::string& path);
  static Tokenizer load_minimal_from_file(const std::string& path);

  bool empty() const { return impl_ == nullptr; }
  bool can_encode_text() const;

  const TokenizerMetadata& metadata() const { return metadata_; }

  std::vector<std::int32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<std::int32_t>& ids, bool skip_special_tokens = true) const;

 private:
  Tokenizer(std::shared_ptr<TokenizerImpl> impl, TokenizerMetadata metadata,
            std::unordered_set<std::int32_t> special_ids);

  std::shared_ptr<TokenizerImpl> impl_;
  TokenizerMetadata metadata_;
  std::unordered_set<std::int32_t> special_ids_;
};

} // namespace ds::rt
