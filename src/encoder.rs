use std::collections::HashMap;

use crate::{PreTokenizer, Vocabulary, bytes_to_unicode};

/// Encodes text into token IDs using Byte Pair Encoding (BPE).
///
/// The encoder converts input text into a sequence of token IDs by:
/// 1. Splitting text on special tokens (if any)
/// 2. Pre-tokenizing regular text into words/chunks using GPT-2 pattern
/// 3. Converting each chunk to byte-level Unicode representation
/// 4. Applying learned merge rules to create larger tokens
/// 5. Looking up final tokens in the vocabulary to get IDs
///
/// # Performance
///
/// The encoder caches the byte-to-unicode mapping to avoid reconstructing it
/// on every encode operation, improving performance for repeated encodings.
///
/// # Examples
///
/// ```
/// use bpe_tokenizer_rs::{Encoder, PreTokenizer, Vocabulary, Trainer};
///
/// let trainer = Trainer::new(0);
/// let merges = trainer.train(&[""]);
/// let vocab = Vocabulary::new(vec![], merges.clone());
/// let pre_tokenizer = PreTokenizer::new();
/// let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);
///
/// let ids = encoder.encode("Hello");
/// assert_eq!(ids, vec![39, 68, 75, 75, 78]);
/// ```
pub struct Encoder {
    merge_rules: Vec<(String, String)>,
    pre_tokenizer: PreTokenizer,
    vocabulary: Vocabulary,
    special_tokens: Vec<String>,
    byte_encoder: HashMap<u8, char>,
}

impl Encoder {
    /// Creates a new encoder with the given merge rules, pre-tokenizer, vocabulary, and special tokens.
    ///
    /// # Arguments
    ///
    /// * `merge_rules` - BPE merge rules learned during training as (token1, token2) pairs
    /// * `pre_tokenizer` - Pre-tokenizer for splitting text into chunks
    /// * `vocabulary` - Vocabulary mapping tokens to IDs
    /// * `special_tokens` - List of special tokens to recognize (e.g., `<|endoftext|>`)
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::{Encoder, PreTokenizer, Vocabulary};
    ///
    /// let vocab = Vocabulary::new(vec![], vec![]);
    /// let pre_tokenizer = PreTokenizer::new();
    /// let encoder = Encoder::new(vec![], pre_tokenizer, vocab, vec![]);
    /// ```
    pub fn new(
        merge_rules: Vec<(String, String)>,
        pre_tokenizer: PreTokenizer,
        vocabulary: Vocabulary,
        special_tokens: Vec<String>,
    ) -> Self {
        let byte_encoder = bytes_to_unicode();
        Encoder {
            merge_rules,
            pre_tokenizer,
            vocabulary,
            special_tokens,
            byte_encoder,
        }
    }

    /// Encodes text into a sequence of token IDs.
    ///
    /// The encoding process:
    /// 1. Splits text on special tokens
    /// 2. For regular text: pre-tokenizes, converts to bytes, applies merges
    /// 3. For special tokens: directly maps to their IDs
    /// 4. Returns the concatenated sequence of IDs
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode
    ///
    /// # Returns
    ///
    /// A vector of token IDs representing the encoded text.
    ///
    /// # Panics
    ///
    /// Panics if a token is not found in the vocabulary, indicating a mismatch
    /// between the vocabulary and merge rules.
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::{Encoder, PreTokenizer, Vocabulary, Trainer};
    ///
    /// let trainer = Trainer::new(0);
    /// let merges = trainer.train(&[""]);
    /// let vocab = Vocabulary::new(vec![], merges.clone());
    /// let pre_tokenizer = PreTokenizer::new();
    /// let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);
    ///
    /// let ids = encoder.encode("AB");
    /// assert_eq!(ids, vec![32, 33]);
    /// ```
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let chunks = self.split_on_special_tokens(text);

        chunks
            .into_iter()
            .flat_map(|(chunk_text, is_special)| {
                if is_special {
                    vec![self.token_to_id(&chunk_text)]
                } else {
                    self.encode_regular_text(&chunk_text)
                }
            })
            .collect()
    }

    fn encode_regular_text(&self, text: &str) -> Vec<u32> {
        self.pre_tokenizer
            .pre_tokenize(text)
            .iter()
            .flat_map(|word| {
                let unicode_symbols: Vec<String> = word
                    .as_bytes()
                    .iter()
                    .map(|&byte| self.byte_encoder[&byte].to_string())
                    .collect();

                let merged_tokens = self.apply_merge_rules(unicode_symbols);

                merged_tokens
                    .into_iter()
                    .map(|token| self.token_to_id(&token))
            })
            .collect()
    }

    fn split_on_special_tokens(&self, text: &str) -> Vec<(String, bool)> {
        if self.special_tokens.is_empty() {
            return vec![(text.to_string(), false)];
        }

        let mut chunks = vec![(text.to_string(), false)];

        for special_token in &self.special_tokens {
            chunks = chunks
                .into_iter()
                .flat_map(|(chunk_text, is_special)| {
                    if is_special {
                        vec![(chunk_text, true)]
                    } else {
                        self.split_chunk_on_token(&chunk_text, special_token)
                    }
                })
                .collect();
        }

        chunks
    }

    fn split_chunk_on_token(&self, text: &str, special_token: &str) -> Vec<(String, bool)> {
        let parts: Vec<&str> = text.split(special_token).collect();
        let mut result = Vec::with_capacity(parts.len() * 2);

        for (i, part) in parts.iter().enumerate() {
            if !part.is_empty() {
                result.push((part.to_string(), false));
            }

            if i < parts.len() - 1 {
                result.push((special_token.to_string(), true));
            }
        }

        result
    }

    /// Returns a reference to the vocabulary used by this encoder.
    ///
    /// This is useful for decoding token IDs back to text.
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn apply_merge_rules(&self, mut symbols: Vec<String>) -> Vec<String> {
        while let Some((rule_idx, positions)) = self.find_best_pair(&symbols) {
            let (first, second) = &self.merge_rules[rule_idx];
            let merged = format!("{}{}", first, second);
            let mut new_symbols = Vec::with_capacity(symbols.len() - positions.len());
            let mut i = 0;

            while i < symbols.len() {
                if positions.contains(&i) {
                    new_symbols.push(merged.clone());
                    i += 2;
                } else {
                    new_symbols.push(std::mem::take(&mut symbols[i]));
                    i += 1;
                }
            }

            symbols = new_symbols;
        }

        symbols
    }

    fn find_best_pair(&self, symbols: &[String]) -> Option<(usize, Vec<usize>)> {
        for (rule_idx, (first, second)) in self.merge_rules.iter().enumerate() {
            let mut positions = Vec::new();
            let mut i = 0;

            while i < symbols.len().saturating_sub(1) {
                if symbols[i] == *first && symbols[i + 1] == *second {
                    positions.push(i);
                    i += 2;
                } else {
                    i += 1;
                }
            }

            if !positions.is_empty() {
                return Some((rule_idx, positions));
            }
        }

        None
    }

    fn token_to_id(&self, token: &str) -> u32 {
        self.vocabulary
            .token_to_id(token)
            .unwrap_or_else(|| panic!("Token '{}' not in vocabulary. This indicates vocabulary and merge rules are out of sync!", token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Trainer;

    #[test]
    fn encode_empty_text() {
        let trainer = Trainer::new(5);
        let merges = trainer.train(&["test"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("");

        assert_eq!(ids, vec![]);
    }

    #[test]
    fn encode_single_ascii_char() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("A");

        assert_eq!(ids, vec![32]);
    }

    #[test]
    fn encode_two_ascii_chars() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("AB");

        assert_eq!(ids, vec![32, 33]);
    }

    #[test]
    fn encode_with_punctuation_split() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("A,B");

        assert_eq!(ids, vec![32, 11, 33]);
    }

    #[test]
    fn encode_utf8_two_bytes() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("é");

        assert_eq!(ids, vec![127, 102]);
    }

    #[test]
    fn encode_with_leading_space() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode(" A");

        assert_eq!(ids, vec![220, 32]);
    }

    #[test]
    fn encode_applies_single_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["aa aa aa"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("aa");

        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 256);
    }

    #[test]
    fn encode_with_learned_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["ab ab ab"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("ab");

        assert_eq!(ids, vec![256]);
    }

    #[test]
    fn encode_japanese_characters() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("日");

        assert_eq!(ids, vec![162, 245, 98]);
    }

    #[test]
    fn encode_russian_text() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("Привет");

        assert_eq!(
            ids,
            vec![140, 253, 141, 222, 140, 116, 140, 110, 140, 113, 141, 224]
        );
    }

    #[test]
    fn encode_mixed_languages() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids_hello = encoder.encode("Hello");
        let ids_chinese = encoder.encode("世界");
        let ids_russian = encoder.encode("Привет");

        assert_eq!(ids_hello, vec![39, 68, 75, 75, 78]);
        assert_eq!(ids_chinese, vec![160, 116, 244, 163, 243, 234]);
        assert_eq!(
            ids_russian,
            vec![140, 253, 141, 222, 140, 116, 140, 110, 140, 113, 141, 224]
        );
    }

    #[test]
    fn encode_emoji() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("🦀");

        assert_eq!(ids, vec![172, 253, 99, 222]);
    }

    #[test]
    fn encode_russian_with_single_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["Привет Привет Привет"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("Привет");

        assert_eq!(
            ids,
            vec![140, 253, 141, 222, 140, 116, 140, 256, 113, 141, 224]
        );
    }

    #[test]
    fn encode_chinese_with_single_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["世界 世界 世界"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);

        let ids = encoder.encode("世界");

        assert_eq!(ids, vec![160, 256, 163, 243, 234]);
    }

    #[test]
    fn encode_special_token_at_start() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);

        let ids = encoder.encode("<|endoftext|>hello");

        assert_eq!(ids, vec![0, 72, 69, 76, 76, 79]);
    }

    #[test]
    fn encode_special_token_at_end() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);

        let ids = encoder.encode("hello<|endoftext|>");

        assert_eq!(ids, vec![72, 69, 76, 76, 79, 0]);
    }

    #[test]
    fn encode_special_token_in_middle() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);

        let ids = encoder.encode("hello<|endoftext|>world");

        assert_eq!(ids, vec![72, 69, 76, 76, 79, 0, 87, 79, 82, 76, 68]);
    }

    #[test]
    fn encode_multiple_special_tokens() {
        let special_tokens = vec!["<|endoftext|>".to_string(), "[PAD]".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);

        let ids = encoder.encode("<|endoftext|>hello[PAD]");

        assert_eq!(ids, vec![0, 73, 70, 77, 77, 80, 1]);
    }

    #[test]
    fn encode_adjacent_special_tokens() {
        let special_tokens = vec!["<|endoftext|>".to_string(), "[PAD]".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);

        let ids = encoder.encode("<|endoftext|>[PAD]");

        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn encode_with_special_tokens_defined_but_not_used() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);

        let ids = encoder.encode("hello world");

        assert_eq!(ids, vec![72, 69, 76, 76, 79, 221, 87, 79, 82, 76, 68]);
    }
}
