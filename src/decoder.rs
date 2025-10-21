use crate::{Vocabulary, unicode_to_bytes};
use std::collections::HashMap;

/// Decodes token IDs back into text using the vocabulary.
///
/// The decoder performs the reverse of encoding:
/// 1. Maps each token ID to its token string using the vocabulary
/// 2. Converts the Unicode representation back to bytes
/// 3. Assembles bytes into a UTF-8 string
///
/// # Performance
///
/// The decoder caches the unicode-to-byte mapping to avoid reconstructing it
/// on every decode operation, improving performance for repeated decodings.
///
/// # Examples
///
/// ```
/// use bpe_tokenizer_rs::{Decoder, Vocabulary};
///
/// let vocab = Vocabulary::new(vec![], vec![]);
/// let decoder = Decoder::new(vocab);
///
/// let text = decoder.decode(&[32, 33, 34]);
/// assert_eq!(text, "ABC");
/// ```
pub struct Decoder {
    vocabulary: Vocabulary,
    unicode_to_byte: HashMap<char, u8>,
}

impl Decoder {
    /// Creates a new decoder with the given vocabulary.
    ///
    /// # Arguments
    ///
    /// * `vocabulary` - The vocabulary mapping token IDs to tokens
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::{Decoder, Vocabulary};
    ///
    /// let vocab = Vocabulary::new(vec![], vec![]);
    /// let decoder = Decoder::new(vocab);
    /// ```
    pub fn new(vocabulary: Vocabulary) -> Self {
        let unicode_to_byte = unicode_to_bytes();
        Decoder {
            vocabulary,
            unicode_to_byte,
        }
    }

    /// Decodes a sequence of token IDs back into text.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Slice of token IDs to decode
    ///
    /// # Returns
    ///
    /// The decoded text as a UTF-8 string.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - A token ID is not found in the vocabulary
    /// - The resulting bytes cannot be decoded as valid UTF-8
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::{Decoder, Vocabulary};
    ///
    /// let vocab = Vocabulary::new(vec![], vec![]);
    /// let decoder = Decoder::new(vocab);
    ///
    /// let text = decoder.decode(&[39, 68, 75, 75, 78]);
    /// assert_eq!(text, "Hello");
    /// ```
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let bytes: Vec<u8> = token_ids
            .iter()
            .flat_map(|&token_id| {
                let token = self.vocabulary.id_to_token(token_id).unwrap_or_else(|| {
                    panic!(
                        "Token ID '{}' not in vocabulary. This indicates vocabulary and merge rules are out of sync!",
                        token_id
                    )
                });
                token.chars().map(|ch| self.unicode_to_byte[&ch]).collect::<Vec<u8>>()
            })
            .collect();

        String::from_utf8(bytes).unwrap_or_else(|e| {
            panic!(
                "Failed to decode bytes to UTF-8: {}. This indicates a bug in the encoder or decoder!",
                e
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Encoder, PreTokenizer, Trainer};

    #[test]
    fn decode_empty_sequence() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[]);

        assert_eq!(text, "");
    }

    #[test]
    fn decode_single_ascii_char() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[32]);

        assert_eq!(text, "A");
    }

    #[test]
    fn decode_multiple_ascii_chars() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[32, 33, 34]);

        assert_eq!(text, "ABC");
    }

    #[test]
    fn decode_with_space() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[39, 68, 75, 75, 78, 220, 54, 78, 81, 75, 67]);

        assert_eq!(text, "Hello World");
    }

    #[test]
    fn decode_utf8_two_bytes() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[127, 102]);

        assert_eq!(text, "é");
    }

    #[test]
    fn decode_japanese_characters() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[162, 245, 98]);

        assert_eq!(text, "日");
    }

    #[test]
    fn decode_russian_text() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[140, 253, 141, 222, 140, 116, 140, 110, 140, 113, 141, 224]);

        assert_eq!(text, "Привет");
    }

    #[test]
    fn decode_chinese_text() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[160, 116, 244, 163, 243, 234]);

        assert_eq!(text, "世界");
    }

    #[test]
    fn decode_emoji() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[172, 253, 99, 222]);

        assert_eq!(text, "🦀");
    }

    #[test]
    fn decode_with_single_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["ab ab ab"]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[256]);

        assert_eq!(text, "ab");
    }

    #[test]
    fn decode_mixed_base_and_merged_tokens() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["ab ab ab"]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[66, 256, 67]);

        assert_eq!(text, "cabd");
    }

    #[test]
    fn decode_russian_with_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["Привет Привет Привет"]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[140, 253, 141, 222, 140, 116, 140, 256, 113, 141, 224]);

        assert_eq!(text, "Привет");
    }

    #[test]
    fn decode_chinese_with_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["世界 世界 世界"]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[160, 256, 163, 243, 234]);

        assert_eq!(text, "世界");
    }

    #[test]
    fn encode_decode_round_trip_ascii() {
        let trainer = Trainer::new(5);
        let merges = trainer.train(&["hello world hello world hello world"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "hello world";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_multilingual() {
        let trainer = Trainer::new(10);
        let merges = trainer.train(&["Hello мир 世界 Hello мир 世界 Hello мир 世界"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "Hello мир 世界";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_with_emoji() {
        let trainer = Trainer::new(3);
        let merges = trainer.train(&["🦀 Rust 🦀 Rust 🦀 Rust"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "🦀 Rust";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_with_punctuation() {
        let trainer = Trainer::new(5);
        let merges = trainer.train(&["Hello, world! How are you? Hello, world! How are you?"]);
        let vocab = Vocabulary::new(vec![], merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, vec![]);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "Hello, world! How are you?";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    #[should_panic(expected = "Token ID '9999' not in vocabulary")]
    fn decode_panics_on_invalid_token_id() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(vec![], merges);
        let decoder = Decoder::new(vocab);

        decoder.decode(&[9999]);
    }

    #[test]
    fn encode_decode_round_trip_special_token_at_start() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "<|endoftext|>hello world";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_special_token_at_end() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "hello world<|endoftext|>";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_special_token_in_middle() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "hello<|endoftext|>world";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_multiple_special_tokens() {
        let special_tokens = vec!["<|start|>".to_string(), "<|end|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "<|start|>hello world<|end|>";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_adjacent_special_tokens() {
        let special_tokens = vec!["<|start|>".to_string(), "<|end|>".to_string()];
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "<|start|><|end|>";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_special_tokens_with_merges() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let trainer = Trainer::new(5);
        let merges = trainer.train(&["hello world hello world hello world"]);
        let vocab = Vocabulary::new(special_tokens.clone(), merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab, special_tokens);
        let decoder = Decoder::new(encoder.vocabulary().clone());

        let original = "<|endoftext|>hello world";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }
}
