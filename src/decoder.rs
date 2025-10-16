use crate::{Vocabulary, unicode_to_bytes};

pub struct Decoder {
    vocabulary: Vocabulary,
}

impl Decoder {
    pub fn new(vocabulary: Vocabulary) -> Self {
        Decoder { vocabulary }
    }

    /// Decodes a sequence of token IDs back into the original text.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let unicode_to_byte = unicode_to_bytes();

        let bytes: Vec<u8> = token_ids
            .iter()
            .flat_map(|&token_id| {
                let token = self.vocabulary.id_to_token(token_id).unwrap_or_else(|| {
                    panic!(
                        "Token ID '{}' not in vocabulary. This indicates vocabulary and merge rules are out of sync!",
                        token_id
                    )
                });
                token.chars().map(|ch| unicode_to_byte[&ch]).collect::<Vec<u8>>()
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
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[]);

        assert_eq!(text, "");
    }

    #[test]
    fn decode_single_ascii_char() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[65]);

        assert_eq!(text, "A");
    }

    #[test]
    fn decode_multiple_ascii_chars() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[65, 66, 67]);

        assert_eq!(text, "ABC");
    }

    #[test]
    fn decode_with_space() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]);

        assert_eq!(text, "Hello World");
    }

    #[test]
    fn decode_utf8_two_bytes() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[0xc3, 0xa9]);

        assert_eq!(text, "é");
    }

    #[test]
    fn decode_japanese_characters() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[0xe6, 0x97, 0xa5]);

        assert_eq!(text, "日");
    }

    #[test]
    fn decode_russian_text() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[
            0xd0, 0x9f, 0xd1, 0x80, 0xd0, 0xb8, 0xd0, 0xb2, 0xd0, 0xb5, 0xd1, 0x82,
        ]);

        assert_eq!(text, "Привет");
    }

    #[test]
    fn decode_chinese_text() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[0xe4, 0xb8, 0x96, 0xe7, 0x95, 0x8c]);

        assert_eq!(text, "世界");
    }

    #[test]
    fn decode_emoji() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[0xf0, 0x9f, 0xa6, 0x80]);

        assert_eq!(text, "🦀");
    }

    #[test]
    fn decode_with_single_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["ab ab ab"]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[256]);

        assert_eq!(text, "ab");
    }

    #[test]
    fn decode_mixed_base_and_merged_tokens() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["ab ab ab"]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[99, 256, 100]);

        assert_eq!(text, "cabd");
    }

    #[test]
    fn decode_russian_with_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["Привет Привет Привет"]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[208, 256, 128, 208, 184, 208, 178, 208, 181, 209, 130]);

        assert_eq!(text, "Привет");
    }

    #[test]
    fn decode_chinese_with_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["世界 世界 世界"]);
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        let text = decoder.decode(&[228, 184, 256, 149, 140]);

        assert_eq!(text, "世界");
    }

    #[test]
    fn encode_decode_round_trip_ascii() {
        let trainer = Trainer::new(5);
        let merges = trainer.train(&["hello world hello world hello world"]);
        let vocab = Vocabulary::new(merges.clone());
        let vocab2 = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);
        let decoder = Decoder::new(vocab2);

        let original = "hello world";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_multilingual() {
        let trainer = Trainer::new(10);
        let merges = trainer.train(&["Hello мир 世界 Hello мир 世界 Hello мир 世界"]);
        let vocab = Vocabulary::new(merges.clone());
        let vocab2 = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);
        let decoder = Decoder::new(vocab2);

        let original = "Hello мир 世界";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_with_emoji() {
        let trainer = Trainer::new(3);
        let merges = trainer.train(&["🦀 Rust 🦀 Rust 🦀 Rust"]);
        let vocab = Vocabulary::new(merges.clone());
        let vocab2 = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);
        let decoder = Decoder::new(vocab2);

        let original = "🦀 Rust";
        let ids = encoder.encode(original);
        let decoded = decoder.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_decode_round_trip_with_punctuation() {
        let trainer = Trainer::new(5);
        let merges = trainer.train(&["Hello, world! How are you? Hello, world! How are you?"]);
        let vocab = Vocabulary::new(merges.clone());
        let vocab2 = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);
        let decoder = Decoder::new(vocab2);

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
        let vocab = Vocabulary::new(merges);
        let decoder = Decoder::new(vocab);

        decoder.decode(&[9999]);
    }
}
