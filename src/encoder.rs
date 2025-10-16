use crate::{PreTokenizer, Vocabulary, bytes_to_unicode};

pub struct Encoder {
    merge_rules: Vec<(String, String)>,
    pre_tokenizer: PreTokenizer,
    vocabulary: Vocabulary,
}

impl Encoder {
    pub fn new(
        merge_rules: Vec<(String, String)>,
        pre_tokenizer: PreTokenizer,
        vocabulary: Vocabulary,
    ) -> Self {
        Encoder {
            merge_rules,
            pre_tokenizer,
            vocabulary,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let byte_encoder = bytes_to_unicode();

        self.pre_tokenizer
            .pre_tokenize(text)
            .iter()
            .flat_map(|chunk| {
                let unicode_symbols: Vec<String> = chunk
                    .as_bytes()
                    .iter()
                    .map(|&byte| byte_encoder[&byte].to_string())
                    .collect();

                let merged_tokens = self.apply_merge_rules(unicode_symbols);

                merged_tokens
                    .into_iter()
                    .map(|token| self.token_to_id(&token))
            })
            .collect()
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
            let mut new_symbols = Vec::new();
            let mut i = 0;

            while i < symbols.len() {
                if positions.contains(&i) {
                    new_symbols.push(merged.clone());
                    i += 2;
                } else {
                    new_symbols.push(symbols[i].clone());
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
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("");

        assert_eq!(ids, vec![]);
    }

    #[test]
    fn encode_single_ascii_char() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("A");

        assert_eq!(ids, vec![65]);
    }

    #[test]
    fn encode_two_ascii_chars() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("AB");

        assert_eq!(ids, vec![65, 66]);
    }

    #[test]
    fn encode_with_punctuation_split() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("A,B");

        assert_eq!(ids, vec![65, 44, 66]);
    }

    #[test]
    fn encode_utf8_two_bytes() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("é");

        assert_eq!(ids, vec![0xc3, 0xa9]);
    }

    #[test]
    fn encode_with_leading_space() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode(" A");

        assert_eq!(ids, vec![32, 65]);
    }

    #[test]
    fn encode_applies_single_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["aa aa aa"]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("aa");

        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 256);
    }

    #[test]
    fn encode_with_learned_merge() {
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["ab ab ab"]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("ab");

        assert_eq!(ids, vec![256]);
    }

    #[test]
    fn encode_japanese_characters() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("日");

        assert_eq!(ids, vec![0xe6, 0x97, 0xa5]);
    }

    #[test]
    fn encode_russian_text() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("Привет");

        assert_eq!(
            ids,
            vec![
                0xd0, 0x9f, 0xd1, 0x80, 0xd0, 0xb8, 0xd0, 0xb2, 0xd0, 0xb5, 0xd1, 0x82
            ]
        );
    }

    #[test]
    fn encode_mixed_languages() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids_hello = encoder.encode("Hello");
        let ids_chinese = encoder.encode("世界");
        let ids_russian = encoder.encode("Привет");

        assert_eq!(ids_hello, vec![72, 101, 108, 108, 111]);
        assert_eq!(ids_chinese, vec![0xe4, 0xb8, 0x96, 0xe7, 0x95, 0x8c]);
        assert_eq!(
            ids_russian,
            vec![
                0xd0, 0x9f, 0xd1, 0x80, 0xd0, 0xb8, 0xd0, 0xb2, 0xd0, 0xb5, 0xd1, 0x82
            ]
        );
    }

    #[test]
    fn encode_emoji() {
        let trainer = Trainer::new(0);
        let merges = trainer.train(&[""]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("🦀");

        assert_eq!(ids, vec![0xf0, 0x9f, 0xa6, 0x80]);
    }

    #[test]
    fn encode_russian_with_single_merge() {
        // Merge: (159, 209) -> 256, where 159 and 209 are UTF-8 bytes from "Привет"
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["Привет Привет Привет"]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("Привет");

        assert_eq!(
            ids,
            vec![208, 256, 128, 208, 184, 208, 178, 208, 181, 209, 130]
        );
    }

    #[test]
    fn encode_chinese_with_single_merge() {
        // Merge: (151, 165) -> 256, where 151 and 165 are UTF-8 bytes from "世界"
        let trainer = Trainer::new(1);
        let merges = trainer.train(&["世界 世界 世界"]);
        let vocab = Vocabulary::new(merges.clone());
        let pre_tokenizer = PreTokenizer::new();
        let encoder = Encoder::new(merges, pre_tokenizer, vocab);

        let ids = encoder.encode("世界");

        assert_eq!(ids, vec![228, 184, 256, 149, 140]);
    }
}
