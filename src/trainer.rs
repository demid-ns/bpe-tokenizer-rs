use crate::{PreTokenizer, bytes_to_unicode};
use std::collections::HashMap;

/// Trains a BPE tokenizer by learning merge rules from training data.
///
/// Uses byte-level tokenization with pre-tokenization (GPT-2 style).
pub struct Trainer {
    num_merges: usize,
    pre_tokenizer: PreTokenizer,
}

impl Trainer {
    /// Creates a new Trainer that will learn the specified number of merge rules.
    pub fn new(num_merges: usize) -> Self {
        Self {
            num_merges,
            pre_tokenizer: PreTokenizer::default(),
        }
    }

    /// Trains the BPE tokenizer on the given texts.
    ///
    /// Returns a list of merge rules learned from the training data.
    /// Each merge is a pair of strings (token1, token2) that should be merged.
    pub fn train(&self, training_texts: &[&str]) -> Vec<(String, String)> {
        let mut merges = Vec::new();
        let mut word_freqs = self.build_word_frequencies(training_texts);

        for _ in 0..self.num_merges {
            let pair_freqs = Self::get_pair_frequencies(&word_freqs);

            if let Some(most_common_pair) = Self::get_most_common_pair(&pair_freqs) {
                word_freqs = Self::merge_pair(&word_freqs, &most_common_pair);
                merges.push(most_common_pair);
            } else {
                break;
            }
        }

        merges
    }

    fn build_word_frequencies(&self, training_texts: &[&str]) -> HashMap<Vec<String>, usize> {
        let byte_encoder = bytes_to_unicode();

        training_texts
            .iter()
            .flat_map(|text| self.pre_tokenizer.pre_tokenize(text))
            .map(|chunk| {
                chunk
                    .as_bytes()
                    .iter()
                    .map(|&byte| byte_encoder[&byte].to_string())
                    .collect::<Vec<String>>()
            })
            .fold(HashMap::new(), |mut word_freqs, tokens| {
                *word_freqs.entry(tokens).or_insert(0) += 1;
                word_freqs
            })
    }

    fn get_pair_frequencies(
        word_freqs: &HashMap<Vec<String>, usize>,
    ) -> HashMap<(String, String), usize> {
        word_freqs
            .iter()
            .flat_map(|(symbols, &count)| {
                symbols
                    .windows(2)
                    .map(move |pair| ((pair[0].clone(), pair[1].clone()), count))
            })
            .fold(HashMap::new(), |mut pair_freqs, (pair, count)| {
                *pair_freqs.entry(pair).or_insert(0) += count;
                pair_freqs
            })
    }

    fn get_most_common_pair(
        pair_freqs: &HashMap<(String, String), usize>,
    ) -> Option<(String, String)> {
        pair_freqs
            .iter()
            .max_by(|(pair_a, count_a), (pair_b, count_b)| {
                // Match HuggingFace tokenizers: sort by count first, then lexicographically
                count_a.cmp(count_b).then_with(|| pair_a.cmp(pair_b))
            })
            .map(|(pair, _)| pair.clone())
    }

    fn merge_pair(
        word_freqs: &HashMap<Vec<String>, usize>,
        pair: &(String, String),
    ) -> HashMap<Vec<String>, usize> {
        let merged_token = format!("{}{}", pair.0, pair.1);

        word_freqs
            .iter()
            .map(|(symbols, &count)| {
                let mut merged_symbols = Vec::new();
                let mut i = 0;

                while i < symbols.len() {
                    if i + 1 < symbols.len() && symbols[i] == pair.0 && symbols[i + 1] == pair.1 {
                        merged_symbols.push(merged_token.clone());
                        i += 2;
                    } else {
                        merged_symbols.push(symbols[i].clone());
                        i += 1;
                    }
                }

                (merged_symbols, count)
            })
            .fold(HashMap::new(), |mut merged_freqs, (symbols, count)| {
                *merged_freqs.entry(symbols).or_insert(0) += count;
                merged_freqs
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_to_tokens(chunk: &str) -> Vec<String> {
        let byte_encoder = bytes_to_unicode();
        chunk
            .as_bytes()
            .iter()
            .map(|&byte| byte_encoder[&byte].to_string())
            .collect()
    }

    #[test]
    fn train_no_merges_returns_empty() {
        let trainer = Trainer::new(0);
        let result = trainer.train(&["hello world"]);

        assert!(result.is_empty());
    }

    #[test]
    fn train_with_merges_returns_correct_count() {
        let trainer = Trainer::new(5);
        let result = trainer.train(&["hello hello hello world"]);

        assert_eq!(result.len(), 5);

        for (first, second) in &result {
            assert!(!first.is_empty());
            assert!(!second.is_empty());
        }
    }

    #[test]
    fn train_produces_ordered_merges() {
        let trainer = Trainer::new(3);
        let result = trainer.train(&["aaaa bbbb"]);

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn train_handles_two_byte_utf8_sequences() {
        let trainer = Trainer::new(2);
        let result = trainer.train(&["café café café"]);

        assert_eq!(result.len(), 2);

        let byte_encoder = bytes_to_unicode();

        // "café café café" pre-tokenizes to ["café", " café", " café"]
        // "café" bytes: [0x63='c', 0x61='a', 0x66='f', 0xc3='Ã', 0xa9='©']
        // " café" bytes: [0x20='Ġ', 0x63='c', 0x61='a', 0x66='f', 0xc3='Ã', 0xa9='©']

        // Most frequent pairs with count 3: (Ã, ©), (c, a), (a, f), (f, Ã)
        // Most frequent pair with count 2: (Ġ, c)
        // First merge should be one of the pairs that appears 3 times
        let e_byte1 = byte_encoder[&0xc3].to_string(); // 'Ã'
        let e_byte2 = byte_encoder[&0xa9].to_string(); // '©'

        // Verify that the UTF-8 byte sequence for 'é' (Ã, ©) is one of the learned merges
        assert!(result.contains(&(e_byte1, e_byte2)));
    }

    #[test]
    fn train_handles_three_byte_utf8_sequences() {
        let trainer = Trainer::new(3);
        let result = trainer.train(&["日本 日本 日本"]);

        assert_eq!(result.len(), 3);

        let byte_encoder = bytes_to_unicode();

        // "日" (U+65E5) in UTF-8: [0xe6, 0x97, 0xa5]
        // These bytes appear 3 times in "日本 日本 日本"
        let byte1 = byte_encoder[&0xe6].to_string();
        let byte2 = byte_encoder[&0x97].to_string();
        let byte3 = byte_encoder[&0xa5].to_string();

        // The three bytes of "日" should be merged together in sequence
        // First merge: two adjacent bytes from "日"
        // Second merge: the result with the third byte
        assert!(
            result.contains(&(byte1.clone(), byte2.clone()))
                || result.contains(&(byte2.clone(), byte3.clone()))
        );
    }

    #[test]
    fn build_word_frequencies_empty_input() {
        let trainer = Trainer::new(10);
        let result = trainer.build_word_frequencies(&[]);

        assert!(result.is_empty());
    }

    #[test]
    fn build_word_frequencies_single_word() {
        let trainer = Trainer::new(10);
        let result = trainer.build_word_frequencies(&["hello"]);

        assert_eq!(result.len(), 1);

        let expected_tokens = chunk_to_tokens("hello");
        assert_eq!(result.get(&expected_tokens), Some(&1));
    }

    #[test]
    fn build_word_frequencies_counts_correctly() {
        let trainer = Trainer::new(10);
        let result = trainer.build_word_frequencies(&["test test test"]);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn build_word_frequencies_handles_punctuation() {
        let trainer = Trainer::new(10);
        let result = trainer.build_word_frequencies(&["hello, world!"]);

        assert_eq!(result.len(), 4);

        assert!(result.contains_key(&chunk_to_tokens(",")));
        assert!(result.contains_key(&chunk_to_tokens("!")));
    }

    #[test]
    fn get_pair_frequencies_empty() {
        let word_freqs = HashMap::new();
        let pair_freqs = Trainer::get_pair_frequencies(&word_freqs);

        assert!(pair_freqs.is_empty());
    }

    #[test]
    fn get_pair_frequencies_finds_pairs() {
        let mut word_freqs = HashMap::new();
        word_freqs.insert(vec!["a".to_string(), "b".to_string(), "c".to_string()], 1);

        let pair_freqs = Trainer::get_pair_frequencies(&word_freqs);

        assert_eq!(pair_freqs.len(), 2);
        assert_eq!(
            pair_freqs.get(&("a".to_string(), "b".to_string())),
            Some(&1)
        );
        assert_eq!(
            pair_freqs.get(&("b".to_string(), "c".to_string())),
            Some(&1)
        );
    }

    #[test]
    fn get_pair_frequencies_counts_frequency() {
        let mut word_freqs = HashMap::new();
        word_freqs.insert(vec!["a".to_string(), "b".to_string()], 3);

        let pair_freqs = Trainer::get_pair_frequencies(&word_freqs);

        assert_eq!(
            pair_freqs.get(&("a".to_string(), "b".to_string())),
            Some(&3)
        );
    }

    #[test]
    fn get_most_common_pair_none() {
        let pair_freqs = HashMap::new();
        let result = Trainer::get_most_common_pair(&pair_freqs);

        assert_eq!(result, None);
    }

    #[test]
    fn get_most_common_pair_finds_max() {
        let mut pair_freqs = HashMap::new();
        pair_freqs.insert(("a".to_string(), "b".to_string()), 5);
        pair_freqs.insert(("c".to_string(), "d".to_string()), 10);
        pair_freqs.insert(("e".to_string(), "f".to_string()), 3);

        let result = Trainer::get_most_common_pair(&pair_freqs);

        assert_eq!(result, Some(("c".to_string(), "d".to_string())));
    }

    #[test]
    fn get_most_common_pair_breaks_tie_lexicographically() {
        let mut pair_freqs = HashMap::new();
        pair_freqs.insert(("z".to_string(), "a".to_string()), 3);
        pair_freqs.insert(("a".to_string(), "b".to_string()), 3);
        pair_freqs.insert(("c".to_string(), "d".to_string()), 3);

        let result = Trainer::get_most_common_pair(&pair_freqs);

        // With equal frequencies, should pick lexicographically largest: ("z", "a")
        // This matches HuggingFace tokenizers behavior
        assert_eq!(result, Some(("z".to_string(), "a".to_string())));
    }

    #[test]
    fn merge_pair_combines_tokens() {
        let mut word_freqs = HashMap::new();
        word_freqs.insert(vec!["a".to_string(), "b".to_string(), "c".to_string()], 1);

        let result = Trainer::merge_pair(&word_freqs, &("a".to_string(), "b".to_string()));

        let expected = vec!["ab".to_string(), "c".to_string()];
        assert_eq!(result.get(&expected), Some(&1));
    }

    #[test]
    fn merge_pair_preserves_frequency() {
        let mut word_freqs = HashMap::new();
        word_freqs.insert(vec!["a".to_string(), "b".to_string()], 5);

        let result = Trainer::merge_pair(&word_freqs, &("a".to_string(), "b".to_string()));

        let expected = vec!["ab".to_string()];
        assert_eq!(result.get(&expected), Some(&5));
    }

    #[test]
    fn merge_pair_handles_multiple_occurrences() {
        let mut word_freqs = HashMap::new();
        word_freqs.insert(
            vec![
                "a".to_string(),
                "b".to_string(),
                "a".to_string(),
                "b".to_string(),
            ],
            1,
        );

        let result = Trainer::merge_pair(&word_freqs, &("a".to_string(), "b".to_string()));

        let expected = vec!["ab".to_string(), "ab".to_string()];
        assert_eq!(result.get(&expected), Some(&1));
    }

    #[test]
    fn train_produces_same_merges_regardless_of_special_tokens() {
        use crate::Vocabulary;

        let trainer = Trainer::new(5);
        let merges = trainer.train(&["hello world hello world hello world"]);

        let vocab_without_special = Vocabulary::new(vec![], merges.clone());

        let special_tokens = vec!["<|endoftext|>".to_string(), "[PAD]".to_string()];
        let vocab_with_special = Vocabulary::new(special_tokens.clone(), merges.clone());

        assert_eq!(merges.len(), 5);

        let first_merge = format!("{}{}", merges[0].0, merges[0].1);

        assert_eq!(vocab_without_special.token_to_id(&first_merge), Some(256));

        assert_eq!(vocab_with_special.token_to_id(&first_merge), Some(258));

        assert_eq!(vocab_without_special.token_to_id("<|endoftext|>"), None);
        assert_eq!(vocab_with_special.token_to_id("<|endoftext|>"), Some(0));

        assert_eq!(vocab_without_special.token_to_id("[PAD]"), None);
        assert_eq!(vocab_with_special.token_to_id("[PAD]"), Some(1));
    }
}
