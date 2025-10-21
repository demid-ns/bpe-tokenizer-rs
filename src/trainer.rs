use crate::{PreTokenizer, bytes_to_unicode};
use std::collections::HashMap;

/// Trains a BPE tokenizer by learning merge rules from training data.
///
/// The trainer implements the Byte Pair Encoding training algorithm:
/// 1. Pre-tokenizes input text into words/chunks
/// 2. Converts each chunk to byte-level Unicode representation
/// 3. Counts frequency of adjacent token pairs
/// 4. Iteratively merges the most frequent pair
/// 5. Repeats until the desired number of merges is reached
///
/// # Algorithm
///
/// The training process uses GPT-2 style byte-level encoding with a pre-tokenizer
/// that splits on whitespace and punctuation. Merge rules are learned by:
/// - Finding the most frequent adjacent pair of tokens
/// - Breaking ties by preferring pairs with lower token IDs
/// - Creating a new merged token from the pair
/// - Updating all occurrences in the training data
///
/// # Examples
///
/// ```
/// use bpe_tokenizer_rs::Trainer;
///
/// let trainer = Trainer::new(10);
/// let merges = trainer.train(&["hello world", "hello there"]);
///
/// assert!(merges.len() <= 10);
/// ```
pub struct Trainer {
    num_merges: usize,
    pre_tokenizer: PreTokenizer,
}

impl Trainer {
    /// Creates a new trainer that will learn the specified number of merge rules.
    ///
    /// # Arguments
    ///
    /// * `num_merges` - Maximum number of merge rules to learn
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::Trainer;
    ///
    /// let trainer = Trainer::new(100);
    /// ```
    pub fn new(num_merges: usize) -> Self {
        Self {
            num_merges,
            pre_tokenizer: PreTokenizer::default(),
        }
    }

    /// Trains the BPE tokenizer on the given texts.
    ///
    /// Learns merge rules by iteratively finding and merging the most frequent
    /// adjacent token pairs in the training data. Training stops when either:
    /// - The requested number of merges is reached
    /// - No more pairs can be merged (all tokens are isolated)
    ///
    /// # Arguments
    ///
    /// * `training_texts` - Slice of text strings to train on
    ///
    /// # Returns
    ///
    /// A vector of merge rules as (token1, token2) pairs, in the order they were learned.
    /// The length may be less than `num_merges` if training converges early.
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::Trainer;
    ///
    /// let trainer = Trainer::new(5);
    /// let merges = trainer.train(&["hello world", "hello there"]);
    ///
    /// assert!(merges.len() <= 5);
    /// ```
    pub fn train(&self, training_texts: &[&str]) -> Vec<(String, String)> {
        let mut merges = Vec::with_capacity(self.num_merges);
        let mut word_freqs = self.build_word_frequencies(training_texts);
        let mut token_to_id = self.build_initial_token_to_id();
        let mut next_id = token_to_id.len() as u32;

        for _ in 0..self.num_merges {
            let pair_freqs = Self::compute_pair_frequencies(&word_freqs);

            if let Some(best_pair) = Self::find_best_pair(&pair_freqs, &token_to_id) {
                word_freqs = Self::apply_merge(&word_freqs, &best_pair);

                let merged_token = Self::create_merged_token(&best_pair);
                token_to_id.insert(merged_token, next_id);
                next_id += 1;

                merges.push(best_pair);
            } else {
                break;
            }
        }

        merges
    }

    fn build_initial_token_to_id(&self) -> HashMap<String, u32> {
        let byte_encoder = bytes_to_unicode();
        let mut byte_chars: Vec<(u8, char)> = byte_encoder.iter().map(|(&b, &c)| (b, c)).collect();
        byte_chars.sort_by_key(|(_, c)| *c as u32);

        byte_chars
            .iter()
            .enumerate()
            .map(|(id, (_, ch))| (ch.to_string(), id as u32))
            .collect()
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

    fn compute_pair_frequencies(
        word_freqs: &HashMap<Vec<String>, usize>,
    ) -> HashMap<(String, String), usize> {
        let mut pair_freqs = HashMap::new();

        for (symbols, &count) in word_freqs.iter() {
            for pair in symbols.windows(2) {
                *pair_freqs.entry((pair[0].clone(), pair[1].clone())).or_insert(0) += count;
            }
        }

        pair_freqs
    }

    fn find_best_pair(
        pair_freqs: &HashMap<(String, String), usize>,
        token_to_id: &HashMap<String, u32>,
    ) -> Option<(String, String)> {
        pair_freqs
            .iter()
            .max_by(|(pair_a, count_a), (pair_b, count_b)| {
                count_a.cmp(count_b).then_with(|| {
                    let ids_a = Self::get_pair_ids(pair_a, token_to_id);
                    let ids_b = Self::get_pair_ids(pair_b, token_to_id);
                    ids_b.cmp(&ids_a)
                })
            })
            .map(|(pair, _)| pair.clone())
    }

    fn get_pair_ids(pair: &(String, String), token_to_id: &HashMap<String, u32>) -> (u32, u32) {
        let id_0 = token_to_id.get(&pair.0).copied().unwrap_or(u32::MAX);
        let id_1 = token_to_id.get(&pair.1).copied().unwrap_or(u32::MAX);
        (id_0, id_1)
    }

    fn create_merged_token(pair: &(String, String)) -> String {
        format!("{}{}", pair.0, pair.1)
    }

    fn apply_merge(
        word_freqs: &HashMap<Vec<String>, usize>,
        pair: &(String, String),
    ) -> HashMap<Vec<String>, usize> {
        let merged_token = Self::create_merged_token(pair);

        word_freqs
            .iter()
            .map(|(symbols, &count)| {
                let merged_symbols = Self::merge_symbols(symbols, pair, &merged_token);
                (merged_symbols, count)
            })
            .fold(HashMap::new(), |mut merged_freqs, (symbols, count)| {
                *merged_freqs.entry(symbols).or_insert(0) += count;
                merged_freqs
            })
    }

    fn merge_symbols(
        symbols: &[String],
        pair: &(String, String),
        merged_token: &str,
    ) -> Vec<String> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < symbols.len() {
            if i + 1 < symbols.len() && symbols[i] == pair.0 && symbols[i + 1] == pair.1 {
                result.push(merged_token.to_string());
                i += 2;
            } else {
                result.push(symbols[i].clone());
                i += 1;
            }
        }

        result
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
    fn train_empty_input_returns_empty() {
        let trainer = Trainer::new(10);
        let result = trainer.train(&[]);

        assert_eq!(result, vec![]);
    }

    #[test]
    fn train_no_merges_returns_empty() {
        let trainer = Trainer::new(0);
        let result = trainer.train(&["hello world"]);

        assert_eq!(result, vec![]);
    }

    #[test]
    fn train_single_character_stops_early() {
        let trainer = Trainer::new(10);
        let result = trainer.train(&["a"]);

        assert_eq!(result, vec![]);
    }

    #[test]
    fn train_simple_text_produces_expected_merges() {
        let trainer = Trainer::new(3);
        let result = trainer.train(&["aa bb cc"]);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], ("a".to_string(), "a".to_string()));
        assert_eq!(result[1], ("b".to_string(), "b".to_string()));
        assert_eq!(result[2], ("c".to_string(), "c".to_string()));
    }

    #[test]
    fn train_respects_max_merges_limit() {
        let trainer = Trainer::new(2);
        let result = trainer.train(&["hello world"]);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn train_with_repeated_text_prioritizes_high_frequency() {
        let trainer = Trainer::new(1);
        let result = trainer.train(&["aaa bbb aaa"]);

        assert_eq!(result[0], ("a".to_string(), "a".to_string()));
    }

    #[test]
    fn train_breaks_tie_by_token_id_not_string() {
        let trainer = Trainer::new(5);
        let result = trainer.train(&["123 456 789"]);

        assert_eq!(result[0], ("1".to_string(), "2".to_string()));
        assert_eq!(result[1], ("4".to_string(), "5".to_string()));
        assert_eq!(result[2], ("7".to_string(), "8".to_string()));
    }

    #[test]
    fn train_handles_multibyte_utf8() {
        let trainer = Trainer::new(2);
        let result = trainer.train(&["こんにちは"]);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("ã".to_string(), "ģ".to_string()));
        assert_eq!(result[1], ("¡".to_string(), "ãģ".to_string()));
    }

    #[test]
    fn train_with_multiple_words_finds_common_pairs() {
        let trainer = Trainer::new(3);
        let result = trainer.train(&["hello", "hello", "world"]);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], ("e".to_string(), "l".to_string()));
        assert_eq!(result[1], ("h".to_string(), "el".to_string()));
        assert_eq!(result[2], ("l".to_string(), "o".to_string()));
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

        let expected_tokens = chunk_to_tokens("hello");
        assert_eq!(result.get(&expected_tokens), Some(&1));
    }

    #[test]
    fn compute_pair_frequencies_empty() {
        let word_freqs = HashMap::new();
        let pair_freqs = Trainer::compute_pair_frequencies(&word_freqs);

        assert!(pair_freqs.is_empty());
    }

    #[test]
    fn compute_pair_frequencies_finds_pairs() {
        let mut word_freqs = HashMap::new();
        word_freqs.insert(vec!["a".to_string(), "b".to_string(), "c".to_string()], 1);

        let pair_freqs = Trainer::compute_pair_frequencies(&word_freqs);

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
    fn find_best_pair_returns_none_when_empty() {
        let pair_freqs = HashMap::new();
        let token_to_id = HashMap::new();
        let result = Trainer::find_best_pair(&pair_freqs, &token_to_id);

        assert_eq!(result, None);
    }

    #[test]
    fn find_best_pair_selects_highest_frequency() {
        let mut pair_freqs = HashMap::new();
        pair_freqs.insert(("a".to_string(), "b".to_string()), 5);
        pair_freqs.insert(("c".to_string(), "d".to_string()), 10);
        pair_freqs.insert(("e".to_string(), "f".to_string()), 3);

        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        token_to_id.insert("d".to_string(), 3);
        token_to_id.insert("e".to_string(), 4);
        token_to_id.insert("f".to_string(), 5);

        let result = Trainer::find_best_pair(&pair_freqs, &token_to_id);

        assert_eq!(result, Some(("c".to_string(), "d".to_string())));
    }

    #[test]
    fn find_best_pair_breaks_tie_by_lowest_token_id() {
        let mut pair_freqs = HashMap::new();
        pair_freqs.insert(("z".to_string(), "a".to_string()), 3);
        pair_freqs.insert(("a".to_string(), "b".to_string()), 3);
        pair_freqs.insert(("c".to_string(), "d".to_string()), 3);

        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        token_to_id.insert("d".to_string(), 3);
        token_to_id.insert("z".to_string(), 25);

        let result = Trainer::find_best_pair(&pair_freqs, &token_to_id);

        assert_eq!(result, Some(("a".to_string(), "b".to_string())));
    }

    #[test]
    fn apply_merge_combines_adjacent_pair() {
        let mut word_freqs = HashMap::new();
        word_freqs.insert(vec!["a".to_string(), "b".to_string(), "c".to_string()], 1);

        let result = Trainer::apply_merge(&word_freqs, &("a".to_string(), "b".to_string()));

        let expected = vec!["ab".to_string(), "c".to_string()];
        assert_eq!(result.get(&expected), Some(&1));
    }

    #[test]
    fn apply_merge_preserves_word_frequency() {
        let mut word_freqs = HashMap::new();
        word_freqs.insert(vec!["a".to_string(), "b".to_string()], 5);

        let result = Trainer::apply_merge(&word_freqs, &("a".to_string(), "b".to_string()));

        let expected = vec!["ab".to_string()];
        assert_eq!(result.get(&expected), Some(&5));
    }

    #[test]
    fn apply_merge_handles_multiple_occurrences_in_same_word() {
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

        let result = Trainer::apply_merge(&word_freqs, &("a".to_string(), "b".to_string()));

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

        let first_merge = format!("{}{}", merges[0].0, merges[0].1);

        assert_eq!(vocab_without_special.token_to_id(&first_merge), Some(256));
        assert_eq!(vocab_with_special.token_to_id(&first_merge), Some(258));
        assert_eq!(vocab_without_special.token_to_id("<|endoftext|>"), None);
        assert_eq!(vocab_with_special.token_to_id("<|endoftext|>"), Some(0));
        assert_eq!(vocab_without_special.token_to_id("[PAD]"), None);
        assert_eq!(vocab_with_special.token_to_id("[PAD]"), Some(1));
    }
}
