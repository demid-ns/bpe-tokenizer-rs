use std::collections::HashMap;

use crate::bytes_to_unicode;

/// Manages bidirectional mapping between tokens and their IDs for BPE tokenization.
///
/// The vocabulary maintains a complete mapping between string tokens and their numeric IDs,
/// supporting O(1) lookups in both directions. It includes:
/// - Special tokens (e.g., `<|endoftext|>`, `[PAD]`) starting at ID 0
/// - Base byte-level tokens (256 characters) covering all possible bytes
/// - Merged tokens learned during BPE training
///
/// # Token ID Assignment
///
/// IDs are assigned sequentially in this order:
/// 1. Special tokens: 0, 1, 2, ...
/// 2. Byte-level tokens: sorted by Unicode character value
/// 3. Merged tokens: in the order they were learned during training
///
/// # Performance
///
/// Uses two data structures for optimal performance:
/// - `token_to_id`: HashMap for fast token → ID lookup (used during encoding)
/// - `id_to_token`: Vec for fast ID → token lookup (used during decoding)
///
/// Vec is used instead of HashMap<u32, String> for `id_to_token` because IDs are
/// sequential (0, 1, 2, ...), making Vec index access more efficient than hash lookup.
///
/// # Examples
///
/// ```
/// use bpe_tokenizer_rs::Vocabulary;
///
/// let special_tokens = vec!["<|endoftext|>".to_string()];
/// let merges = vec![("h".to_string(), "e".to_string())];
/// let vocab = Vocabulary::new(special_tokens, merges);
///
/// assert_eq!(vocab.token_to_id("<|endoftext|>"), Some(0));
/// assert_eq!(vocab.token_to_id("he"), Some(257));
/// assert_eq!(vocab.id_to_token(0), Some("<|endoftext|>"));
/// ```
#[derive(Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl Vocabulary {
    /// Creates a new vocabulary from special tokens and merge rules.
    ///
    /// The vocabulary is constructed by adding tokens in this order:
    /// 1. Special tokens (if any)
    /// 2. All 256 byte-level base tokens (sorted by Unicode value)
    /// 3. Merged tokens from BPE training
    ///
    /// # Arguments
    ///
    /// * `special_tokens` - Vector of special tokens (e.g., `<|endoftext|>`, `[PAD]`)
    /// * `merges` - Vector of merge rules as (token1, token2) pairs
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::Vocabulary;
    ///
    /// let vocab = Vocabulary::new(vec![], vec![]);
    /// assert_eq!(vocab.token_to_id("A"), Some(32));
    /// ```
    pub fn new(special_tokens: Vec<String>, merges: Vec<(String, String)>) -> Self {
        let total_size = special_tokens.len() + 256 + merges.len();
        let mut token_to_id = HashMap::with_capacity(total_size);
        let mut id_to_token = Vec::with_capacity(total_size);

        for special_token in special_tokens {
            let id = id_to_token.len() as u32;
            token_to_id.insert(special_token.clone(), id);
            id_to_token.push(special_token);
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_chars: Vec<(u8, char)> = byte_encoder.iter().map(|(&b, &c)| (b, c)).collect();
        byte_chars.sort_by_key(|(_, c)| *c as u32);

        for (_, ch) in byte_chars {
            let token = ch.to_string();
            let id = id_to_token.len() as u32;
            token_to_id.insert(token.clone(), id);
            id_to_token.push(token);
        }

        for (part1, part2) in merges {
            let token = format!("{}{}", part1, part2);
            let id = id_to_token.len() as u32;
            token_to_id.insert(token.clone(), id);
            id_to_token.push(token);
        }

        Vocabulary {
            token_to_id,
            id_to_token,
        }
    }

    /// Converts a token string to its corresponding ID.
    ///
    /// # Arguments
    ///
    /// * `token` - The token string to look up
    ///
    /// # Returns
    ///
    /// * `Some(id)` if the token exists in the vocabulary
    /// * `None` if the token is not found
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::Vocabulary;
    ///
    /// let vocab = Vocabulary::new(vec![], vec![]);
    /// assert_eq!(vocab.token_to_id("A"), Some(32));
    /// assert_eq!(vocab.token_to_id("unknown"), None);
    /// ```
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Converts a token ID to its corresponding string.
    ///
    /// # Arguments
    ///
    /// * `id` - The token ID to look up
    ///
    /// # Returns
    ///
    /// * `Some(&str)` if the ID exists in the vocabulary
    /// * `None` if the ID is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::Vocabulary;
    ///
    /// let vocab = Vocabulary::new(vec![], vec![]);
    /// assert_eq!(vocab.id_to_token(32), Some("A"));
    /// assert_eq!(vocab.id_to_token(99999), None);
    /// ```
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_base_tokens_correct() {
        let vocab = Vocabulary::new(vec![], vec![]);

        assert_eq!(vocab.token_to_id("A"), Some(32));
        assert_eq!(vocab.id_to_token(32), Some("A"));

        assert_eq!(vocab.token_to_id("Ā"), Some(188));
        assert_eq!(vocab.id_to_token(188), Some("Ā"));

        assert_eq!(vocab.token_to_id("Ġ"), Some(220));
        assert_eq!(vocab.id_to_token(220), Some("Ġ"));
    }

    #[test]
    fn vocabulary_with_merges() {
        let merges = vec![
            ("n".to_string(), "a".to_string()),
            ("na".to_string(), "na".to_string()),
        ];
        let vocab = Vocabulary::new(vec![], merges);

        assert_eq!(vocab.token_to_id("n"), Some(77));
        assert_eq!(vocab.token_to_id("a"), Some(64));

        assert_eq!(vocab.token_to_id("na"), Some(256));
        assert_eq!(vocab.id_to_token(256), Some("na"));

        assert_eq!(vocab.token_to_id("nana"), Some(257));
        assert_eq!(vocab.id_to_token(257), Some("nana"));
    }

    #[test]
    fn vocabulary_unknown_token() {
        let vocab = Vocabulary::new(vec![], vec![]);

        assert_eq!(vocab.token_to_id("unknown_token"), None);
    }

    #[test]
    fn vocabulary_unknown_id() {
        let vocab = Vocabulary::new(vec![], vec![]);

        // ID beyond vocabulary size
        assert_eq!(vocab.id_to_token(10000), None);
    }

    #[test]
    fn vocabulary_round_trip() {
        let merges = vec![
            ("t".to_string(), "h".to_string()),
            ("th".to_string(), "e".to_string()),
        ];
        let vocab = Vocabulary::new(vec![], merges);

        // Test round-trip: token → id → token
        let token = "the";
        let id = vocab.token_to_id(token).unwrap();
        let recovered_token = vocab.id_to_token(id).unwrap();
        assert_eq!(token, recovered_token);
    }

    #[test]
    fn vocabulary_with_one_special_token() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let vocab = Vocabulary::new(special_tokens, vec![]);

        assert_eq!(vocab.token_to_id("<|endoftext|>"), Some(0));
        assert_eq!(vocab.id_to_token(0), Some("<|endoftext|>"));

        assert_eq!(vocab.token_to_id("Ā"), Some(189));
        assert_eq!(vocab.id_to_token(189), Some("Ā"));

        assert_eq!(vocab.token_to_id("A"), Some(33));
        assert_eq!(vocab.id_to_token(33), Some("A"));
    }

    #[test]
    fn vocabulary_with_multiple_special_tokens() {
        let special_tokens = vec![
            "<|endoftext|>".to_string(),
            "[PAD]".to_string(),
            "[UNK]".to_string(),
        ];
        let vocab = Vocabulary::new(special_tokens, vec![]);

        assert_eq!(vocab.token_to_id("<|endoftext|>"), Some(0));
        assert_eq!(vocab.id_to_token(0), Some("<|endoftext|>"));

        assert_eq!(vocab.token_to_id("[PAD]"), Some(1));
        assert_eq!(vocab.id_to_token(1), Some("[PAD]"));

        assert_eq!(vocab.token_to_id("[UNK]"), Some(2));
        assert_eq!(vocab.id_to_token(2), Some("[UNK]"));

        assert_eq!(vocab.token_to_id("Ā"), Some(191));
        assert_eq!(vocab.id_to_token(191), Some("Ā"));

        assert_eq!(vocab.token_to_id("A"), Some(35));
        assert_eq!(vocab.id_to_token(35), Some("A"));
    }

    #[test]
    fn vocabulary_with_special_tokens_and_merges() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
        ];
        let vocab = Vocabulary::new(special_tokens, merges);

        assert_eq!(vocab.token_to_id("<|endoftext|>"), Some(0));

        assert_eq!(vocab.token_to_id("h"), Some(72));
        assert_eq!(vocab.token_to_id("e"), Some(69));
        assert_eq!(vocab.token_to_id("l"), Some(76));

        assert_eq!(vocab.token_to_id("he"), Some(257));
        assert_eq!(vocab.id_to_token(257), Some("he"));

        assert_eq!(vocab.token_to_id("hel"), Some(258));
        assert_eq!(vocab.id_to_token(258), Some("hel"));
    }

    #[test]
    fn vocabulary_special_token_round_trip() {
        let special_tokens = vec!["<|endoftext|>".to_string(), "[PAD]".to_string()];
        let vocab = Vocabulary::new(special_tokens, vec![]);

        let token1 = "<|endoftext|>";
        let id1 = vocab.token_to_id(token1).unwrap();
        let recovered1 = vocab.id_to_token(id1).unwrap();
        assert_eq!(token1, recovered1);

        let token2 = "[PAD]";
        let id2 = vocab.token_to_id(token2).unwrap();
        let recovered2 = vocab.id_to_token(id2).unwrap();
        assert_eq!(token2, recovered2);
    }
}
