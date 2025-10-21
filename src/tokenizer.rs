use crate::{Decoder, Encoder, PreTokenizer, Trainer, Vocabulary};

/// A complete Byte Pair Encoding (BPE) tokenizer for encoding and decoding text.
///
/// `BpeTokenizer` provides a high-level interface for BPE tokenization, combining
/// an encoder and decoder with a shared vocabulary. It supports:
/// - Byte-level encoding compatible with GPT-2
/// - Special tokens (e.g., `<|endoftext|>`, `[PAD]`)
/// - Custom merge rules learned from training data
///
/// # Examples
///
/// ## Creating from merge rules
///
/// ```
/// use bpe_tokenizer_rs::BpeTokenizer;
///
/// let merges = vec![("h".to_string(), "e".to_string())];
/// let special_tokens = vec!["<|endoftext|>".to_string()];
/// let tokenizer = BpeTokenizer::new(merges, special_tokens);
///
/// let ids = tokenizer.encode("hello");
/// let text = tokenizer.decode(&ids);
/// assert_eq!(text, "hello");
/// ```
///
/// ## Training from scratch
///
/// ```
/// use bpe_tokenizer_rs::{BpeTokenizer, Trainer};
///
/// let trainer = Trainer::new(10);
/// let training_data = &["hello world", "hello there"];
/// let tokenizer = BpeTokenizer::from_trainer(&trainer, training_data, vec![]);
///
/// let ids = tokenizer.encode("hello");
/// let text = tokenizer.decode(&ids);
/// assert_eq!(text, "hello");
/// ```
pub struct BpeTokenizer {
    encoder: Encoder,
    decoder: Decoder,
}

impl BpeTokenizer {
    /// Creates a new tokenizer from merge rules and special tokens.
    ///
    /// # Arguments
    ///
    /// * `merges` - BPE merge rules as (token1, token2) pairs
    /// * `special_tokens` - List of special tokens (e.g., `<|endoftext|>`, `[PAD]`)
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::BpeTokenizer;
    ///
    /// let tokenizer = BpeTokenizer::new(vec![], vec![]);
    /// let ids = tokenizer.encode("Hello");
    /// assert_eq!(tokenizer.decode(&ids), "Hello");
    /// ```
    pub fn new(merges: Vec<(String, String)>, special_tokens: Vec<String>) -> Self {
        let pre_tokenizer = PreTokenizer::new();
        let vocabulary = Vocabulary::new(special_tokens.clone(), merges.clone());
        let encoder = Encoder::new(merges, pre_tokenizer, vocabulary.clone(), special_tokens);
        let decoder = Decoder::new(vocabulary);

        BpeTokenizer { encoder, decoder }
    }

    /// Encodes text into a sequence of token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode
    ///
    /// # Returns
    ///
    /// A vector of token IDs representing the encoded text.
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::BpeTokenizer;
    ///
    /// let tokenizer = BpeTokenizer::new(vec![], vec![]);
    /// let ids = tokenizer.encode("AB");
    /// assert_eq!(ids, vec![32, 33]);
    /// ```
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encoder.encode(text)
    }

    /// Decodes a sequence of token IDs back into text.
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of token IDs to decode
    ///
    /// # Returns
    ///
    /// The decoded text as a UTF-8 string.
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::BpeTokenizer;
    ///
    /// let tokenizer = BpeTokenizer::new(vec![], vec![]);
    /// let text = tokenizer.decode(&[32, 33]);
    /// assert_eq!(text, "AB");
    /// ```
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decoder.decode(ids)
    }

    /// Creates a tokenizer by training on the provided texts.
    ///
    /// This is a convenience method that trains a BPE model and creates a tokenizer
    /// in one step.
    ///
    /// # Arguments
    ///
    /// * `trainer` - The trainer configured with the desired number of merges
    /// * `training_texts` - Texts to train on
    /// * `special_tokens` - List of special tokens to include
    ///
    /// # Returns
    ///
    /// A new `BpeTokenizer` with merge rules learned from the training data.
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::{BpeTokenizer, Trainer};
    ///
    /// let trainer = Trainer::new(5);
    /// let tokenizer = BpeTokenizer::from_trainer(
    ///     &trainer,
    ///     &["hello world", "hello"],
    ///     vec![]
    /// );
    ///
    /// let ids = tokenizer.encode("hello");
    /// assert_eq!(tokenizer.decode(&ids), "hello");
    /// ```
    pub fn from_trainer(
        trainer: &Trainer,
        training_texts: &[&str],
        special_tokens: Vec<String>,
    ) -> BpeTokenizer {
        let merges = trainer.train(training_texts);

        Self::new(merges, special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_tokenizer_with_no_merges() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let ids = tokenizer.encode("A");

        assert_eq!(ids, vec![32]);
    }

    #[test]
    fn new_creates_tokenizer_with_special_tokens() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let tokenizer = BpeTokenizer::new(vec![], special_tokens);

        let ids = tokenizer.encode("<|endoftext|>");

        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn new_creates_tokenizer_with_merges() {
        let merges = vec![("a".to_string(), "b".to_string())];
        let tokenizer = BpeTokenizer::new(merges, vec![]);

        let ids = tokenizer.encode("ab");

        assert_eq!(ids, vec![256]);
    }

    #[test]
    fn encode_empty_string() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let ids = tokenizer.encode("");

        assert_eq!(ids, vec![]);
    }

    #[test]
    fn encode_single_ascii_char() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let ids = tokenizer.encode("B");

        assert_eq!(ids, vec![33]);
    }

    #[test]
    fn encode_multiple_ascii_chars() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let ids = tokenizer.encode("ABC");

        assert_eq!(ids, vec![32, 33, 34]);
    }

    #[test]
    fn encode_utf8_two_bytes() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let ids = tokenizer.encode("é");

        assert_eq!(ids, vec![127, 102]);
    }

    #[test]
    fn encode_japanese() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let ids = tokenizer.encode("日");

        assert_eq!(ids, vec![162, 245, 98]);
    }

    #[test]
    fn decode_empty_sequence() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let text = tokenizer.decode(&[]);

        assert_eq!(text, "");
    }

    #[test]
    fn decode_single_ascii_char() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let text = tokenizer.decode(&[32]);

        assert_eq!(text, "A");
    }

    #[test]
    fn decode_multiple_ascii_chars() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let text = tokenizer.decode(&[39, 72]);

        assert_eq!(text, "Hi");
    }

    #[test]
    fn decode_utf8_two_bytes() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let text = tokenizer.decode(&[127, 102]);

        assert_eq!(text, "é");
    }

    #[test]
    fn decode_japanese() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let text = tokenizer.decode(&[162, 245, 98]);

        assert_eq!(text, "日");
    }

    #[test]
    fn roundtrip_ascii() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let original = "Hello";
        let ids = tokenizer.encode(original);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn roundtrip_utf8() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let original = "Hello 世界";
        let ids = tokenizer.encode(original);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn roundtrip_with_special_tokens() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let tokenizer = BpeTokenizer::new(vec![], special_tokens);

        let original = "<|endoftext|>Hello<|endoftext|>";
        let ids = tokenizer.encode(original);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn roundtrip_with_merges() {
        let trainer = Trainer::new(5);
        let tokenizer = BpeTokenizer::from_trainer(&trainer, &["hello world"], vec![]);

        let original = "hello";
        let ids = tokenizer.encode(original);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn from_trainer_creates_working_tokenizer() {
        let trainer = Trainer::new(1);
        let tokenizer = BpeTokenizer::from_trainer(&trainer, &["aa aa aa"], vec![]);

        let ids = tokenizer.encode("aa");

        assert_eq!(ids, vec![256]);
    }

    #[test]
    fn from_trainer_with_special_tokens() {
        let trainer = Trainer::new(0);
        let special_tokens = vec!["[PAD]".to_string()];
        let tokenizer = BpeTokenizer::from_trainer(&trainer, &["test"], special_tokens);

        let ids = tokenizer.encode("[PAD]");

        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn chinese_with_single_merge() {
        let trainer = Trainer::new(1);
        let tokenizer = BpeTokenizer::from_trainer(&trainer, &["世界 世界 世界"], vec![]);

        let ids = tokenizer.encode("世界");

        assert_eq!(ids, vec![160, 256, 163, 243, 234]);
    }

    #[test]
    fn chinese_roundtrip_with_merge() {
        let trainer = Trainer::new(1);
        let tokenizer = BpeTokenizer::from_trainer(&trainer, &["世界 世界 世界"], vec![]);

        let original = "世界";
        let ids = tokenizer.encode(original);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn russian_with_single_merge() {
        let trainer = Trainer::new(1);
        let tokenizer = BpeTokenizer::from_trainer(&trainer, &["Привет Привет"], vec![]);

        let ids = tokenizer.encode("Привет");

        assert_eq!(
            ids,
            vec![140, 253, 141, 222, 140, 116, 140, 256, 113, 141, 224]
        );
    }

    #[test]
    fn emoji_roundtrip() {
        let tokenizer = BpeTokenizer::new(vec![], vec![]);

        let original = "🦀";
        let ids = tokenizer.encode(original);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, original);
    }

    #[test]
    fn multiple_special_tokens() {
        let special_tokens = vec!["<|start|>".to_string(), "<|end|>".to_string()];
        let tokenizer = BpeTokenizer::new(vec![], special_tokens);

        let start_ids = tokenizer.encode("<|start|>");
        let end_ids = tokenizer.encode("<|end|>");

        assert_eq!(start_ids, vec![0]);
        assert_eq!(end_ids, vec![1]);
    }

    #[test]
    fn special_tokens_with_text() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let tokenizer = BpeTokenizer::new(vec![], special_tokens);

        let ids = tokenizer.encode("<|endoftext|>A");

        assert_eq!(ids, vec![0, 33]);
    }
}
