use regex::Regex;

/// Pre-tokenizes text into chunks before BPE encoding.
///
/// The pre-tokenizer splits text into words, punctuation, and whitespace chunks
/// using a regex pattern compatible with GPT-2's tokenization strategy. This ensures
/// that BPE merges don't cross logical word boundaries.
///
/// # Pattern
///
/// The regex pattern matches:
/// - Contractions: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`
/// - Letters (with optional leading space): ` ?\p{L}+`
/// - Numbers (with optional leading space): ` ?\p{N}+`
/// - Punctuation (with optional leading space): ` ?[^\s\p{L}\p{N}]+`
/// - Remaining whitespace: `\s+`
///
/// # Examples
///
/// ```
/// use bpe_tokenizer_rs::PreTokenizer;
///
/// let pre_tokenizer = PreTokenizer::new();
/// let tokens = pre_tokenizer.pre_tokenize("Hello, world!");
///
/// assert_eq!(tokens, vec!["Hello", ",", " world", "!"]);
/// ```
pub struct PreTokenizer {
    pub pattern: Regex,
}

impl Default for PreTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PreTokenizer {
    /// Creates a new pre-tokenizer with GPT-2 style regex pattern.
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::PreTokenizer;
    ///
    /// let pre_tokenizer = PreTokenizer::new();
    /// ```
    pub fn new() -> Self {
        let pattern =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();

        PreTokenizer { pattern }
    }

    /// Pre-tokenizes text into chunks.
    ///
    /// Splits the input text according to the GPT-2 pattern, preserving spaces
    /// that precede words, numbers, or punctuation.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to pre-tokenize
    ///
    /// # Returns
    ///
    /// A vector of string chunks representing the pre-tokenized text.
    ///
    /// # Examples
    ///
    /// ```
    /// use bpe_tokenizer_rs::PreTokenizer;
    ///
    /// let pre_tokenizer = PreTokenizer::new();
    /// let tokens = pre_tokenizer.pre_tokenize("I'm happy!");
    ///
    /// assert_eq!(tokens, vec!["I", "'m", " happy", "!"]);
    /// ```
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        self.pattern
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_tokenize_basic_sentence() {
        let tokenizer = PreTokenizer::new();
        let result = tokenizer.pre_tokenize("Hello, world!");

        assert_eq!(result, vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn pre_tokenize_contractions() {
        let tokenizer = PreTokenizer::new();
        let result = tokenizer.pre_tokenize("don't");

        assert_eq!(result, vec!["don", "'t"]);
    }

    #[test]
    fn pre_tokenize_multiple_contractions() {
        let tokenizer = PreTokenizer::new();
        let result = tokenizer.pre_tokenize("I'm sure it's fine");

        assert_eq!(result, vec!["I", "'m", " sure", " it", "'s", " fine"]);
    }

    #[test]
    fn pre_tokenize_numbers() {
        let tokenizer = PreTokenizer::new();
        let result = tokenizer.pre_tokenize("I have 123 apples");

        assert_eq!(result, vec!["I", " have", " 123", " apples"]);
    }

    #[test]
    fn pre_tokenize_punctuation() {
        let tokenizer = PreTokenizer::new();
        let result = tokenizer.pre_tokenize("Hello... What?!");

        assert_eq!(result, vec!["Hello", "...", " What", "?!"]);
    }

    #[test]
    fn pre_tokenize_keeps_spaces_with_words() {
        let tokenizer = PreTokenizer::new();
        let result = tokenizer.pre_tokenize("Hello world");

        assert_eq!(result, vec!["Hello", " world"]);
    }
}
