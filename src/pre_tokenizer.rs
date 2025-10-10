use regex::Regex;

pub struct PreTokenizer {
    pub pattern: Regex,
}

impl Default for PreTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PreTokenizer {
    pub fn new() -> Self {
        // GPT-2 regex pattern, simplified for Rust's regex crate (no lookahead support)
        // Matches: contractions, letters (with optional space), numbers (with optional space),
        // punctuation (with optional space), and remaining whitespace
        let pattern =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();

        PreTokenizer { pattern }
    }

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
