pub fn word_to_symbols(word: &str) -> Result<Vec<String>, String> {
    if word.contains(' ') {
        return Err(format!("Input contains multiple words: '{}'", word));
    }

    let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    symbols.push("</w>".to_string());
    Ok(symbols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_to_symbols_basic() {
        let result = word_to_symbols("banana").unwrap();
        let expected = vec!["b", "a", "n", "a", "n", "a", "</w>"];
        assert_eq!(result, expected);
    }

    #[test]
    fn word_to_symbols_single_char() {
        let result = word_to_symbols("a").unwrap();
        let expected = vec!["a", "</w>"];
        assert_eq!(result, expected);
    }

    #[test]
    fn word_to_symbols_with_hyphen() {
        let result = word_to_symbols("co-op").unwrap();
        let expected = vec!["c", "o", "-", "o", "p", "</w>"];
        assert_eq!(result, expected);
    }

    #[test]
    fn word_to_symbols_empty() {
        let result = word_to_symbols("").unwrap();
        let expected = vec!["</w>"];
        assert_eq!(result, expected);
    }

    #[test]
    fn word_to_symbols_multiple_words_fails() {
        let result = word_to_symbols("hello world");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("multiple words"));
    }
}
