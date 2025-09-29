pub fn word_to_symbols(word: &str) -> Vec<String> {
    let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    symbols.push("</w>".to_string());
    symbols
}
