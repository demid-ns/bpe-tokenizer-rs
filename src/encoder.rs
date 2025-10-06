use crate::word_to_symbols;

pub struct Encoder {
    merge_rules: Vec<(String, String)>,
}

impl Encoder {
    pub fn new(merge_rules: Vec<(String, String)>) -> Self {
        Encoder { merge_rules }
    }

    pub fn encode(&self, text: &str) -> Vec<Vec<String>> {
        text.split_whitespace()
            .filter_map(|word| word_to_symbols(word).ok())
            .map(|symbols| self.apply_merge_rules(symbols))
            .collect()
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
}

#[cfg(test)]
mod tests {
    use crate::test_utils::to_merges;

    use super::*;

    #[test]
    fn encode_correctly_applies_merge_rules() {
        let merge_rules = to_merges(&[
            ("n", "a"),
            ("na", "na"),
            ("nana", "</w>"),
            ("b", "a"),
            ("ba", "nana</w>"),
        ]);

        let encoder = Encoder::new(merge_rules);
        let result = encoder.encode("banana nanao");

        let expected = vec![vec!["banana</w>"], vec!["nana", "o", "</w>"]];

        assert_eq!(result, expected);
    }
}
