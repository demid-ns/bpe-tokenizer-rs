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

    pub fn encode(&self, text: &str) -> Vec<usize> {
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

    fn token_to_id(&self, token: &str) -> usize {
        self.vocabulary
            .token_to_id(token)
            .unwrap_or_else(|| panic!("Token '{}' not in vocabulary. This indicates vocabulary and merge rules are out of sync!", token))
    }
}
