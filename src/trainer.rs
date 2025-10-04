use crate::word_to_symbols;
use std::collections::HashMap;

pub struct Trainer {
    num_merges: usize,
}

impl Trainer {
    pub fn new(num_merges: usize) -> Trainer {
        Self { num_merges }
    }

    pub fn train(&self, training_texts: &[&str]) -> Vec<(String, String)> {
        let mut merges = Vec::new();
        let mut word_freqs = Self::build_word_frequencies(training_texts);

        for _ in 1..=self.num_merges {
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

    fn build_word_frequencies(training_texts: &[&str]) -> HashMap<Vec<String>, usize> {
        let mut word_freqs = HashMap::new();

        training_texts
            .iter()
            .flat_map(|text| text.split_whitespace())
            .filter_map(|word| word_to_symbols(word).ok())
            .for_each(|symbols| {
                *word_freqs.entry(symbols).or_insert(0) += 1;
            });

        word_freqs
    }

    fn get_pair_frequencies(
        word_freqs: &HashMap<Vec<String>, usize>,
    ) -> HashMap<(String, String), usize> {
        let mut pair_freqs = HashMap::new();

        word_freqs.iter().for_each(|(symbols, &count)| {
            symbols.windows(2).for_each(|pair| {
                let key = (pair[0].clone(), pair[1].clone());
                *pair_freqs.entry(key).or_insert(0) += count;
            });
        });

        pair_freqs
    }

    fn get_most_common_pair(
        pair_freqs: &HashMap<(String, String), usize>,
    ) -> Option<(String, String)> {
        pair_freqs
            .iter()
            .max_by(|(pair_a, count_a), (pair_b, count_b)| {
                match count_a.cmp(count_b) {
                    std::cmp::Ordering::Equal => pair_a.cmp(pair_b), // pick lexicographically smallest
                    other => other,
                }
            })
            .map(|(pair, _)| pair.clone())
    }

    fn merge_pair(
        word_freqs: &HashMap<Vec<String>, usize>,
        pair: &(String, String),
    ) -> HashMap<Vec<String>, usize> {
        let mut merged_word_freqs = HashMap::new();
        let merged_symbol = format!("{}{}", pair.0, pair.1);

        for (symbols, &count) in word_freqs {
            let mut i = 0;
            let mut merged_symbols = Vec::new();

            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == pair.0 && symbols[i + 1] == pair.1 {
                    merged_symbols.push(merged_symbol.clone());
                    i += 2;
                } else {
                    merged_symbols.push(symbols[i].clone());
                    i += 1;
                }
            }

            *merged_word_freqs.entry(merged_symbols).or_insert(0) += count;
        }

        merged_word_freqs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_no_merges_returns_empty() {
        let trainer = Trainer::new(0);
        let result = trainer.train(&vec!["banana banana nana", "banana"]);

        assert!(result.is_empty());
    }

    #[test]
    fn train_with_merges_applies_merges_correctly() {
        let trainer = Trainer::new(2);
        let result = trainer.train(&vec!["banana banana nana", "banana"]);

        let expected = to_merges(&[("n", "a"), ("na", "na")]);

        assert_eq!(result, expected);
    }

    #[test]
    fn train_with_max_merges_merges_all_possible() {
        let trainer = Trainer::new(10);
        let result = trainer.train(&vec!["banana banana nana", "banana"]);

        let expected = to_merges(&[
            ("n", "a"),
            ("na", "na"),
            ("nana", "</w>"),
            ("b", "a"),
            ("ba", "nana</w>"),
        ]);

        assert_eq!(result, expected);
    }

    fn to_merges(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    #[test]
    fn build_word_frequencies_two_texts() {
        let result = Trainer::build_word_frequencies(&vec!["banana banana nana", "banana"]);

        let expected: HashMap<Vec<String>, usize> =
            HashMap::from([(to_symbols("banana"), 3), (to_symbols("nana"), 1)]);

        assert_eq!(result, expected);
    }

    #[test]
    fn build_word_frequencies_with_punctuation() {
        let result = Trainer::build_word_frequencies(&vec!["banana ,! banana,", "banana"]);

        let expected: HashMap<Vec<String>, usize> = HashMap::from([
            (to_symbols("banana"), 2),
            (to_symbols("banana,"), 1),
            (to_symbols(",!"), 1),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn build_word_frequencies_empty_input() {
        let result = Trainer::build_word_frequencies(&vec![]);

        let expected: HashMap<Vec<String>, usize> = HashMap::new();

        assert_eq!(result, expected);
    }

    fn to_symbols(word: &str) -> Vec<String> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        symbols.push("</w>".to_string());
        symbols
    }

    #[test]
    fn get_pair_frequencies_some_pairs() {
        let word_freqs = Trainer::build_word_frequencies(&vec!["banana banana nana", "banana"]);

        let pair_freqs = Trainer::get_pair_frequencies(&word_freqs);

        let expected_result = HashMap::from([
            (to_pair_count(("a", "n"), 7)),
            (to_pair_count(("a", "</w>"), 4)),
            (to_pair_count(("n", "a"), 8)),
            (to_pair_count(("b", "a"), 3)),
        ]);

        assert_eq!(pair_freqs, expected_result);
    }

    #[test]
    fn get_pair_frequencies_empty() {
        let word_freqs = Trainer::build_word_frequencies(&vec![]);

        let pair_freqs = Trainer::get_pair_frequencies(&word_freqs);

        let expected_result = HashMap::new();

        assert_eq!(pair_freqs, expected_result);
    }

    fn to_pair_count(pair: (&str, &str), count: usize) -> ((String, String), usize) {
        ((pair.0.to_string(), pair.1.to_string()), count)
    }

    #[test]
    fn get_most_common_pair_some() {
        let word_freqs = Trainer::build_word_frequencies(&vec!["banana banana nana", "banana"]);

        let pair_freqs = Trainer::get_pair_frequencies(&word_freqs);

        let result = Trainer::get_most_common_pair(&pair_freqs);

        assert_eq!(result, Some(("n".to_string(), "a".to_string())));
    }

    #[test]
    fn get_most_common_pair_none() {
        let most_common_pair = Trainer::get_most_common_pair(&HashMap::new());
        assert_eq!(most_common_pair, None);
    }

    #[test]
    fn get_most_common_pair_breaks_tie_lexicographically() {
        let mut pair_freqs = HashMap::new();
        pair_freqs.insert(("z".to_string(), "a".to_string()), 3);
        pair_freqs.insert(("a".to_string(), "b".to_string()), 3);
        pair_freqs.insert(("c".to_string(), "d".to_string()), 3);

        let result = Trainer::get_most_common_pair(&pair_freqs);

        assert_eq!(result, Some(("z".to_string(), "a".to_string())));
    }

    #[test]
    fn merge_pair_some() {
        let word_freqs = Trainer::build_word_frequencies(&vec!["banana banana nana", "banana"]);

        let pair_freqs = Trainer::get_pair_frequencies(&word_freqs);
        let most_common_pair = Trainer::get_most_common_pair(&pair_freqs);

        let result = Trainer::merge_pair(&word_freqs, &most_common_pair.unwrap());

        let expected_result = HashMap::from([
            (to_vec_symbols(&["b", "a", "na", "na", "</w>"]), 3),
            (to_vec_symbols(&["na", "na", "</w>"]), 1),
        ]);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn merge_pair_with_punctuation() {
        let word_freqs =
            Trainer::build_word_frequencies(&vec!["banana , , , , banana nana , ,!", "banana!!!"]);

        let pair_freqs = Trainer::get_pair_frequencies(&word_freqs);
        let most_common_pair = Trainer::get_most_common_pair(&pair_freqs);

        let result = Trainer::merge_pair(&word_freqs, &most_common_pair.unwrap());

        let expected_result = HashMap::from([
            (
                to_vec_symbols(&["b", "a", "na", "na", "!", "!", "!", "</w>"]),
                1,
            ),
            (to_vec_symbols(&[",", "!", "</w>"]), 1),
            (to_vec_symbols(&["na", "na", "</w>"]), 1),
            (to_vec_symbols(&[",", "</w>"]), 5),
            (to_vec_symbols(&["b", "a", "na", "na", "</w>"]), 2),
        ]);

        assert_eq!(result, expected_result);
    }

    fn to_vec_symbols(symbols: &[&str]) -> Vec<String> {
        symbols.iter().map(|t| t.to_string()).collect()
    }
}
