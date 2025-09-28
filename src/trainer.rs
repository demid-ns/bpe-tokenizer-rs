use std::collections::HashMap;

pub struct Trainer {
    num_merges: usize,
}

impl Trainer {
    pub fn new(num_merges: usize) -> Trainer {
        Self { num_merges }
    }

    pub fn train(&self, training_texts: &[&str]) -> HashMap<Vec<String>, usize> {
        let mut vocab = Self::get_vocab(training_texts);

        for _ in 1..=self.num_merges {
            let pair_counts = Self::get_pair_counts(&vocab);

            if let Some(most_frequent) = Self::get_most_frequent_pair(&pair_counts) {
                vocab = Self::merge_most_frequent_pair(&vocab, &most_frequent);
            } else {
                break;
            }
        }

        vocab
    }

    fn get_vocab(training_texts: &[&str]) -> HashMap<Vec<String>, usize> {
        let mut vocab = HashMap::new();

        training_texts
            .iter()
            .flat_map(|text| text.split_whitespace())
            .map(|word| {
                let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                tokens.push("</w>".to_string());
                tokens
            })
            .for_each(|tokens| {
                *vocab.entry(tokens).or_insert(0) += 1;
            });

        vocab
    }

    fn get_pair_counts(vocab: &HashMap<Vec<String>, usize>) -> HashMap<(String, String), usize> {
        let mut pair_counts = HashMap::new();

        vocab.iter().for_each(|(tokens, &count)| {
            tokens.windows(2).for_each(|pair| {
                let key = (pair[0].clone(), pair[1].clone());
                *pair_counts.entry(key).or_insert(0) += count;
            });
        });

        pair_counts
    }

    fn get_most_frequent_pair(
        pair_counts: &HashMap<(String, String), usize>,
    ) -> Option<(String, String)> {
        pair_counts
            .iter()
            .max_by(|(pair_a, count_a), (pair_b, count_b)| {
                match count_a.cmp(count_b) {
                    std::cmp::Ordering::Equal => pair_a.cmp(pair_b), // pick lexicographically smallest
                    other => other,
                }
            })
            .map(|(pair, _)| pair.clone())
    }

    fn merge_most_frequent_pair(
        vocab: &HashMap<Vec<String>, usize>,
        pair: &(String, String),
    ) -> HashMap<Vec<String>, usize> {
        let mut merged_vocab = HashMap::new();
        let pair_to_merge = format!("{}{}", pair.0, pair.1);

        for (tokens, &count) in vocab {
            let mut i = 0;
            let mut merged_tokens = Vec::new();

            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    merged_tokens.push(pair_to_merge.clone());
                    i += 2;
                } else {
                    merged_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            *merged_vocab.entry(merged_tokens).or_insert(0) += count;
        }

        merged_vocab
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn train_no_merges() {
        let trainer = Trainer::new(0);
        let result = trainer.train(&vec!["banana banana nana", "banana"]);

        let expected: HashMap<Vec<String>, usize> =
            HashMap::from([(to_tokens("banana"), 3), (to_tokens("nana"), 1)]);

        assert_eq!(result, expected);
    }

    #[test]
    fn train_with_merges() {
        let trainer = Trainer::new(2);
        let result = trainer.train(&vec!["banana banana nana", "banana"]);

        let expected: HashMap<Vec<String>, usize> = HashMap::from([
            (to_vec_tokens(&["b", "a", "nana", "</w>"]), 3),
            (to_vec_tokens(&["nana", "</w>"]), 1),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn train_with_max_merges() {
        let trainer = Trainer::new(10);
        let result = trainer.train(&vec!["banana banana nana", "banana"]);

        let expected: HashMap<Vec<String>, usize> = HashMap::from([
            (to_vec_tokens(&["banana</w>"]), 3),
            (to_vec_tokens(&["nana</w>"]), 1),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn get_vocab_two_texts() {
        let result = Trainer::get_vocab(&vec!["banana banana nana", "banana"]);

        let expected: HashMap<Vec<String>, usize> =
            HashMap::from([(to_tokens("banana"), 3), (to_tokens("nana"), 1)]);

        assert_eq!(result, expected);
    }

    #[test]
    fn get_vocab_punctuation() {
        let result = Trainer::get_vocab(&vec!["banana ,! banana,", "banana"]);

        let expected: HashMap<Vec<String>, usize> = HashMap::from([
            (to_tokens("banana"), 2),
            (to_tokens("banana,"), 1),
            (to_tokens(",!"), 1),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn get_vocab_empty() {
        let result = Trainer::get_vocab(&vec![]);

        let expected: HashMap<Vec<String>, usize> = HashMap::new();

        assert_eq!(result, expected);
    }

    fn to_tokens(word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        tokens.push("</w>".to_string());
        tokens
    }

    #[test]
    fn get_pair_counts_some() {
        let vocab = Trainer::get_vocab(&vec!["banana banana nana", "banana"]);

        let pair_counts = Trainer::get_pair_counts(&vocab);

        let expected_result = HashMap::from([
            (to_pair_count(("a", "n"), 7)),
            (to_pair_count(("a", "</w>"), 4)),
            (to_pair_count(("n", "a"), 8)),
            (to_pair_count(("b", "a"), 3)),
        ]);

        assert_eq!(pair_counts, expected_result);
    }

    #[test]
    fn get_pair_counts_empty() {
        let vocab = Trainer::get_vocab(&vec![]);

        let pair_counts = Trainer::get_pair_counts(&vocab);

        let expected_result = HashMap::new();

        assert_eq!(pair_counts, expected_result);
    }

    fn to_pair_count(pair: (&str, &str), count: usize) -> ((String, String), usize) {
        ((pair.0.to_string(), pair.1.to_string()), count)
    }

    #[test]
    fn get_most_frequent_pair_some() {
        let vocab = Trainer::get_vocab(&vec!["banana banana nana", "banana"]);

        let pair_counts = Trainer::get_pair_counts(&vocab);

        let result = Trainer::get_most_frequent_pair(&pair_counts);

        assert_eq!(result, Some(("n".to_string(), "a".to_string())));
    }

    #[test]
    fn get_most_frequent_pair_none() {
        let most_frequent_pair = Trainer::get_most_frequent_pair(&HashMap::new());
        assert_eq!(most_frequent_pair, None);
    }

    #[test]
    fn get_most_frequent_pair_peaks_lexicographically_smallest() {
        let mut pair_counts = HashMap::new();
        pair_counts.insert(("z".to_string(), "a".to_string()), 3);
        pair_counts.insert(("a".to_string(), "b".to_string()), 3);
        pair_counts.insert(("c".to_string(), "d".to_string()), 3);

        let result = Trainer::get_most_frequent_pair(&pair_counts);

        assert_eq!(result, Some(("z".to_string(), "a".to_string())));
    }

    #[test]
    fn merge_most_frequent_pair_some() {
        let vocab = Trainer::get_vocab(&vec!["banana banana nana", "banana"]);

        let pair_counts = Trainer::get_pair_counts(&vocab);
        let most_frequent_pair = Trainer::get_most_frequent_pair(&pair_counts);

        let result = Trainer::merge_most_frequent_pair(&vocab, &most_frequent_pair.unwrap());

        let expected_result = HashMap::from([
            (to_vec_tokens(&["b", "a", "na", "na", "</w>"]), 3),
            (to_vec_tokens(&["na", "na", "</w>"]), 1),
        ]);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn merge_most_frequent_pair_punctuation() {
        let vocab = Trainer::get_vocab(&vec!["banana , , , , banana nana , ,!", "banana!!!"]);

        let pair_counts = Trainer::get_pair_counts(&vocab);
        let most_frequent_pair = Trainer::get_most_frequent_pair(&pair_counts);

        let result = Trainer::merge_most_frequent_pair(&vocab, &most_frequent_pair.unwrap());

        let expected_result = HashMap::from([
            (
                to_vec_tokens(&["b", "a", "na", "na", "!", "!", "!", "</w>"]),
                1,
            ),
            (to_vec_tokens(&[",", "!", "</w>"]), 1),
            (to_vec_tokens(&["na", "na", "</w>"]), 1),
            (to_vec_tokens(&[",", "</w>"]), 5),
            (to_vec_tokens(&["b", "a", "na", "na", "</w>"]), 2),
        ]);

        assert_eq!(result, expected_result);
    }

    fn to_vec_tokens(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|t| t.to_string()).collect()
    }
}
