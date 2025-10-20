use crate::{Decoder, Encoder, PreTokenizer, Trainer, Vocabulary};

pub struct BpeTokenizer {
    encoder: Encoder,
    decoder: Decoder,
}

impl BpeTokenizer {
    pub fn new(merges: Vec<(String, String)>, special_tokens: Vec<String>) -> Self {
        let pre_tokenizer = PreTokenizer::new();
        let vocabulary = Vocabulary::new(special_tokens.clone(), merges.clone());
        let encoder = Encoder::new(merges, pre_tokenizer, vocabulary.clone(), special_tokens);
        let decoder = Decoder::new(vocabulary);

        BpeTokenizer { encoder, decoder }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encoder.encode(text)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.decoder.decode(ids)
    }

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

        assert_eq!(ids, vec![162, 151, 165]);
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

        let text = tokenizer.decode(&[162, 151, 165]);

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

        assert_eq!(ids, vec![228, 184, 256, 149, 140]);
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

        assert_eq!(ids, vec![208, 256, 128, 208, 184, 208, 178, 208, 181, 209, 130]);
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
