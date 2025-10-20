use bpe_tokenizer_rs::{BpeTokenizer, Trainer};
use std::fs;
use std::io::Write;
use tempfile::TempDir;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::{AddedToken, Tokenizer, TokenizerBuilder};

fn train_hf_tokenizer(
    training_texts: &[&str],
    num_merges: usize,
    special_tokens: Vec<String>,
) -> Tokenizer {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("train.txt");

    let mut file = fs::File::create(&file_path).unwrap();
    for text in training_texts {
        writeln!(file, "{}", text).unwrap();
    }
    drop(file);

    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(256 + num_merges + special_tokens.len())
        .min_frequency(0)
        .show_progress(false)
        .special_tokens(
            special_tokens
                .into_iter()
                .map(|s| AddedToken::from(s, true))
                .collect(),
        )
        .initial_alphabet(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
                .into_iter()
                .collect(),
        )
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::default().add_prefix_space(false),
        ))
        .with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()))
        .with_normalizer(None::<tokenizers::normalizers::Sequence>)
        .with_post_processor(None::<tokenizers::processors::sequence::Sequence>)
        .build()
        .unwrap();

    tokenizer
        .train_from_files(&mut trainer, vec![file_path.to_str().unwrap().to_string()])
        .unwrap();

    tokenizer.into()
}

fn create_tokenizers_without_merges(training_data: &[&str]) -> (BpeTokenizer, Tokenizer) {
    let trainer = Trainer::new(0);
    let our = BpeTokenizer::from_trainer(&trainer, training_data, vec![]);
    let hf = train_hf_tokenizer(training_data, 0, vec![]);
    (our, hf)
}

fn create_tokenizers_with_merges(
    training_data: &[&str],
    num_merges: usize,
) -> (BpeTokenizer, Tokenizer) {
    let trainer = Trainer::new(num_merges);
    let our = BpeTokenizer::from_trainer(&trainer, training_data, vec![]);
    let hf = train_hf_tokenizer(training_data, num_merges, vec![]);
    (our, hf)
}

fn create_tokenizers_with_special_tokens(
    training_data: &[&str],
    special_tokens: Vec<String>,
) -> (BpeTokenizer, Tokenizer) {
    let trainer = Trainer::new(0);
    let merges = trainer.train(training_data);
    let our = BpeTokenizer::new(merges, special_tokens.clone());
    let hf = train_hf_tokenizer(training_data, 0, special_tokens);
    (our, hf)
}

fn assert_encoding_matches(our: &BpeTokenizer, hf: &Tokenizer, text: &str) {
    let our_ids = our.encode(text);
    let hf_ids = hf.encode(text, false).unwrap().get_ids().to_vec();
    assert_eq!(our_ids, hf_ids);
}

#[test]
fn empty_string_matches_hf() {
    let (our, hf) = create_tokenizers_without_merges(&[""]);
    assert_encoding_matches(&our, &hf, "");
}

#[test]
fn single_ascii_char_matches_hf() {
    let (our, hf) = create_tokenizers_without_merges(&[""]);
    assert_encoding_matches(&our, &hf, "A");
}

#[test]
fn multiple_ascii_chars_match_hf() {
    let (our, hf) = create_tokenizers_without_merges(&[""]);
    assert_encoding_matches(&our, &hf, "Hello");
}

#[test]
fn utf8_two_bytes_matches_hf() {
    let (our, hf) = create_tokenizers_without_merges(&[""]);
    assert_encoding_matches(&our, &hf, "é");
}

#[test]
fn japanese_matches_hf() {
    let (our, hf) = create_tokenizers_without_merges(&[""]);
    assert_encoding_matches(&our, &hf, "日本");
}

#[test]
fn emoji_matches_hf() {
    let (our, hf) = create_tokenizers_without_merges(&[""]);
    assert_encoding_matches(&our, &hf, "🦀");
}

#[test]
fn with_single_merge_matches_hf() {
    let (our, hf) = create_tokenizers_with_merges(&["aa aa aa"], 1);
    assert_encoding_matches(&our, &hf, "aa");
}

#[test]
fn with_multiple_merges_matches_hf() {
    let (our, hf) = create_tokenizers_with_merges(&["hello hello hello world world"], 5);
    assert_encoding_matches(&our, &hf, "hello world");
}

#[test]
fn with_special_token_matches_hf() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let (our, hf) = create_tokenizers_with_special_tokens(&["hello world"], special_tokens);
    assert_encoding_matches(&our, &hf, "<|endoftext|>");
}

#[test]
fn special_token_with_text_matches_hf() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let (our, hf) = create_tokenizers_with_special_tokens(&["hello world"], special_tokens);
    assert_encoding_matches(&our, &hf, "<|endoftext|>hello");
}

#[test]
fn multiple_special_tokens_match_hf() {
    let special_tokens = vec!["<|start|>".to_string(), "<|end|>".to_string()];
    let (our, hf) = create_tokenizers_with_special_tokens(&["test"], special_tokens);
    assert_encoding_matches(&our, &hf, "<|start|>test<|end|>");
}

#[test]
fn roundtrip_matches_hf() {
    let (our, hf) = create_tokenizers_with_merges(&["Hello world 世界"], 10);
    let text = "Hello world";

    let our_ids = our.encode(text);
    let hf_ids = hf.encode(text, false).unwrap().get_ids().to_vec();
    assert_eq!(our_ids, hf_ids);

    let our_decoded = our.decode(&our_ids);
    let hf_decoded = hf.decode(&hf_ids, false).unwrap();
    assert_eq!(our_decoded, hf_decoded);
    assert_eq!(our_decoded, text);
}

#[test]
fn chinese_with_merge_matches_hf() {
    let (our, hf) = create_tokenizers_with_merges(&["世界 世界 世界"], 1);
    assert_encoding_matches(&our, &hf, "世界");
}

#[test]
fn russian_with_merge_matches_hf() {
    let (our, hf) = create_tokenizers_with_merges(&["Привет Привет Привет"], 1);
    assert_encoding_matches(&our, &hf, "Привет");
}

#[test]
fn complex_text_matches_hf() {
    let (our, hf) =
        create_tokenizers_with_merges(&["Hello, world! How are you?", "I'm fine, thanks!"], 20);
    assert_encoding_matches(&our, &hf, "Hello, I'm fine!");
}

#[test]
fn gpt2_contractions_match_hf() {
    let (our, hf) = create_tokenizers_with_merges(&["don't won't can't"], 5);
    assert_encoding_matches(&our, &hf, "don't");
}

#[test]
fn numbers_match_hf() {
    let (our, hf) = create_tokenizers_with_merges(&["123 456 789"], 3);
    assert_encoding_matches(&our, &hf, "123");
}

#[test]
fn mixed_content_matches_hf() {
    let (our, hf) = create_tokenizers_with_merges(&["Hello 世界 123 🦀"], 10);
    assert_encoding_matches(&our, &hf, "Hello 世界");
}

#[test]
fn complex_text_with_special_tokens_matches_hf() {
    let special_tokens = vec!["<|begin|>".to_string(), "<|end|>".to_string()];
    let training_data = vec![
        "Hello, world! 你好世界",
        "Привет, мир! 123",
        "🦀 Rust is great!",
    ];

    let trainer = Trainer::new(15);
    let merges = trainer.train(&training_data);
    let our = BpeTokenizer::new(merges.clone(), special_tokens.clone());
    let hf = train_hf_tokenizer(&training_data, 15, special_tokens);

    let test_cases = vec![
        "<|begin|>Hello, world!<|end|>",
        "<|begin|>你好<|end|>",
        "Привет<|end|>123<|begin|>",
        "<|begin|>🦀<|end|>",
    ];

    for text in test_cases {
        assert_encoding_matches(&our, &hf, text);
    }
}
