use bpe_tokenizer_rs::{BpeTokenizer, Trainer};

fn main() {
    println!("=== BPE Tokenizer Example ===\n");

    // Example 1: Basic tokenization without training
    println!("Example 1: Basic tokenization (no merges)");
    println!("-----------------------------------------");
    let tokenizer = BpeTokenizer::new(vec![], vec![]);

    let text = "Hello, world!";
    let ids = tokenizer.encode(text);
    let decoded = tokenizer.decode(&ids);

    println!("Original text: {}", text);
    println!("Token IDs: {:?}", ids);
    println!("Decoded text: {}", decoded);
    println!("Match: {}\n", text == decoded);

    // Example 2: Training a tokenizer
    println!("Example 2: Training a tokenizer");
    println!("--------------------------------");
    let trainer = Trainer::new(50);
    let training_data = vec![
        "The quick brown fox jumps over the lazy dog.",
        "The five boxing wizards jump quickly.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
    ];

    println!("Training on {} sentences with max {} merges...", training_data.len(), 50);
    let merges = trainer.train(&training_data);
    println!("Learned {} merge rules\n", merges.len());

    let trained_tokenizer = BpeTokenizer::new(merges.clone(), vec![]);

    let test_text = "The quick fox jumps";
    let trained_ids = trained_tokenizer.encode(test_text);
    let trained_decoded = trained_tokenizer.decode(&trained_ids);

    println!("Test text: {}", test_text);
    println!("Token count: {}", trained_ids.len());
    println!("Token IDs: {:?}", trained_ids);
    println!("Decoded: {}", trained_decoded);
    println!("Match: {}\n", test_text == trained_decoded);

    // Example 3: Special tokens
    println!("Example 3: Using special tokens");
    println!("--------------------------------");
    let special_tokens = vec![
        "<|endoftext|>".to_string(),
        "<|startoftext|>".to_string(),
        "[PAD]".to_string(),
    ];

    let tokenizer_with_special = BpeTokenizer::new(merges.clone(), special_tokens.clone());

    let special_text = "<|startoftext|>Hello, world!<|endoftext|>";
    let special_ids = tokenizer_with_special.encode(special_text);
    let special_decoded = tokenizer_with_special.decode(&special_ids);

    println!("Special tokens: {:?}", special_tokens);
    println!("Text with special tokens: {}", special_text);
    println!("Token IDs: {:?}", special_ids);
    println!("Decoded: {}", special_decoded);
    println!("Match: {}\n", special_text == special_decoded);

    // Example 4: Multilingual text
    println!("Example 4: Multilingual text");
    println!("----------------------------");
    let multilingual_texts = vec![
        "Hello world",
        "Привет мир",
        "你好世界",
        "こんにちは世界",
        "🦀 Rust",
    ];

    println!("Encoding multilingual texts:");
    for text in &multilingual_texts {
        let ids = trained_tokenizer.encode(text);
        let decoded = trained_tokenizer.decode(&ids);
        println!("  '{}' -> {} tokens -> '{}'", text, ids.len(), decoded);
        assert_eq!(*text, decoded, "Roundtrip failed!");
    }
    println!();

    // Example 5: Demonstrating merge efficiency
    println!("Example 5: Merge efficiency comparison");
    println!("---------------------------------------");
    let comparison_text = "hello hello hello world world";

    let no_merge_tokenizer = BpeTokenizer::new(vec![], vec![]);
    let no_merge_ids = no_merge_tokenizer.encode(comparison_text);

    let with_merge_ids = trained_tokenizer.encode(comparison_text);

    println!("Text: {}", comparison_text);
    println!("Without merges: {} tokens", no_merge_ids.len());
    println!("With merges: {} tokens", with_merge_ids.len());
    println!("Compression: {:.1}%\n",
             (1.0 - with_merge_ids.len() as f64 / no_merge_ids.len() as f64) * 100.0);

    // Example 6: Using from_trainer convenience method
    println!("Example 6: Using from_trainer()");
    println!("--------------------------------");
    let quick_trainer = Trainer::new(20);
    let quick_data = vec!["Rust is fast", "Rust is safe", "Rust is fun"];

    let quick_tokenizer = BpeTokenizer::from_trainer(
        &quick_trainer,
        &quick_data,
        vec![]
    );

    let quick_text = "Rust is awesome";
    let quick_ids = quick_tokenizer.encode(quick_text);

    println!("Trained on: {:?}", quick_data);
    println!("Test text: {}", quick_text);
    println!("Token count: {}", quick_ids.len());
    println!("Decoded: {}", quick_tokenizer.decode(&quick_ids));

    println!("\n=== All examples completed successfully! ===");
}
