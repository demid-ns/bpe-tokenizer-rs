# BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer implementation in Rust - created for learning purposes with focus on the core BPE training algorithm.

> [!IMPORTANT]
> This project is currently in active development. The API may change and features are still being implemented.

## How it works

The tokenizer implements the standard BPE training algorithm:

- Starts with character-level vocabulary for each word
- Iteratively finds the most frequent adjacent token pairs
- Merges the most frequent pairs into single tokens
- Continues for a specified number of merge operations
- Uses lexicographic tie-breaking for consistent results

## Usage

```rust
use bpe_tokenizer_rs::Trainer;

let trainer = Trainer::new(100); // 100 merge operations
let training_texts = vec!["hello world", "hello rust", "world peace"];
let vocab = trainer.train(&training_texts);

// The vocab contains the learned BPE tokens with their frequencies
for (tokens, count) in vocab {
    println!("{:?}: {}", tokens, count);
}
```

Run the tests:
```bash
cargo test
```

## Philosophy

Clean implementation focusing on the core BPE algorithm mechanics rather than comprehensive tokenization features. Designed for educational purposes and understanding how BPE tokenizers work under the hood.