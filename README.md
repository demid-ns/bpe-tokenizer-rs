# BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer implementation in Rust with byte-level pre-tokenization (GPT-2 style). Fully compatible with HuggingFace tokenizers when configured with ByteLevel pre-tokenizer and decoder.

> [!IMPORTANT]
> This project is currently in active development. The API may change and features are still being implemented.

## Features

- Byte-level BPE tokenization (GPT-2 style)
- 100% compatible with HuggingFace tokenizers
- Special token support
- Encode/decode with merge rules
- Token ID-based tie-breaking for deterministic training

## Usage

```rust
use bpe_tokenizer_rs::{Trainer, BpeTokenizer};

// Train a tokenizer
let trainer = Trainer::new(100);
let training_texts = vec!["hello world", "hello rust"];
let merges = trainer.train(&training_texts);

// Create tokenizer with special tokens
let special_tokens = vec!["<|endoftext|>".to_string()];
let tokenizer = BpeTokenizer::new(merges, special_tokens);

// Encode and decode
let ids = tokenizer.encode("hello world");
let text = tokenizer.decode(&ids);
```

## Testing

```bash
cargo test
```