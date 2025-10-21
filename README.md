# BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer implementation in Rust with byte-level pre-tokenization (GPT-2 style).

> [!IMPORTANT]
> **Educational Project**: This is an educational implementation created to understand BPE tokenization internals. While functional and well-tested, it is **not intended for production use**. For production applications, use [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) or other battle-tested libraries.

> [!NOTE]
> **HuggingFace Compatibility**: This tokenizer is compatible with HuggingFace tokenizers when configured with:
> - `ByteLevel` pre-tokenizer (with `add_prefix_space=false`)
> - `ByteLevel` decoder
> - No normalizer or post-processor
>
> See the `tests/huggingface_compatibility.rs` file for exact configuration details. This is **not** a general drop-in replacement for HuggingFace tokenizers.

## Features

- Byte-level BPE tokenization (GPT-2 style)
- Special token support (`<|endoftext|>`, `[PAD]`, etc.)
- Training from scratch with configurable merge count
- Deterministic training with token ID-based tie-breaking
- Full encode/decode roundtrip support for all Unicode text
- Compatible with specific HuggingFace tokenizer configurations

## Quick Start

### Basic Usage

```rust
use bpe_tokenizer_rs::BpeTokenizer;

// Create a tokenizer (no training)
let tokenizer = BpeTokenizer::new(vec![], vec![]);

// Encode and decode
let ids = tokenizer.encode("Hello, world!");
let text = tokenizer.decode(&ids);
assert_eq!(text, "Hello, world!");
```

### Training a Tokenizer

```rust
use bpe_tokenizer_rs::{Trainer, BpeTokenizer};

// Train on your data
let trainer = Trainer::new(100);  // 100 merge rules
let training_data = &["hello world", "hello rust", "rust is fast"];
let merges = trainer.train(training_data);

// Create tokenizer with learned merges
let tokenizer = BpeTokenizer::new(merges, vec![]);

// Or use the convenience method
let tokenizer = BpeTokenizer::from_trainer(&trainer, training_data, vec![]);
```

### With Special Tokens

```rust
use bpe_tokenizer_rs::BpeTokenizer;

let special_tokens = vec!["<|endoftext|>".to_string()];
let tokenizer = BpeTokenizer::new(vec![], special_tokens);

let text = "<|endoftext|>Hello!";
let ids = tokenizer.encode(text);
let decoded = tokenizer.decode(&ids);
```

## Examples

Run the comprehensive example:

```bash
cargo run --example runner
```

This demonstrates:
- Basic tokenization
- Training a tokenizer
- Using special tokens
- Multilingual text support
- Compression efficiency
- And more!

## Testing

```bash
# Run all tests
cargo test

# Run only unit tests
cargo test --lib

# Run HuggingFace compatibility tests
cargo test --test huggingface_compatibility

# Generate and view documentation
cargo doc --open
```

## Project Structure

```
src/
├── lib.rs              # Public API exports
├── tokenizer.rs        # Main BpeTokenizer struct
├── encoder.rs          # Text → token IDs
├── decoder.rs          # Token IDs → text
├── trainer.rs          # BPE training algorithm
├── vocabulary.rs       # Token ↔ ID mapping
├── pre_tokenizer.rs    # GPT-2 style text splitting
└── byte_encoder.rs     # Byte-level encoding utilities

tests/
└── huggingface_compatibility.rs  # HF compatibility tests

examples/
└── runner.rs           # Comprehensive usage examples
```