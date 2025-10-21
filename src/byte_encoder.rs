use std::collections::HashMap;

/// Creates a mapping from bytes (0-255) to Unicode characters.
///
/// This is GPT-2's byte-level encoding scheme that maps all 256 possible byte values
/// to valid Unicode characters. The mapping ensures that:
/// - Printable ASCII characters (33-126, 161-172, 174-255) map to themselves
/// - Control characters and other bytes map to unused Unicode points (256-355)
///
/// This allows treating arbitrary byte sequences as valid Unicode strings during
/// tokenization, which is essential for byte-level BPE.
///
/// # Returns
///
/// A HashMap mapping each byte value (0-255) to its corresponding Unicode character.
///
/// # Examples
///
/// ```
/// use bpe_tokenizer_rs::bytes_to_unicode;
///
/// let mapping = bytes_to_unicode();
/// assert_eq!(mapping[&65], 'A');  // ASCII 'A' maps to itself
/// assert_eq!(mapping[&0], 'Ā');   // Byte 0 maps to Unicode 256
/// ```
pub fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut byte_to_char = HashMap::new();
    let mut n = 0u32;

    for b in 0u8..=255u8 {
        if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
            byte_to_char.insert(b, b as char);
        } else {
            byte_to_char.insert(b, char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    byte_to_char
}

/// Creates a mapping from Unicode characters back to bytes.
///
/// This is the inverse of `bytes_to_unicode()`, used during decoding to convert
/// the Unicode representation back to the original byte values.
///
/// # Returns
///
/// A HashMap mapping each Unicode character to its corresponding byte value (0-255).
///
/// # Examples
///
/// ```
/// use bpe_tokenizer_rs::unicode_to_bytes;
///
/// let mapping = unicode_to_bytes();
/// assert_eq!(mapping[&'A'], 65);  // 'A' maps back to ASCII 65
/// assert_eq!(mapping[&'Ā'], 0);   // Unicode 256 maps back to byte 0
/// ```
pub fn unicode_to_bytes() -> HashMap<char, u8> {
    let mut byte_to_char = HashMap::new();
    let mut n = 0u32;

    for b in 0u8..=255u8 {
        if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
            byte_to_char.insert(b as char, b);
        } else {
            byte_to_char.insert(char::from_u32(256 + n).unwrap(), b);
            n += 1;
        }
    }

    byte_to_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_to_unicode_return_correct_values() {
        let mapping = bytes_to_unicode();
        assert_eq!(mapping.get(&65), Some(&'A'));
        assert_eq!(mapping.get(&66), Some(&'B'));
        assert_eq!(mapping.get(&0), Some(&'Ā'));
        assert_eq!(mapping.get(&10), Some(&'Ċ'));
        assert_eq!(mapping.get(&255), Some(&'ÿ'));
    }

    #[test]
    fn unicode_to_bytes_return_correct_values() {
        let mapping = unicode_to_bytes();
        assert_eq!(mapping.get(&'A'), Some(&65));
        assert_eq!(mapping.get(&'B'), Some(&66));
        assert_eq!(mapping.get(&'Ā'), Some(&0));
        assert_eq!(mapping.get(&'Ċ'), Some(&10));
        assert_eq!(mapping.get(&'ÿ'), Some(&255));
    }
}
