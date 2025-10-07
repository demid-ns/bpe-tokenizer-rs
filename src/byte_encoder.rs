use std::collections::HashMap;

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
