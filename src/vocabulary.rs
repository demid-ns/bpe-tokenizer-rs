use std::collections::HashMap;

use crate::bytes_to_unicode;

/// Manages bidirectional mapping between tokens and their IDs.
///
/// Uses two data structures for O(1) lookups in both directions:
/// - `token_to_id`: HashMap for fast token → ID lookup (used during encoding)
/// - `id_to_token`: Vec for fast ID → token lookup (used during decoding)
///
/// We use Vec instead of HashMap<usize, String> for `id_to_token` because IDs are
/// sequential (0, 1, 2, ...), making Vec index access more efficient than hash lookup.
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
}

impl Vocabulary {
    pub fn new(merges: Vec<(String, String)>) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        let byte_encoder = bytes_to_unicode();

        for byte in 0u8..=255 {
            let ch = byte_encoder[&byte];
            let token = ch.to_string();

            id_to_token.push(token.clone());
            token_to_id.insert(token, byte as usize);
        }

        for (part1, part2) in merges {
            let token = format!("{}{}", part1, part2);
            let id = id_to_token.len();

            id_to_token.push(token.clone());
            token_to_id.insert(token, id);
        }

        Vocabulary {
            token_to_id,
            id_to_token,
        }
    }

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }
}
