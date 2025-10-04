use crate::word_to_symbols;

pub struct Encoder {
    merge_rules: Vec<(String, String)>,
}

impl Encoder {
    pub fn new(merge_rules: Vec<(String, String)>) -> Encoder {
        Self { merge_rules }
    }

    pub fn encode(&self, input: &str) -> Result<Vec<String>, String> {
        let tokens = Vec::new();

        let _symbols = word_to_symbols(input)?;

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
