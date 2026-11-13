//! Server-side tokenizers: turning text into a vocabulary's token ids inside PostgreSQL.
//!
//! A vocabulary may or may not have one. [`crate::registry::tokenizer`] is the single lookup the
//! write paths ask, by vocabulary id: text input, the `text -> tokens` cast, comparing a column
//! with a text literal, and `pgtoken.tokenize`. None of them knows which tokenizer it gets. Today
//! the only implementation is the o200k preset's; a later one (a Hugging Face `tokenizer.json`
//! stored as an artefact function, like a ranking) plugs into the same lookup.
//!
//! Token ids remain what a column stores and what `pg_dump` writes, so a dump never needs a
//! tokenizer to restore.

use fancy_regex::Regex;
use rustc_hash::FxHashMap;

use pgtoken_core::tables::ByteMap;

/// Text to token ids, under one vocabulary.
pub trait Tokenizer {
    /// Tokenize `text`. Special tokens are not recognised: `<|endoftext|>` in the text is plain
    /// text, as in tiktoken's `encode_ordinary`.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String>;
}

/// A tiktoken-style byte-level BPE: a pre-tokenizer regex splits the text into pieces, and each
/// piece is merged pair by pair, lowest rank first.
///
/// Built from the vocabulary's own `token_id -> bytes` mapping, because for tiktoken encodings a
/// token's id *is* its merge rank. So the preset ships no second copy of its vocabulary: the
/// mapping `pgtoken.text` already uses is the whole merge table.
pub struct TiktokenBpe {
    ranks: FxHashMap<Vec<u8>, u32>,
    pattern: Regex,
}

impl TiktokenBpe {
    /// `mergeable` bounds the ids that take part in merges: special tokens sit above it and are
    /// left out, so text can never turn into one.
    pub fn from_mapping(map: &ByteMap, mergeable: u32, pattern: &str) -> Result<Self, String> {
        let mut ranks = FxHashMap::default();
        ranks.reserve(mergeable as usize);
        for id in 0..mergeable {
            if let Some(bytes) = map.get(id) {
                ranks.insert(bytes.to_vec(), id);
            }
        }
        let pattern = Regex::new(pattern).map_err(|e| format!("bad pre-tokenizer regex: {e}"))?;
        Ok(TiktokenBpe { ranks, pattern })
    }

    fn rank(&self, bytes: &[u8]) -> u32 {
        self.ranks.get(bytes).copied().unwrap_or(u32::MAX)
    }

    /// One pre-tokenized piece to ids: tiktoken's `byte_pair_encode`, step for step, so ties
    /// break the same way (the leftmost of equal ranks merges first).
    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) -> Result<(), String> {
        if let Some(&id) = self.ranks.get(piece) {
            out.push(id);
            return Ok(());
        }
        // (start offset, rank of the pair starting here)
        let mut parts: Vec<(usize, u32)> = Vec::with_capacity(piece.len() + 1);
        let mut min = (u32::MAX, usize::MAX);
        for i in 0..piece.len() - 1 {
            let r = self.rank(&piece[i..i + 2]);
            if r < min.0 {
                min = (r, i);
            }
            parts.push((i, r));
        }
        parts.push((piece.len() - 1, u32::MAX));
        parts.push((piece.len(), u32::MAX));

        let pair_rank = |parts: &[(usize, u32)], i: usize| {
            if i + 3 < parts.len() {
                self.rank(&piece[parts[i].0..parts[i + 3].0])
            } else {
                u32::MAX
            }
        };
        while min.0 != u32::MAX {
            let i = min.1;
            if i > 0 {
                parts[i - 1].1 = pair_rank(&parts, i - 1);
            }
            parts[i].1 = pair_rank(&parts, i);
            parts.remove(i + 1);
            min = (u32::MAX, usize::MAX);
            for (j, &(_, r)) in parts[..parts.len() - 1].iter().enumerate() {
                if r < min.0 {
                    min = (r, j);
                }
            }
        }
        for w in parts.windows(2) {
            let bytes = &piece[w[0].0..w[1].0];
            match self.ranks.get(bytes) {
                Some(&id) => out.push(id),
                None => return Err(format!("no token for bytes {bytes:?}")),
            }
        }
        Ok(())
    }
}

impl Tokenizer for TiktokenBpe {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut out = Vec::with_capacity(text.len() / 4 + 1);
        for m in self.pattern.find_iter(text) {
            let m = m.map_err(|e| format!("pre-tokenizer failed: {e}"))?;
            self.encode_piece(m.as_str().as_bytes(), &mut out)?;
        }
        Ok(out)
    }
}

/// The pre-tokenizer of tiktoken's `o200k_base`, verbatim (`tiktoken_ext/openai_public.py`).
pub const O200K_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

#[cfg(test)]
mod unit_tests {
    use crate::preset::O200K;

    fn o200k(text: &str) -> Vec<u32> {
        O200K.tokenizer().unwrap().tokenize(text).unwrap()
    }

    #[test]
    fn matches_tiktoken_on_known_strings() {
        // Expected ids from tiktoken 0.13 `o200k_base`, `encode_ordinary`.
        assert_eq!(o200k("Hello, world!"), [13225, 11, 2375, 0]);
        assert_eq!(o200k("🦜"), [4103, 99, 250]);
        assert_eq!(o200k(""), Vec::<u32>::new());
    }

    #[test]
    fn special_tokens_are_plain_text() {
        let ids = o200k("<|endoftext|>");
        assert!(!ids.contains(&199_999), "{ids:?}");
        assert!(ids.len() > 1);
    }
}
