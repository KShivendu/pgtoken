//! Turning stored token IDs back into text.
//!
//! A lookup and a concatenation — which is why this can live here while the crate still has no
//! tokenizer. Tokenizing is an algorithm; detokenizing is a table.
//!
//! Bytes are concatenated first and interpreted **once**, at the end. Per-token interpretation
//! would be wrong: a single character routinely spans two tokens, so neither token is valid
//! UTF-8 alone.

use core::fmt;

use crate::tables::ByteMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetokError {
    /// A stored id has no entry in the mapping. The mapping and the ids disagree about which
    /// tokenizer produced them, which is a fault rather than an empty string.
    UnmappedId(u32),
}

impl fmt::Display for DetokError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetokError::UnmappedId(id) => {
                write!(f, "token id {id} has no entry in the mapping")
            }
        }
    }
}

impl std::error::Error for DetokError {}

/// Concatenate the mapped bytes for `ids` and interpret them as UTF-8, lossily.
///
/// Any byte sequence that is not valid UTF-8 becomes U+FFFD, the replacement character, the same
/// as `bytes.decode("utf-8", errors="replace")` in Python and tiktoken's own `decode`. The usual
/// cause is a chunk cut by token count in the middle of a character: 🦜 is three o200k tokens,
/// and the first one alone is two bytes of a four-byte character. Dropping those bytes would turn
/// that one-token chunk into an empty string, which reads like "nothing here"; a replacement
/// character says "something here that is not a whole character".
pub fn to_text(ids: &[u32], map: &ByteMap) -> Result<String, DetokError> {
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        match map.get(id) {
            Some(b) => bytes.extend_from_slice(b),
            None => return Err(DetokError::UnmappedId(id)),
        }
    }
    Ok(match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::ByteMap;

    /// A tiny byte-level mapping: ids 0-3 are ASCII, 4 and 5 are the two halves of the
    /// two-byte UTF-8 sequence for 'é' (0xC3 0xA9), so a chunk can be cut mid-character.
    fn map() -> ByteMap {
        ByteMap::build(
            &[
                (0, b"Hello".to_vec()),
                (1, b", ".to_vec()),
                (2, b"world".to_vec()),
                (3, b"!".to_vec()),
                (4, vec![0xC3]),
                (5, vec![0xA9]),
            ],
            8,
        )
        .unwrap()
    }

    #[test]
    fn concatenates_mapped_bytes() {
        assert_eq!(to_text(&[0, 1, 2, 3], &map()).unwrap(), "Hello, world!");
    }

    #[test]
    fn joins_a_character_split_across_two_tokens() {
        // Neither 0xC3 nor 0xA9 is valid UTF-8 alone; together they are 'é'. This is why the
        // mapping is id -> bytes and not id -> text.
        assert_eq!(to_text(&[4, 5], &map()).unwrap(), "é");
    }

    #[test]
    fn replaces_an_incomplete_trailing_character() {
        // The chunk ended mid-character, an artefact of chunking by token count.
        assert_eq!(to_text(&[0, 4], &map()).unwrap(), "Hello\u{FFFD}");
    }

    #[test]
    fn replaces_invalid_bytes_in_the_middle() {
        assert_eq!(to_text(&[5, 0], &map()).unwrap(), "\u{FFFD}Hello");
    }

    #[test]
    fn errors_on_an_unmapped_id() {
        let err = to_text(&[0, 6], &map()).expect_err("id 6 has no entry");
        assert_eq!(err, DetokError::UnmappedId(6));
    }

    #[test]
    fn empty_input_is_empty_text() {
        assert_eq!(to_text(&[], &map()).unwrap(), "");
    }

    #[test]
    fn a_wholly_unreadable_chunk_is_a_replacement_character_not_empty() {
        // One token that is only the first half of a character must not read as "".
        assert_eq!(to_text(&[4], &map()).unwrap(), "\u{FFFD}");
    }
}
