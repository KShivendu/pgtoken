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
    /// A byte was invalid where it stood, rather than the input merely ending mid-character.
    InvalidUtf8 { at: usize },
}

impl fmt::Display for DetokError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetokError::UnmappedId(id) => {
                write!(f, "token id {id} has no entry in the mapping")
            }
            DetokError::InvalidUtf8 { at } => write!(
                f,
                "detokenized bytes are not valid UTF-8 at byte {at}, and not merely cut short"
            ),
        }
    }
}

impl std::error::Error for DetokError {}

/// Concatenate the mapped bytes for `ids` and interpret them as UTF-8.
///
/// An incomplete sequence at the **end** is dropped: the chunk was cut by token count and a
/// multi-byte character straddled the boundary, which is an artefact of chunking rather than a
/// fault. An invalid byte **anywhere else** is an error — it means the mapping is wrong, the ids
/// came from another tokenizer, or the data is corrupt, and rendering it would hand back
/// plausible-looking garbage.
pub fn to_text(ids: &[u32], map: &ByteMap) -> Result<String, DetokError> {
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        match map.get(id) {
            Some(b) => bytes.extend_from_slice(b),
            None => return Err(DetokError::UnmappedId(id)),
        }
    }

    match core::str::from_utf8(&bytes) {
        Ok(s) => Ok(s.to_owned()),
        Err(e) => match e.error_len() {
            // `None` means the input ended mid-character: the only failure a correctly-mapped
            // chunk boundary can produce. Keep what is readable. The prefix up to
            // `valid_up_to()` is valid UTF-8 by construction (that is what the field means), so
            // this can only be a no-op conversion, never a substitution.
            None => Ok(core::str::from_utf8(&bytes[..e.valid_up_to()])
                .expect("prefix up to valid_up_to() is valid UTF-8 by construction")
                .to_owned()),
            Some(_) => Err(DetokError::InvalidUtf8 {
                at: e.valid_up_to(),
            }),
        },
    }
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
    fn drops_an_incomplete_trailing_character() {
        // The chunk ended mid-character, which is an artefact of chunking by token count, not a
        // fault. Return what is readable.
        assert_eq!(to_text(&[0, 4], &map()).unwrap(), "Hello");
    }

    #[test]
    fn errors_on_invalid_bytes_in_the_middle() {
        // A stray continuation byte followed by more text cannot be a chunk boundary artefact —
        // it means the mapping or the ids are wrong.
        let err = to_text(&[5, 0], &map()).expect_err("must not be papered over");
        assert!(matches!(err, DetokError::InvalidUtf8 { .. }), "got {err:?}");
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
    fn a_wholly_unreadable_chunk_is_empty_not_an_error() {
        // One token that is only the first half of a character: everything is an incomplete
        // trailing sequence, so there is nothing readable and nothing wrong.
        assert_eq!(to_text(&[4], &map()).unwrap(), "");
    }
}
