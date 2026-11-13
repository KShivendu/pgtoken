//! `freq`: frequency-rank remap, then one LEB128 varint per rank.
//!
//! The recommended codec. Common tokens become small integers that a varint packs in one
//! byte, and there is no general compressor in the path, so decoding is close to the cost of
//! raw packing. See [`crate::tables::RankTable`] for how unseen tokens stay lossless without
//! anyone declaring a vocabulary size.
//!
//! Each rank is unsigned LEB128: 7 bits per byte, low bits first, high bit set on every byte
//! but the last, so the number of ranks is the number of bytes with the high bit clear; the
//! decoder checks it equals the header's token count. LEB128 replaced streamvbyte here because it
//! measured both smaller (5-7%) and faster to decode on real o200k text
//! (`core/examples/leb_vs_svb.rs`).
//!
//! The encoding is canonical, which `=` and hashing rely on, so a decoder refuses anything an
//! encoder would not write: a payload ending mid-varint, a varint with a redundant trailing
//! zero byte, and a rank that does not fit `u32`.

use crate::header::write_varint;
use crate::tables::{RankTable, TableError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreqError {
    /// The ID could not be remapped; see [`TableError::IdTooLarge`].
    Table(TableError),
    /// The payload ends with a continuation byte.
    Truncated,
    /// A multi-byte varint ends in `0x00`, so a shorter encoding of the same rank exists.
    NonMinimal { offset: usize },
    /// A varint encodes a value above `u32::MAX`.
    Overflow { offset: usize },
    /// The payload holds a different number of ranks than the header's count.
    CountMismatch { want: usize, got: usize },
    /// A `freq_lz4` payload is not a valid lz4 block, or decompresses past its bound.
    Lz4(String),
}

impl std::fmt::Display for FreqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FreqError::Table(e) => write!(f, "{e}"),
            FreqError::Truncated => write!(f, "freq payload ends inside a varint"),
            FreqError::NonMinimal { offset } => write!(
                f,
                "freq payload has a non-minimal varint ending at byte {offset}"
            ),
            FreqError::Overflow { offset } => write!(
                f,
                "freq payload has a varint at byte {offset} larger than 32 bits"
            ),
            FreqError::CountMismatch { want, got } => write!(
                f,
                "freq payload holds {got} tokens but the header says {want}"
            ),
            FreqError::Lz4(e) => write!(f, "freq_lz4 payload does not decompress: {e}"),
        }
    }
}

impl std::error::Error for FreqError {}

impl From<TableError> for FreqError {
    fn from(e: TableError) -> Self {
        FreqError::Table(e)
    }
}

/// Encode token IDs as frequency ranks, one LEB128 varint each.
pub fn encode_freq(ids: &[u32], table: &RankTable, out: &mut Vec<u8>) -> Result<(), FreqError> {
    out.reserve(ids.len() * 2);
    for &id in ids {
        write_varint(table.rank(id)?, out);
    }
    Ok(())
}

/// Walk a payload's varints, calling `f` with each rank, and reject a non-canonical stream.
#[inline]
fn for_each_rank(payload: &[u8], mut f: impl FnMut(u32)) -> Result<(), FreqError> {
    let mut v: u32 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in payload.iter().enumerate() {
        let bits = (b & 0x7F) as u32;
        if shift == 28 && b > 0x0F {
            // The fifth byte holds only the top 4 bits of a u32; anything more, including a
            // continuation bit, overflows.
            return Err(FreqError::Overflow { offset: i });
        }
        v |= bits << shift;
        if b & 0x80 == 0 {
            if b == 0 && shift > 0 {
                return Err(FreqError::NonMinimal { offset: i });
            }
            f(v);
            v = 0;
            shift = 0;
        } else {
            shift += 7;
        }
    }
    if shift != 0 {
        return Err(FreqError::Truncated);
    }
    Ok(())
}

/// Decode a `freq` payload of `n` tokens back to token IDs.
pub fn decode_freq(payload: &[u8], n: usize, table: &RankTable) -> Result<Vec<u32>, FreqError> {
    let mut out = Vec::with_capacity(n);
    for_each_rank(payload, |r| out.push(table.token(r)))?;
    if out.len() != n {
        return Err(FreqError::CountMismatch {
            want: n,
            got: out.len(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> RankTable {
        let mut corpus = vec![3u32; 10];
        corpus.extend(vec![1u32; 5]);
        corpus.extend(vec![2u32; 2]);
        RankTable::train(&corpus, None).unwrap()
    }

    fn roundtrip(ids: &[u32], t: &RankTable) -> Vec<u32> {
        let mut out = Vec::new();
        encode_freq(ids, t, &mut out).expect("encode");
        decode_freq(&out, ids.len(), t).expect("decode")
    }

    #[test]
    fn roundtrips_ranked_tokens() {
        let t = table();
        let ids = vec![3, 1, 2, 3, 3, 1];
        assert_eq!(roundtrip(&ids, &t), ids);
    }

    #[test]
    fn roundtrips_ids_the_table_never_saw() {
        // No vocabulary is declared anywhere, so arbitrarily large IDs must still work.
        let t = table();
        let ids = vec![0, 4, 65_535, 65_536, 200_018, 1_000_000, u32::MAX - 3];
        assert_eq!(roundtrip(&ids, &t), ids);
    }

    #[test]
    fn roundtrips_a_mix_of_ranked_and_unranked() {
        let t = table();
        let ids = vec![3, 999_999, 1, 0, 2, 123_456];
        assert_eq!(roundtrip(&ids, &t), ids);
    }

    #[test]
    fn empty_roundtrips() {
        let t = table();
        let mut out = Vec::new();
        encode_freq(&[], &t, &mut out).expect("encode");
        assert!(out.is_empty());
        assert_eq!(decode_freq(&out, 0, &t).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn golden_varints() {
        // Ranks 0, 127, 128, 16383, 16384 and u32::MAX at every varint width boundary.
        let t = RankTable::train(&[0], None).unwrap(); // k = 1: rank r is id r - 1 above 0
        let ids = [0u32, 126, 127, 16_382, 16_383, u32::MAX - 1];
        let mut out = Vec::new();
        encode_freq(&ids, &t, &mut out).unwrap();
        assert_eq!(
            out,
            [
                0x00, // 0
                0x7F, // 127
                0x80, 0x01, // 128
                0xFF, 0x7F, // 16383
                0x80, 0x80, 0x01, // 16384
                0xFF, 0xFF, 0xFF, 0xFF, 0x0F, // u32::MAX
            ]
        );
        assert_eq!(decode_freq(&out, ids.len(), &t).unwrap(), ids);
    }

    #[test]
    fn remap_shrinks_frequent_high_ids() {
        // The point of the remap: a frequent but high-numbered token packs in one byte, where
        // its bare ID would have taken three.
        let t = RankTable::train(&vec![199_999u32; 100], None).unwrap();
        assert_eq!(t.rank(199_999).unwrap(), 0);
        let mut out = Vec::new();
        encode_freq(&[199_999u32; 4], &t, &mut out).expect("encode");
        assert_eq!(out, [0, 0, 0, 0], "one byte per token, no control bytes");
    }

    #[test]
    fn rejects_ids_that_cannot_be_remapped() {
        let t = table();
        let mut out = Vec::new();
        assert!(matches!(
            encode_freq(&[u32::MAX], &t, &mut out),
            Err(FreqError::Table(TableError::IdTooLarge { .. }))
        ));
    }

    #[test]
    fn rejects_a_truncated_varint() {
        let t = table();
        assert_eq!(decode_freq(&[0x00, 0x80], 2, &t), Err(FreqError::Truncated));
        assert_eq!(decode_freq(&[0xFF], 1, &t), Err(FreqError::Truncated));
    }

    #[test]
    fn rejects_a_non_minimal_varint() {
        let t = table();
        // 1 as two bytes, and 0 as three.
        assert_eq!(
            decode_freq(&[0x81, 0x00], 1, &t),
            Err(FreqError::NonMinimal { offset: 1 })
        );
        assert_eq!(
            decode_freq(&[0x05, 0x80, 0x80, 0x00], 2, &t),
            Err(FreqError::NonMinimal { offset: 3 })
        );
    }

    #[test]
    fn rejects_a_rank_that_overflows_u32() {
        let t = table();
        // 2^32 needs 0x10 in the fifth byte; a sixth byte is over regardless.
        assert_eq!(
            decode_freq(&[0x80, 0x80, 0x80, 0x80, 0x10], 1, &t),
            Err(FreqError::Overflow { offset: 4 })
        );
        assert_eq!(
            decode_freq(&[0xFF, 0xFF, 0xFF, 0xFF, 0x8F, 0x00], 1, &t),
            Err(FreqError::Overflow { offset: 4 })
        );
        assert!(decode_freq(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F], 1, &t).is_ok());
    }

    #[test]
    fn rejects_a_payload_that_disagrees_with_the_count() {
        let t = table();
        assert_eq!(
            decode_freq(&[0, 1, 2], 2, &t),
            Err(FreqError::CountMismatch { want: 2, got: 3 })
        );
        assert_eq!(
            decode_freq(&[0x80, 0x01], 2, &t),
            Err(FreqError::CountMismatch { want: 2, got: 1 })
        );
    }
}
