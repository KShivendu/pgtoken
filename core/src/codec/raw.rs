//! Raw token-ID packing: no coding table, no entropy model.
//!
//! Ports `tnbench.pack3` / `tnbench.unpack3` (tnbench.py:128-140) plus the `uint16` path
//! the Python harness gets implicitly from `np.array(ids, dtype=np.uint16).tobytes()`.
//!
//! Byte order is deliberately asymmetric so both match the harness exactly:
//! `raw24` is **big-endian** (that is what `pack3` writes), `raw16` is **little-endian**
//! (numpy's native order on x86). Getting either backwards still roundtrips within this
//! module, so only the cross-language test catches it.

use crate::header::{Codec, HeaderError, Tokenizer};

/// Anything wrong with the token IDs themselves, as opposed to the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawError {
    /// A token ID is outside the tokenizer's vocabulary.
    IdOutOfRange { id: u32, vocab: u32 },
    /// A token ID does not fit the packing width.
    IdTooWide { id: u32, codec: Codec },
}

impl std::fmt::Display for RawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RawError::IdOutOfRange { id, vocab } => {
                write!(f, "token id {id} is outside the vocabulary (size {vocab})")
            }
            RawError::IdTooWide { id, codec } => {
                write!(f, "token id {id} does not fit codec {}", codec.as_str())
            }
        }
    }
}

impl std::error::Error for RawError {}

/// Reject IDs the tokenizer could never have produced.
///
/// Without this an out-of-range ID would be silently truncated by the packing and decode
/// to a different, valid-looking token.
pub fn validate_ids(ids: &[u32], tokenizer: Tokenizer) -> Result<(), RawError> {
    let vocab = tokenizer.vocab();
    for &id in ids {
        if id >= vocab {
            return Err(RawError::IdOutOfRange { id, vocab });
        }
    }
    Ok(())
}

/// The narrowest raw codec that can represent this tokenizer's IDs.
pub fn preferred_raw(tokenizer: Tokenizer) -> Codec {
    if tokenizer.fits_u16() {
        Codec::Raw16
    } else {
        Codec::Raw24
    }
}

/// 2 bytes/id, little-endian.
pub fn encode16(ids: &[u32], out: &mut Vec<u8>) -> Result<(), RawError> {
    out.reserve(ids.len() * 2);
    for &id in ids {
        let narrow = u16::try_from(id).map_err(|_| RawError::IdTooWide { id, codec: Codec::Raw16 })?;
        out.extend_from_slice(&narrow.to_le_bytes());
    }
    Ok(())
}

/// 3 bytes/id, big-endian. Mirrors `tnbench.pack3`.
pub fn encode24(ids: &[u32], out: &mut Vec<u8>) -> Result<(), RawError> {
    out.reserve(ids.len() * 3);
    for &id in ids {
        if id > 0x00FF_FFFF {
            return Err(RawError::IdTooWide { id, codec: Codec::Raw24 });
        }
        out.push((id >> 16) as u8);
        out.push((id >> 8) as u8);
        out.push(id as u8);
    }
    Ok(())
}

/// Decode a `raw16` payload. `Header::parse` has already checked the length.
pub fn decode16(payload: &[u8], n: usize) -> Result<Vec<u32>, HeaderError> {
    if payload.len() != n * 2 {
        return Err(HeaderError::PayloadLen { want: n * 2, got: payload.len() });
    }
    Ok(payload.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect())
}

/// Decode a `raw24` payload. Mirrors `tnbench.unpack3`.
pub fn decode24(payload: &[u8], n: usize) -> Result<Vec<u32>, HeaderError> {
    if payload.len() != n * 3 {
        return Err(HeaderError::PayloadLen { want: n * 3, got: payload.len() });
    }
    Ok(payload
        .chunks_exact(3)
        .map(|c| ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw24_matches_pack3_byte_order() {
        // pack3 writes (id >> 16, id >> 8, id) — big-endian. Pin the exact bytes so a
        // future "optimisation" to LE cannot pass silently.
        let mut out = Vec::new();
        encode24(&[0x01_02_03], &mut out).unwrap();
        assert_eq!(out, vec![0x01, 0x02, 0x03]);
    }

    #[test]
    fn raw16_matches_numpy_uint16_byte_order() {
        // np.array([0x0102], dtype=np.uint16).tobytes() == b'\x02\x01' on x86.
        let mut out = Vec::new();
        encode16(&[0x0102], &mut out).unwrap();
        assert_eq!(out, vec![0x02, 0x01]);
    }

    #[test]
    fn raw16_roundtrips_across_r50k_range() {
        let ids: Vec<u32> = (0..50_257).collect();
        let mut out = Vec::new();
        encode16(&ids, &mut out).unwrap();
        assert_eq!(out.len(), ids.len() * 2);
        assert_eq!(decode16(&out, ids.len()).unwrap(), ids);
    }

    #[test]
    fn raw24_roundtrips_across_o200k_range() {
        let ids: Vec<u32> = (0..200_019).collect();
        let mut out = Vec::new();
        encode24(&ids, &mut out).unwrap();
        assert_eq!(out.len(), ids.len() * 3);
        assert_eq!(decode24(&out, ids.len()).unwrap(), ids);
    }

    #[test]
    fn empty_roundtrips() {
        let mut out = Vec::new();
        encode24(&[], &mut out).unwrap();
        assert!(out.is_empty());
        assert_eq!(decode24(&out, 0).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn rejects_out_of_vocab_id() {
        assert_eq!(
            validate_ids(&[50_257], Tokenizer::R50k),
            Err(RawError::IdOutOfRange { id: 50_257, vocab: 50_257 })
        );
        assert!(validate_ids(&[50_256], Tokenizer::R50k).is_ok());
    }

    #[test]
    fn rejects_id_too_wide_for_codec() {
        let mut out = Vec::new();
        assert_eq!(
            encode16(&[65_536], &mut out),
            Err(RawError::IdTooWide { id: 65_536, codec: Codec::Raw16 })
        );
        let mut out = Vec::new();
        assert_eq!(
            encode24(&[0x0100_0000], &mut out),
            Err(RawError::IdTooWide { id: 0x0100_0000, codec: Codec::Raw24 })
        );
    }

    #[test]
    fn decode_rejects_wrong_length() {
        assert!(decode16(&[0, 0, 0], 1).is_err());
        assert!(decode24(&[0, 0], 1).is_err());
    }

    #[test]
    fn preferred_raw_narrows_only_for_r50k() {
        assert_eq!(preferred_raw(Tokenizer::R50k), Codec::Raw16);
        assert_eq!(preferred_raw(Tokenizer::Cl100k), Codec::Raw24);
        assert_eq!(preferred_raw(Tokenizer::O200k), Codec::Raw24);
    }
}
