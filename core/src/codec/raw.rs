//! Raw token-ID packing: no table, no model.
//!
//! `raw24` is **big-endian**, `raw16` **little-endian**. Both orders are fixed by the format
//! and pinned by tests, since either would roundtrip within this module if reversed.

use crate::header::{Codec, HeaderError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawError {
    /// A token ID does not fit the packing width.
    IdTooWide { id: u32, codec: Codec },
}

impl std::fmt::Display for RawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RawError::IdTooWide { id, codec } => write!(
                f,
                "token id {id} does not fit codec {} (max {})",
                codec.as_str(),
                codec.max_id().unwrap_or(u32::MAX)
            ),
        }
    }
}

impl std::error::Error for RawError {}

/// The narrowest raw codec that can hold every ID in `ids`.
///
/// Chosen from the data rather than from a declared vocabulary, so nothing needs to know how
/// large the tokenizer is.
pub fn preferred_raw(ids: &[u32]) -> Codec {
    match ids.iter().copied().max() {
        Some(m) if m > u16::MAX as u32 => Codec::Raw24,
        _ => Codec::Raw16,
    }
}

/// 2 bytes/id, little-endian.
pub fn encode16(ids: &[u32], out: &mut Vec<u8>) -> Result<(), RawError> {
    out.reserve(ids.len() * 2);
    for &id in ids {
        let narrow = u16::try_from(id).map_err(|_| RawError::IdTooWide {
            id,
            codec: Codec::Raw16,
        })?;
        out.extend_from_slice(&narrow.to_le_bytes());
    }
    Ok(())
}

/// 3 bytes/id, big-endian.
pub fn encode24(ids: &[u32], out: &mut Vec<u8>) -> Result<(), RawError> {
    out.reserve(ids.len() * 3);
    for &id in ids {
        if id > 0x00FF_FFFF {
            return Err(RawError::IdTooWide {
                id,
                codec: Codec::Raw24,
            });
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
        return Err(HeaderError::PayloadLen {
            want: n * 2,
            got: payload.len(),
        });
    }
    Ok(payload
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
        .collect())
}

/// Decode a `raw24` payload.
pub fn decode24(payload: &[u8], n: usize) -> Result<Vec<u32>, HeaderError> {
    if payload.len() != n * 3 {
        return Err(HeaderError::PayloadLen {
            want: n * 3,
            got: payload.len(),
        });
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
    fn raw24_is_big_endian() {
        let mut out = Vec::new();
        encode24(&[0x01_02_03], &mut out).unwrap();
        assert_eq!(out, vec![0x01, 0x02, 0x03]);
    }

    #[test]
    fn raw16_is_little_endian() {
        let mut out = Vec::new();
        encode16(&[0x0102], &mut out).unwrap();
        assert_eq!(out, vec![0x02, 0x01]);
    }

    #[test]
    fn raw16_roundtrips_its_whole_range() {
        let ids: Vec<u32> = (0..=u16::MAX as u32).collect();
        let mut out = Vec::new();
        encode16(&ids, &mut out).unwrap();
        assert_eq!(out.len(), ids.len() * 2);
        assert_eq!(decode16(&out, ids.len()).unwrap(), ids);
    }

    #[test]
    fn raw24_roundtrips_across_a_large_range() {
        let ids: Vec<u32> = (0..300_000).collect();
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
    fn rejects_id_too_wide_for_codec() {
        let mut out = Vec::new();
        assert_eq!(
            encode16(&[65_536], &mut out),
            Err(RawError::IdTooWide {
                id: 65_536,
                codec: Codec::Raw16
            })
        );
        let mut out = Vec::new();
        assert_eq!(
            encode24(&[0x0100_0000], &mut out),
            Err(RawError::IdTooWide {
                id: 0x0100_0000,
                codec: Codec::Raw24
            })
        );
    }

    #[test]
    fn decode_rejects_wrong_length() {
        assert!(decode16(&[0, 0, 0], 1).is_err());
        assert!(decode24(&[0, 0], 1).is_err());
    }

    #[test]
    fn preferred_raw_widens_only_when_the_data_needs_it() {
        assert_eq!(preferred_raw(&[]), Codec::Raw16);
        assert_eq!(preferred_raw(&[0, 1, 65_535]), Codec::Raw16);
        assert_eq!(preferred_raw(&[0, 65_536]), Codec::Raw24);
        assert_eq!(preferred_raw(&[200_018]), Codec::Raw24);
    }
}
