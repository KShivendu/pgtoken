//! `freq`: frequency-rank remap, then streamvbyte.
//!
//! The recommended codec. Common tokens become small integers that a varint packs in one
//! byte, and there is no general compressor in the path, so decoding is close to the cost of
//! raw packing. See [`crate::tables::RankTable`] for how unseen tokens stay lossless without
//! anyone declaring a vocabulary size.

use stream_vbyte::{decode::decode, encode::encode, scalar::Scalar};

use crate::tables::{RankTable, TableError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreqError {
    /// The ID could not be remapped; see [`TableError::IdTooLarge`].
    Table(TableError),
    /// The varint stream did not consume the whole payload, so the payload and the declared
    /// token count disagree.
    TrailingBytes { consumed: usize, len: usize },
}

impl std::fmt::Display for FreqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FreqError::Table(e) => write!(f, "{e}"),
            FreqError::TrailingBytes { consumed, len } => write!(
                f,
                "streamvbyte consumed {consumed} of {len} payload bytes; the token count and \
                 payload disagree"
            ),
        }
    }
}

impl std::error::Error for FreqError {}

impl From<TableError> for FreqError {
    fn from(e: TableError) -> Self {
        FreqError::Table(e)
    }
}

/// Encode token IDs as frequency ranks packed with streamvbyte.
pub fn encode_freq(ids: &[u32], table: &RankTable, out: &mut Vec<u8>) -> Result<(), FreqError> {
    if ids.is_empty() {
        return Ok(());
    }
    let mut ranks = Vec::with_capacity(ids.len());
    for &id in ids {
        ranks.push(table.rank(id)?);
    }
    // The crate's guidance for a worst-case buffer is 5x the input length: up to 4 data bytes
    // per value plus one control byte per quad.
    let mut buf = vec![0u8; ranks.len() * 5];
    let written = encode::<Scalar>(&ranks, &mut buf);
    out.extend_from_slice(&buf[..written]);
    Ok(())
}

/// Decode a `freq` payload back to token IDs.
pub fn decode_freq(payload: &[u8], n: usize, table: &RankTable) -> Result<Vec<u32>, FreqError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    // `decode` documents that the output slice must hold at least 4 values regardless of
    // `count`, and it panics rather than erroring if the buffer is short. Oversize it.
    let mut ranks = vec![0u32; n.max(4)];
    let consumed = decode::<Scalar>(payload, n, &mut ranks);
    if consumed != payload.len() {
        return Err(FreqError::TrailingBytes {
            consumed,
            len: payload.len(),
        });
    }
    ranks.truncate(n);
    Ok(ranks.into_iter().map(|r| table.token(r)).collect())
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
    fn roundtrips_every_quad_remainder() {
        // streamvbyte groups values into quads, so lengths 1..=9 exercise the complete-quad
        // path and every partial remainder.
        let t = table();
        for n in 1..=9usize {
            let ids: Vec<u32> = (0..n as u32).map(|i| (i * 7) % 100).collect();
            assert_eq!(roundtrip(&ids, &t), ids, "failed at n={n}");
        }
    }

    #[test]
    fn roundtrips_ids_the_table_never_saw() {
        // No vocabulary is declared anywhere, so arbitrarily large IDs must still work.
        let t = table();
        let ids = vec![0, 4, 65_535, 65_536, 200_018, 1_000_000];
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
    fn remap_shrinks_frequent_high_ids() {
        // The point of the remap: a frequent but high-numbered token packs in one byte, where
        // its bare ID would have taken three.
        let t = RankTable::train(&vec![199_999u32; 100], None).unwrap();
        assert_eq!(t.rank(199_999).unwrap(), 0);

        let ids = vec![199_999u32; 4];
        let mut out = Vec::new();
        encode_freq(&ids, &t, &mut out).expect("encode");
        assert_eq!(
            out.len(),
            5,
            "4 values in one quad: 1 control byte + 1 data byte each"
        );
        assert!(out.len() < ids.len() * 3, "should beat 3-byte raw packing");
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
}
