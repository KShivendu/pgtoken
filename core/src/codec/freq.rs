//! `+freq`: frequency-rank remap, then streamvbyte.
//!
//! Ports `tnbench.svb_encode_arr` / `svb_decode_arr` (tnbench.py:195-208) combined with the
//! rank remap from [`crate::tables::RankTable`].
//!
//! This is the recommended default codec. It gives most of ANS's ratio (2.73x vs 3.40x on
//! English/o200k) at the fastest decode of any of them, because there is no general
//! compressor in the path at all — just a varint container over small integers.
//!
//! Byte compatibility with the Python harness is **not** expected here: the harness uses
//! `pyfastpfor`'s streamvbyte, whose container layout differs from the `stream-vbyte`
//! crate's. The cross-language contract is equal token IDs after a roundtrip and
//! compression ratio within 1%, not identical bytes.

use stream_vbyte::{decode::decode, encode::encode, scalar::Scalar};

use crate::tables::RankTable;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreqError {
    /// A token ID was outside the table's vocabulary.
    IdOutOfRange { id: u32, vocab: u32 },
    /// A decoded rank was outside the table's vocabulary, meaning the payload does not
    /// match the table it claims.
    RankOutOfRange { rank: u32, vocab: u32 },
    /// The varint stream did not consume the whole payload, so the payload and the declared
    /// token count disagree.
    TrailingBytes { consumed: usize, len: usize },
}

impl std::fmt::Display for FreqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FreqError::IdOutOfRange { id, vocab } => {
                write!(f, "token id {id} is outside the table vocabulary ({vocab})")
            }
            FreqError::RankOutOfRange { rank, vocab } => {
                write!(f, "decoded rank {rank} is outside the table vocabulary ({vocab})")
            }
            FreqError::TrailingBytes { consumed, len } => write!(
                f,
                "streamvbyte consumed {consumed} of {len} payload bytes; the token count and \
                 payload disagree"
            ),
        }
    }
}

impl std::error::Error for FreqError {}

/// Worst-case streamvbyte output: 4 data bytes per value plus one control byte per quad.
fn encode_capacity(n: usize) -> usize {
    // The crate's own guidance is 5x the input length, which covers both terms.
    n * 5
}

/// Encode token IDs as frequency ranks packed with streamvbyte.
pub fn encode_freq(ids: &[u32], table: &RankTable, out: &mut Vec<u8>) -> Result<(), FreqError> {
    if ids.is_empty() {
        return Ok(());
    }
    let vocab = table.vocab();
    let mut ranks = Vec::with_capacity(ids.len());
    for &id in ids {
        match table.rank_of(id) {
            Some(r) => ranks.push(r),
            None => return Err(FreqError::IdOutOfRange { id, vocab }),
        }
    }

    let mut buf = vec![0u8; encode_capacity(ranks.len())];
    let written = encode::<Scalar>(&ranks, &mut buf);
    out.extend_from_slice(&buf[..written]);
    Ok(())
}

/// Decode a `+freq` payload back to token IDs.
pub fn decode_freq(payload: &[u8], n: usize, table: &RankTable) -> Result<Vec<u32>, FreqError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    // `decode` documents that the output slice must hold at least 4 values regardless of
    // `count`, and it panics rather than erroring if the buffer is short. Oversize it.
    let mut ranks = vec![0u32; n.max(4)];
    let consumed = decode::<Scalar>(payload, n, &mut ranks);
    if consumed != payload.len() {
        return Err(FreqError::TrailingBytes { consumed, len: payload.len() });
    }
    ranks.truncate(n);

    let vocab = table.vocab();
    let mut ids = Vec::with_capacity(n);
    for &r in &ranks {
        match table.token_of_rank(r) {
            Some(t) => ids.push(t),
            None => return Err(FreqError::RankOutOfRange { rank: r, vocab }),
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Tokenizer;

    fn table() -> RankTable {
        // Token 3 most common, then 1, then 2.
        let mut corpus = vec![3u32; 10];
        corpus.extend(vec![1u32; 5]);
        corpus.extend(vec![2u32; 2]);
        RankTable::train(&corpus, Tokenizer::O200k)
    }

    fn roundtrip(ids: &[u32], t: &RankTable) -> Vec<u32> {
        let mut out = Vec::new();
        encode_freq(ids, t, &mut out).expect("encode");
        decode_freq(&out, ids.len(), t).expect("decode")
    }

    #[test]
    fn roundtrips_common_tokens() {
        let t = table();
        let ids = vec![3, 1, 2, 3, 3, 1];
        assert_eq!(roundtrip(&ids, &t), ids);
    }

    #[test]
    fn roundtrips_every_quad_remainder() {
        // streamvbyte groups values into quads, so lengths 1..=9 exercise both the
        // complete-quad path and every partial-quad remainder.
        let t = table();
        for n in 1..=9usize {
            let ids: Vec<u32> = (0..n as u32).map(|i| (i * 7) % 100).collect();
            assert_eq!(roundtrip(&ids, &t), ids, "failed at n={n}");
        }
    }

    #[test]
    fn roundtrips_wide_ids() {
        let t = table();
        let ids = vec![0, 1, 200_018, 65_535, 65_536, 100_000];
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
    fn rejects_id_outside_vocabulary() {
        let t = table();
        let mut out = Vec::new();
        assert_eq!(
            encode_freq(&[200_019], &t, &mut out),
            Err(FreqError::IdOutOfRange { id: 200_019, vocab: 200_019 })
        );
    }

    #[test]
    fn remap_shrinks_frequent_tokens() {
        // The point of the rank remap: a high-ID but frequent token encodes in one byte
        // once remapped, where its raw ID would have taken three.
        let corpus = vec![199_999u32; 100];
        let t = RankTable::train(&corpus, Tokenizer::O200k);
        assert_eq!(t.rank_of(199_999), Some(0));

        let ids = vec![199_999u32; 4];
        let mut remapped = Vec::new();
        encode_freq(&ids, &t, &mut remapped).expect("encode");

        // 4 values in one quad: 1 control byte + 1 data byte each.
        assert_eq!(remapped.len(), 5);
        // Raw 3-byte packing of the same IDs would be 12 bytes.
        assert!(remapped.len() < ids.len() * 3);
    }
}
