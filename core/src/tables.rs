//! The frequency-rank table behind the `freq` codec.
//!
//! BPE numbers its vocabulary in merge-discovery order, so a common token can sit at ID
//! 40,000 while a rare one sits at 400. Remapping each ID to its descending-frequency rank
//! puts common tokens on small integers, which is what a varint codec rewards.
//!
//! The table is **sparse**: it holds only the tokens seen during training, in frequency
//! order. Anything else maps to `k + id`, and decodes back by subtracting `k`. That single
//! decision keeps this library free of any vocabulary size — nothing needs to know how large
//! the tokenizer's vocabulary is, or which tokenizer it was.
//!
//! ```text
//! encode:  r = rank_of(t)        if t was seen in training
//!          r = k + t             otherwise
//! decode:  t = token_of_rank[r]  if r < k
//!          t = r - k             otherwise
//! ```
//!
//! Lossless for every ID, because ranks below `k` are only ever produced for in-table tokens
//! and ranks at or above `k` only for out-of-table ones. The cost is that a token absent from
//! training encodes as a slightly wider varint than its bare ID would.
//!
//! # On-disk format
//!
//! ```text
//! off  size  field
//!   0     4  magic "TNTT"
//!   4     1  version (1)
//!   5     1  kind (1 = rank)
//!   6     2  reserved, must be zero
//!   8     4  k, the number of ranked tokens (u32 LE)
//!  12     -  token_of_rank: k x u32 LE
//! ```
//!
//! `rank_of` is the inverse and is rebuilt on load rather than stored, which halves the file.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

pub const TABLE_MAGIC: &[u8; 4] = b"TNTT";
pub const TABLE_VERSION: u8 = 1;
pub const TABLE_HEADER_LEN: usize = 12;

const KIND_RANK: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableError {
    TooShort(usize),
    BadMagic,
    UnsupportedVersion(u8),
    UnknownKind(u8),
    ReservedNotZero,
    PayloadLen { want: usize, got: usize },
    /// `token_of_rank` listed the same token twice, which would make decoding ambiguous.
    DuplicateToken(u32),
    /// Training saw no tokens at all.
    Empty,
    /// `k + id` would overflow, so this ID cannot be remapped against this table.
    IdTooLarge { id: u32, k: u32 },
}

impl std::fmt::Display for TableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableError::TooShort(n) => write!(f, "table file is only {n} bytes"),
            TableError::BadMagic => write!(f, "table file has bad magic, expected TNTT"),
            TableError::UnsupportedVersion(v) => write!(f, "unsupported table version {v}"),
            TableError::UnknownKind(k) => write!(f, "unknown table kind {k}"),
            TableError::ReservedNotZero => write!(f, "reserved table header bytes are not zero"),
            TableError::PayloadLen { want, got } => {
                write!(f, "table payload is {got} bytes, expected {want}")
            }
            TableError::DuplicateToken(t) => {
                write!(f, "token {t} appears twice in the table; ranks must be a bijection")
            }
            TableError::Empty => write!(f, "cannot train a table on an empty corpus"),
            TableError::IdTooLarge { id, k } => {
                write!(f, "token id {id} is too large to remap against a table of {k} ranks")
            }
        }
    }
}

impl std::error::Error for TableError {}

/// SHA-256 of a table file, used to content-address it.
pub fn table_digest(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

pub fn digest_hex(digest: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankTable {
    /// rank -> token id, most frequent first
    token_of_rank: Vec<u32>,
    /// token id -> rank, only for tokens seen in training
    rank_of: HashMap<u32, u32>,
}

impl RankTable {
    /// Train on a corpus of token IDs.
    ///
    /// Ranks every token that appears, descending by count, breaking ties by ascending token
    /// ID so the result is reproducible. `max_ranks` caps the table size; tokens beyond the
    /// cap take the `k + id` fallback.
    pub fn train(ids: &[u32], max_ranks: Option<usize>) -> Result<Self, TableError> {
        if ids.is_empty() {
            return Err(TableError::Empty);
        }
        let mut counts: HashMap<u32, u64> = HashMap::new();
        for &id in ids {
            *counts.entry(id).or_insert(0) += 1;
        }
        let mut ordered: Vec<(u32, u64)> = counts.into_iter().collect();
        ordered.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if let Some(cap) = max_ranks {
            ordered.truncate(cap);
        }
        Ok(Self::from_ranked(ordered.into_iter().map(|(t, _)| t).collect()))
    }

    fn from_ranked(token_of_rank: Vec<u32>) -> Self {
        let rank_of = token_of_rank.iter().enumerate().map(|(r, &t)| (t, r as u32)).collect();
        RankTable { token_of_rank, rank_of }
    }

    /// Number of ranked tokens, and the offset applied to unranked IDs.
    pub fn k(&self) -> u32 {
        self.token_of_rank.len() as u32
    }

    /// Map a token ID to the integer that actually gets packed.
    #[inline]
    pub fn rank(&self, token: u32) -> Result<u32, TableError> {
        if let Some(&r) = self.rank_of.get(&token) {
            return Ok(r);
        }
        let k = self.k();
        token.checked_add(k).ok_or(TableError::IdTooLarge { id: token, k })
    }

    /// Inverse of [`RankTable::rank`].
    #[inline]
    pub fn token(&self, rank: u32) -> u32 {
        match self.token_of_rank.get(rank as usize) {
            Some(&t) => t,
            None => rank - self.k(),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(TABLE_HEADER_LEN + self.token_of_rank.len() * 4);
        out.extend_from_slice(TABLE_MAGIC);
        out.push(TABLE_VERSION);
        out.push(KIND_RANK);
        out.extend_from_slice(&[0u8, 0u8]); // reserved
        out.extend_from_slice(&self.k().to_le_bytes());
        debug_assert_eq!(out.len(), TABLE_HEADER_LEN);
        for &t in &self.token_of_rank {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, TableError> {
        if buf.len() < TABLE_HEADER_LEN {
            return Err(TableError::TooShort(buf.len()));
        }
        if &buf[0..4] != TABLE_MAGIC {
            return Err(TableError::BadMagic);
        }
        if buf[4] != TABLE_VERSION {
            return Err(TableError::UnsupportedVersion(buf[4]));
        }
        if buf[5] != KIND_RANK {
            return Err(TableError::UnknownKind(buf[5]));
        }
        if buf[6] != 0 || buf[7] != 0 {
            return Err(TableError::ReservedNotZero);
        }
        let k = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        let payload = &buf[TABLE_HEADER_LEN..];
        if payload.len() != k * 4 {
            return Err(TableError::PayloadLen { want: k * 4, got: payload.len() });
        }
        let token_of_rank: Vec<u32> = payload
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // A duplicate would make two ranks decode to the same token and silently corrupt the
        // inverse mapping, so verify rather than trust the file.
        let mut rank_of = HashMap::with_capacity(token_of_rank.len());
        for (r, &t) in token_of_rank.iter().enumerate() {
            if rank_of.insert(t, r as u32).is_some() {
                return Err(TableError::DuplicateToken(t));
            }
        }
        Ok(RankTable { token_of_rank, rank_of })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Token 3 most common, then 1, then 2. Nothing else is seen.
    fn corpus() -> Vec<u32> {
        let mut v = vec![3u32; 10];
        v.extend(vec![1u32; 5]);
        v.extend(vec![2u32; 2]);
        v
    }

    #[test]
    fn ranks_by_descending_frequency() {
        let t = RankTable::train(&corpus(), None).unwrap();
        assert_eq!(t.k(), 3);
        assert_eq!(t.rank(3).unwrap(), 0);
        assert_eq!(t.rank(1).unwrap(), 1);
        assert_eq!(t.rank(2).unwrap(), 2);
    }

    #[test]
    fn breaks_ties_by_ascending_token_id() {
        let ids = vec![7u32, 7, 4, 4, 9, 9]; // all count 2
        let t = RankTable::train(&ids, None).unwrap();
        assert_eq!(t.token(0), 4);
        assert_eq!(t.token(1), 7);
        assert_eq!(t.token(2), 9);
    }

    #[test]
    fn training_is_deterministic() {
        let a = RankTable::train(&corpus(), None).unwrap();
        let b = RankTable::train(&corpus(), None).unwrap();
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn roundtrips_ranked_tokens() {
        let t = RankTable::train(&corpus(), None).unwrap();
        for tok in [1u32, 2, 3] {
            assert_eq!(t.token(t.rank(tok).unwrap()), tok);
        }
    }

    #[test]
    fn roundtrips_unranked_tokens_of_any_size() {
        // The property that removes vocabulary from the interface entirely: a token the table
        // never saw still roundtrips, whatever its ID.
        let t = RankTable::train(&corpus(), None).unwrap();
        for tok in [0u32, 4, 99, 50_000, 200_018, 1_000_000, u32::MAX - 3] {
            let r = t.rank(tok).unwrap();
            assert!(r >= t.k(), "unranked token {tok} should map at or above k");
            assert_eq!(t.token(r), tok, "unranked token {tok} did not roundtrip");
        }
    }

    #[test]
    fn ranked_and_unranked_ranges_never_collide() {
        let t = RankTable::train(&corpus(), None).unwrap();
        let k = t.k();
        for tok in 0..200u32 {
            let r = t.rank(tok).unwrap();
            assert_eq!(t.token(r), tok, "token {tok} did not roundtrip");
            assert_eq!(r < k, [1u32, 2, 3].contains(&tok));
        }
    }

    #[test]
    fn rejects_ids_that_would_overflow() {
        let t = RankTable::train(&corpus(), None).unwrap();
        let too_big = u32::MAX - 1;
        assert_eq!(t.rank(too_big), Err(TableError::IdTooLarge { id: too_big, k: 3 }));
    }

    #[test]
    fn max_ranks_caps_the_table() {
        let t = RankTable::train(&corpus(), Some(2)).unwrap();
        assert_eq!(t.k(), 2);
        assert_eq!(t.rank(3).unwrap(), 0);
        assert_eq!(t.rank(1).unwrap(), 1);
        // Token 2 fell outside the cap and now takes the fallback path, still losslessly.
        assert!(t.rank(2).unwrap() >= t.k());
        assert_eq!(t.token(t.rank(2).unwrap()), 2);
    }

    #[test]
    fn roundtrips_through_bytes() {
        let t = RankTable::train(&corpus(), None).unwrap();
        let bytes = t.to_bytes();
        let back = RankTable::from_bytes(&bytes).expect("should load");
        assert_eq!(back, t);
        assert_eq!(back.to_bytes(), bytes);
    }

    #[test]
    fn table_file_scales_with_observed_tokens_not_vocabulary() {
        // 3 distinct tokens seen -> a 12-byte header plus 3 x u32, regardless of how large the
        // tokenizer's vocabulary happens to be.
        let t = RankTable::train(&corpus(), None).unwrap();
        assert_eq!(t.to_bytes().len(), TABLE_HEADER_LEN + 3 * 4);
    }

    #[test]
    fn rejects_empty_corpus() {
        assert_eq!(RankTable::train(&[], None), Err(TableError::Empty));
    }

    #[test]
    fn rejects_duplicate_token_in_file() {
        let t = RankTable::train(&corpus(), None).unwrap();
        let mut bytes = t.to_bytes();
        let a = TABLE_HEADER_LEN;
        let dup: Vec<u8> = bytes[a..a + 4].to_vec();
        bytes[a + 4..a + 8].copy_from_slice(&dup);
        assert!(matches!(RankTable::from_bytes(&bytes), Err(TableError::DuplicateToken(_))));
    }

    #[test]
    fn rejects_bad_magic_version_and_kind() {
        let base = RankTable::train(&corpus(), None).unwrap().to_bytes();

        let mut b = base.clone();
        b[0] = b'X';
        assert_eq!(RankTable::from_bytes(&b), Err(TableError::BadMagic));

        let mut b = base.clone();
        b[4] = 9;
        assert_eq!(RankTable::from_bytes(&b), Err(TableError::UnsupportedVersion(9)));

        let mut b = base.clone();
        b[5] = 7;
        assert_eq!(RankTable::from_bytes(&b), Err(TableError::UnknownKind(7)));

        let mut b = base;
        b[6] = 1;
        assert_eq!(RankTable::from_bytes(&b), Err(TableError::ReservedNotZero));
    }

    #[test]
    fn digest_is_stable_and_hex_is_64_chars() {
        let bytes = RankTable::train(&corpus(), None).unwrap().to_bytes();
        assert_eq!(table_digest(&bytes), table_digest(&bytes));
        assert_eq!(digest_hex(&table_digest(&bytes)).len(), 64);
    }
}
