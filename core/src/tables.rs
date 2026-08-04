//! Trained coding tables: the frequency-rank permutation and the ANS unigram model.
//!
//! Ports `tnbench.build_rank_table` (tnbench.py:212) and `tnbench.build_ans_model`
//! (tnbench.py:223). Both are deterministic and contain no RNG draw, which the Python
//! versions call out explicitly; that property is preserved here and tested.
//!
//! Tables are content-addressed by SHA-256 and referenced from a value's header by a small
//! `table_id`. That indirection is what lets `decode` be `IMMUTABLE` in Postgres: the
//! stored bytes name the exact table needed to interpret them, so the same input always
//! produces the same output.
//!
//! # On-disk format
//!
//! ```text
//! off  size  field
//!   0     4  magic "TNTT"
//!   4     1  version (1)
//!   5     1  kind    1=rank 2=ans
//!   6     1  tokenizer id
//!   7     1  reserved, must be zero
//!   8     4  vocab (u32 LE)
//!  12     4  entry count (u32 LE, == vocab)
//!  16     -  payload: `vocab` x u32 LE
//! ```
//!
//! For a rank table the payload is `token_of_rank`; `rank_of` is its inverse and is rebuilt
//! on load rather than stored, which halves the file. For an ANS table the payload is the
//! Laplace-smoothed counts, from which the model is rebuilt. Storing counts rather than a
//! quantized CDF keeps the file independent of how a given `constriction` version chooses
//! to quantize, at the cost of doing that quantization once at load.

use sha2::{Digest, Sha256};

use crate::header::Tokenizer;

pub const TABLE_MAGIC: &[u8; 4] = b"TNTT";
pub const TABLE_VERSION: u8 = 1;
pub const TABLE_HEADER_LEN: usize = 16;

const KIND_RANK: u8 = 1;
const KIND_ANS: u8 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableError {
    TooShort(usize),
    BadMagic,
    UnsupportedVersion(u8),
    UnknownKind(u8),
    ReservedNotZero,
    /// The file says it is for a different tokenizer than the value that referenced it.
    TokenizerMismatch { table: u8, want: u8 },
    /// The declared vocabulary does not match the tokenizer's real vocabulary.
    VocabMismatch { table: u32, want: u32 },
    PayloadLen { want: usize, got: usize },
    /// `token_of_rank` was not a permutation of `0..vocab`.
    NotAPermutation,
    /// A count was zero, which would make an ANS symbol unencodable.
    ZeroCount(u32),
    /// `constriction` rejected the probability vector.
    ModelBuild,
}

impl std::fmt::Display for TableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableError::TooShort(n) => write!(f, "table file is only {n} bytes"),
            TableError::BadMagic => write!(f, "table file has bad magic, expected TNTT"),
            TableError::UnsupportedVersion(v) => write!(f, "unsupported table version {v}"),
            TableError::UnknownKind(k) => write!(f, "unknown table kind {k}"),
            TableError::ReservedNotZero => write!(f, "reserved table header byte is not zero"),
            TableError::TokenizerMismatch { table, want } => {
                write!(f, "table is for tokenizer id {table}, value needs {want}")
            }
            TableError::VocabMismatch { table, want } => {
                write!(f, "table declares vocab {table}, tokenizer has {want}")
            }
            TableError::PayloadLen { want, got } => {
                write!(f, "table payload is {got} bytes, expected {want}")
            }
            TableError::NotAPermutation => {
                write!(f, "token_of_rank is not a permutation of the vocabulary")
            }
            TableError::ZeroCount(id) => {
                write!(f, "ANS count for token {id} is zero; every symbol must be encodable")
            }
            TableError::ModelBuild => write!(f, "could not build the ANS model from the counts"),
        }
    }
}

impl std::error::Error for TableError {}

/// Count token occurrences into a `vocab`-sized histogram. Equivalent to
/// `np.bincount(ids, minlength=vocab)`, ignoring any out-of-range ID.
fn bincount(ids: &[u32], vocab: u32) -> Vec<u32> {
    let mut counts = vec![0u32; vocab as usize];
    for &id in ids {
        if id < vocab {
            counts[id as usize] = counts[id as usize].saturating_add(1);
        }
    }
    counts
}

fn write_table_header(out: &mut Vec<u8>, kind: u8, tokenizer: Tokenizer, vocab: u32) {
    out.extend_from_slice(TABLE_MAGIC);
    out.push(TABLE_VERSION);
    out.push(kind);
    out.push(tokenizer as u8);
    out.push(0); // reserved
    out.extend_from_slice(&vocab.to_le_bytes());
    out.extend_from_slice(&vocab.to_le_bytes());
    debug_assert_eq!(out.len(), TABLE_HEADER_LEN);
}

/// Validate a table file header and return `(kind, payload_as_u32)`.
fn parse_table(buf: &[u8], tokenizer: Tokenizer) -> Result<(u8, Vec<u32>), TableError> {
    if buf.len() < TABLE_HEADER_LEN {
        return Err(TableError::TooShort(buf.len()));
    }
    if &buf[0..4] != TABLE_MAGIC {
        return Err(TableError::BadMagic);
    }
    if buf[4] != TABLE_VERSION {
        return Err(TableError::UnsupportedVersion(buf[4]));
    }
    let kind = buf[5];
    if kind != KIND_RANK && kind != KIND_ANS {
        return Err(TableError::UnknownKind(kind));
    }
    if buf[6] != tokenizer as u8 {
        return Err(TableError::TokenizerMismatch { table: buf[6], want: tokenizer as u8 });
    }
    if buf[7] != 0 {
        return Err(TableError::ReservedNotZero);
    }
    let vocab = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    if vocab != tokenizer.vocab() {
        return Err(TableError::VocabMismatch { table: vocab, want: tokenizer.vocab() });
    }
    let n = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let payload = &buf[TABLE_HEADER_LEN..];
    if payload.len() != n * 4 {
        return Err(TableError::PayloadLen { want: n * 4, got: payload.len() });
    }
    let vals = payload
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((kind, vals))
}

/// SHA-256 of a table file, used to content-address it in the catalog.
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

/// The frequency-rank permutation behind the `+freq` codec.
///
/// BPE numbers its vocabulary in merge-discovery order, so a common token can sit at ID
/// 40,000 while a rare one sits at 400. Remapping each ID to its descending-frequency rank
/// puts the common tokens on small integers, which is what a varint codec rewards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankTable {
    pub tokenizer: Tokenizer,
    /// token id -> frequency rank
    rank_of: Vec<u32>,
    /// frequency rank -> token id
    token_of_rank: Vec<u32>,
}

impl RankTable {
    /// Train on a corpus of token IDs.
    ///
    /// Ties are broken by ascending token ID, which numpy's `argsort` does *not* guarantee
    /// (its default quicksort is unstable). Specifying the tiebreak makes the Rust table
    /// reproducible; it can differ from Python's in the zero-count tail, which affects
    /// neither correctness nor measured ratio.
    pub fn train(ids: &[u32], tokenizer: Tokenizer) -> Self {
        let vocab = tokenizer.vocab();
        let counts = bincount(ids, vocab);

        let mut order: Vec<u32> = (0..vocab).collect();
        order.sort_by(|&a, &b| {
            counts[b as usize]
                .cmp(&counts[a as usize]) // descending count
                .then(a.cmp(&b)) // ascending token id
        });

        let mut rank_of = vec![0u32; vocab as usize];
        for (rank, &tok) in order.iter().enumerate() {
            rank_of[tok as usize] = rank as u32;
        }
        RankTable { tokenizer, rank_of, token_of_rank: order }
    }

    pub fn vocab(&self) -> u32 {
        self.tokenizer.vocab()
    }

    #[inline]
    pub fn rank_of(&self, token: u32) -> Option<u32> {
        self.rank_of.get(token as usize).copied()
    }

    #[inline]
    pub fn token_of_rank(&self, rank: u32) -> Option<u32> {
        self.token_of_rank.get(rank as usize).copied()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(TABLE_HEADER_LEN + self.token_of_rank.len() * 4);
        write_table_header(&mut out, KIND_RANK, self.tokenizer, self.vocab());
        for &t in &self.token_of_rank {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(buf: &[u8], tokenizer: Tokenizer) -> Result<Self, TableError> {
        let (kind, token_of_rank) = parse_table(buf, tokenizer)?;
        if kind != KIND_RANK {
            return Err(TableError::UnknownKind(kind));
        }
        let vocab = tokenizer.vocab();

        // A corrupt permutation would silently map tokens onto each other, so verify it is
        // a true bijection rather than trusting the file.
        let mut rank_of = vec![u32::MAX; vocab as usize];
        for (rank, &tok) in token_of_rank.iter().enumerate() {
            if tok >= vocab || rank_of[tok as usize] != u32::MAX {
                return Err(TableError::NotAPermutation);
            }
            rank_of[tok as usize] = rank as u32;
        }
        if rank_of.contains(&u32::MAX) {
            return Err(TableError::NotAPermutation);
        }
        Ok(RankTable { tokenizer, rank_of, token_of_rank })
    }
}

/// The static Laplace-smoothed unigram model behind the `+ANS` codec.
///
/// Trained once on a corpus and shared across every document. A per-document table
/// compresses better but has to travel with each document (~900 B per 512-token chunk),
/// which erases the gain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnsTable {
    pub tokenizer: Tokenizer,
    /// Laplace-smoothed counts, one per vocabulary entry. Always >= 1.
    counts: Vec<u32>,
}

impl AnsTable {
    /// Train on a corpus of token IDs with Laplace `+1` smoothing.
    ///
    /// The `+1` is not cosmetic: ANS cannot encode a symbol with zero probability, so an
    /// unsmoothed table would fail on the first unseen token rather than compress worse.
    pub fn train(ids: &[u32], tokenizer: Tokenizer) -> Self {
        let vocab = tokenizer.vocab();
        let mut counts = bincount(ids, vocab);
        for c in counts.iter_mut() {
            *c += 1;
        }
        AnsTable { tokenizer, counts }
    }

    pub fn vocab(&self) -> u32 {
        self.tokenizer.vocab()
    }

    /// Normalized probabilities, matching `counts / counts.sum()` in the Python harness.
    pub fn probabilities(&self) -> Vec<f64> {
        let total: f64 = self.counts.iter().map(|&c| c as f64).sum();
        self.counts.iter().map(|&c| c as f64 / total).collect()
    }

    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(TABLE_HEADER_LEN + self.counts.len() * 4);
        write_table_header(&mut out, KIND_ANS, self.tokenizer, self.vocab());
        for &c in &self.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(buf: &[u8], tokenizer: Tokenizer) -> Result<Self, TableError> {
        let (kind, counts) = parse_table(buf, tokenizer)?;
        if kind != KIND_ANS {
            return Err(TableError::UnknownKind(kind));
        }
        if let Some(pos) = counts.iter().position(|&c| c == 0) {
            return Err(TableError::ZeroCount(pos as u32));
        }
        Ok(AnsTable { tokenizer, counts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny synthetic corpus: token 3 most common, then 1, then 2, rest unseen.
    fn corpus() -> Vec<u32> {
        let mut v = vec![3u32; 10];
        v.extend(vec![1u32; 5]);
        v.extend(vec![2u32; 2]);
        v
    }

    #[test]
    fn rank_table_orders_by_descending_frequency() {
        let t = RankTable::train(&corpus(), Tokenizer::R50k);
        assert_eq!(t.token_of_rank(0), Some(3));
        assert_eq!(t.token_of_rank(1), Some(1));
        assert_eq!(t.token_of_rank(2), Some(2));
        assert_eq!(t.rank_of(3), Some(0));
        assert_eq!(t.rank_of(1), Some(1));
    }

    #[test]
    fn rank_table_breaks_ties_by_ascending_token_id() {
        // Tokens 0 and 4..vocab all have count 0 and must come out in ID order.
        let t = RankTable::train(&corpus(), Tokenizer::R50k);
        assert_eq!(t.token_of_rank(3), Some(0));
        assert_eq!(t.token_of_rank(4), Some(4));
        assert_eq!(t.token_of_rank(5), Some(5));
    }

    #[test]
    fn rank_table_training_is_deterministic() {
        let a = RankTable::train(&corpus(), Tokenizer::R50k);
        let b = RankTable::train(&corpus(), Tokenizer::R50k);
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn rank_table_is_a_permutation() {
        let t = RankTable::train(&corpus(), Tokenizer::R50k);
        let vocab = Tokenizer::R50k.vocab();
        let mut seen = vec![false; vocab as usize];
        for r in 0..vocab {
            let tok = t.token_of_rank(r).expect("rank in range");
            assert!(!seen[tok as usize], "token {tok} appears twice");
            seen[tok as usize] = true;
            assert_eq!(t.rank_of(tok), Some(r), "rank_of is not the inverse");
        }
    }

    #[test]
    fn rank_table_roundtrips_through_bytes() {
        let t = RankTable::train(&corpus(), Tokenizer::O200k);
        let bytes = t.to_bytes();
        let back = RankTable::from_bytes(&bytes, Tokenizer::O200k).expect("should load");
        assert_eq!(back.to_bytes(), bytes);
        assert_eq!(back.rank_of(3), t.rank_of(3));
    }

    #[test]
    fn rank_table_rejects_wrong_tokenizer() {
        let bytes = RankTable::train(&corpus(), Tokenizer::R50k).to_bytes();
        assert_eq!(
            RankTable::from_bytes(&bytes, Tokenizer::O200k),
            Err(TableError::TokenizerMismatch { table: 1, want: 3 })
        );
    }

    #[test]
    fn rank_table_rejects_non_permutation() {
        let t = RankTable::train(&corpus(), Tokenizer::R50k);
        let mut bytes = t.to_bytes();
        // Duplicate rank 0's token into rank 1's slot.
        let a = TABLE_HEADER_LEN;
        let dup: Vec<u8> = bytes[a..a + 4].to_vec();
        bytes[a + 4..a + 8].copy_from_slice(&dup);
        assert_eq!(
            RankTable::from_bytes(&bytes, Tokenizer::R50k),
            Err(TableError::NotAPermutation)
        );
    }

    #[test]
    fn ans_table_applies_laplace_smoothing() {
        let t = AnsTable::train(&corpus(), Tokenizer::R50k);
        assert_eq!(t.counts()[3], 11); // 10 seen + 1
        assert_eq!(t.counts()[1], 6); //   5 seen + 1
        assert_eq!(t.counts()[0], 1); //   unseen, still encodable
        assert!(t.counts().iter().all(|&c| c >= 1), "every symbol must be encodable");
    }

    #[test]
    fn ans_probabilities_sum_to_one() {
        let t = AnsTable::train(&corpus(), Tokenizer::R50k);
        let sum: f64 = t.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "probabilities summed to {sum}");
    }

    #[test]
    fn ans_table_roundtrips_through_bytes() {
        let t = AnsTable::train(&corpus(), Tokenizer::Cl100k);
        let bytes = t.to_bytes();
        let back = AnsTable::from_bytes(&bytes, Tokenizer::Cl100k).expect("should load");
        assert_eq!(back.counts(), t.counts());
    }

    #[test]
    fn ans_table_rejects_zero_count() {
        let t = AnsTable::train(&corpus(), Tokenizer::R50k);
        let mut bytes = t.to_bytes();
        bytes[TABLE_HEADER_LEN..TABLE_HEADER_LEN + 4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            AnsTable::from_bytes(&bytes, Tokenizer::R50k),
            Err(TableError::ZeroCount(0))
        );
    }

    #[test]
    fn table_kinds_are_not_interchangeable() {
        let rank = RankTable::train(&corpus(), Tokenizer::R50k).to_bytes();
        assert_eq!(
            AnsTable::from_bytes(&rank, Tokenizer::R50k),
            Err(TableError::UnknownKind(KIND_RANK))
        );
        let ans = AnsTable::train(&corpus(), Tokenizer::R50k).to_bytes();
        assert_eq!(
            RankTable::from_bytes(&ans, Tokenizer::R50k),
            Err(TableError::UnknownKind(KIND_ANS))
        );
    }

    #[test]
    fn digest_is_stable_and_hex_is_64_chars() {
        let bytes = AnsTable::train(&corpus(), Tokenizer::R50k).to_bytes();
        let d1 = table_digest(&bytes);
        let d2 = table_digest(&bytes);
        assert_eq!(d1, d2);
        assert_eq!(digest_hex(&d1).len(), 64);
    }

    #[test]
    fn table_rejects_bad_magic_and_version() {
        let mut bytes = AnsTable::train(&corpus(), Tokenizer::R50k).to_bytes();
        bytes[0] = b'X';
        assert_eq!(AnsTable::from_bytes(&bytes, Tokenizer::R50k), Err(TableError::BadMagic));

        let mut bytes = AnsTable::train(&corpus(), Tokenizer::R50k).to_bytes();
        bytes[4] = 9;
        assert_eq!(
            AnsTable::from_bytes(&bytes, Tokenizer::R50k),
            Err(TableError::UnsupportedVersion(9))
        );
    }
}
