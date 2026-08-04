//! `+ANS`: static Laplace-smoothed unigram entropy coding over token IDs.
//!
//! Ports the encode/decode halves of the Python harness's ANS path, which builds its model
//! with `constriction.stream.model.Categorical(probs, perfect=False)` and codes with
//! `AnsCoder.encode_reverse` (see tnbench.py:223 and the blog's reference implementation).
//!
//! Two details make this cross-compatible with the Python side, and both are load-bearing:
//!
//! - `perfect=False` on the Python side corresponds to
//!   `from_floating_point_probabilities_fast` here. The `_perfect` variant quantizes
//!   differently and produces a different, mutually undecodable table.
//! - rANS is a stack, so symbols must be *pushed* in reverse to *pop* in forward order.
//!   Python calls `encode_reverse`; here that is `encode_iid_symbols_reverse`.
//!
//! This codec gives the best ratio of the four (3.40x on English/o200k) but the slowest
//! decode (~32 us vs ~4 us for `+freq`), which is why `+freq` is the default.

use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::stack::DefaultAnsCoder;
use constriction::stream::Decode;

use crate::tables::{AnsTable, TableError};

/// Word type of `DefaultAnsCoder`; the compressed buffer is a sequence of these.
type Word = u32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnsError {
    IdOutOfRange { id: u32, vocab: u32 },
    /// The payload length is not a whole number of coder words.
    NotWordAligned { len: usize },
    /// `constriction` could not build the model from the table's counts.
    ModelBuild,
    /// The coder rejected the compressed buffer or ran out of data mid-symbol.
    Corrupt(String),
}

impl std::fmt::Display for AnsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnsError::IdOutOfRange { id, vocab } => {
                write!(f, "token id {id} is outside the table vocabulary ({vocab})")
            }
            AnsError::NotWordAligned { len } => {
                write!(f, "ANS payload is {len} bytes, not a multiple of {}", size_of::<Word>())
            }
            AnsError::ModelBuild => write!(f, "could not build the ANS model from the table"),
            AnsError::Corrupt(m) => write!(f, "ANS payload is corrupt: {m}"),
        }
    }
}

impl std::error::Error for AnsError {}

impl From<TableError> for AnsError {
    fn from(_: TableError) -> Self {
        AnsError::ModelBuild
    }
}

/// Build the entropy model from a table's counts.
///
/// Kept private and rebuilt per call site rather than cached in the table, because the model
/// borrows its CDF; callers that care about the cost should hold the returned model.
pub fn build_model(
    table: &AnsTable,
) -> Result<DefaultContiguousCategoricalEntropyModel, AnsError> {
    let probs = table.probabilities();
    // `None` normalization means "these already sum to 1", matching the Python harness,
    // which passes normalized probabilities.
    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
        &probs, None,
    )
    .map_err(|_| AnsError::ModelBuild)
}

/// Encode token IDs with the static ANS model.
pub fn encode_ans(ids: &[u32], table: &AnsTable, out: &mut Vec<u8>) -> Result<(), AnsError> {
    if ids.is_empty() {
        return Ok(());
    }
    let vocab = table.vocab();
    for &id in ids {
        if id >= vocab {
            return Err(AnsError::IdOutOfRange { id, vocab });
        }
    }
    let model = build_model(table)?;

    let mut coder = DefaultAnsCoder::new();
    let symbols: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
    coder
        .encode_iid_symbols_reverse(&symbols, &model)
        .map_err(|e| AnsError::Corrupt(format!("{e:?}")))?;

    let words = coder.into_compressed().map_err(|e| AnsError::Corrupt(format!("{e:?}")))?;
    out.reserve(words.len() * size_of::<Word>());
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    Ok(())
}

/// Decode an `+ANS` payload back to token IDs.
pub fn decode_ans(payload: &[u8], n: usize, table: &AnsTable) -> Result<Vec<u32>, AnsError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if !payload.len().is_multiple_of(size_of::<Word>()) {
        return Err(AnsError::NotWordAligned { len: payload.len() });
    }
    let words: Vec<Word> = payload
        .chunks_exact(size_of::<Word>())
        .map(|c| Word::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let model = build_model(table)?;
    let mut coder = DefaultAnsCoder::from_compressed(words)
        .map_err(|_| AnsError::Corrupt("not a valid ANS buffer".into()))?;

    let mut ids = Vec::with_capacity(n);
    for sym in coder.decode_iid_symbols(n, &model) {
        let sym = sym.map_err(|e| AnsError::Corrupt(format!("{e:?}")))?;
        ids.push(sym as u32);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Tokenizer;

    fn table(tok: Tokenizer) -> AnsTable {
        let mut corpus = vec![3u32; 1000];
        corpus.extend(vec![1u32; 500]);
        corpus.extend(vec![2u32; 200]);
        corpus.extend(vec![7u32; 50]);
        AnsTable::train(&corpus, tok)
    }

    fn roundtrip(ids: &[u32], t: &AnsTable) -> Vec<u32> {
        let mut out = Vec::new();
        encode_ans(ids, t, &mut out).expect("encode");
        decode_ans(&out, ids.len(), t).expect("decode")
    }

    #[test]
    fn roundtrips_in_forward_order() {
        // The critical property: rANS is a stack, so a missing `reverse` would hand back
        // the sequence backwards. An asymmetric input catches that; a palindrome would not.
        let t = table(Tokenizer::O200k);
        let ids = vec![3, 1, 1, 2, 7, 3];
        assert_eq!(roundtrip(&ids, &t), ids);
    }

    #[test]
    fn roundtrips_unseen_tokens() {
        // Laplace smoothing means a token absent from the training corpus is still
        // encodable, just expensively.
        let t = table(Tokenizer::O200k);
        let ids = vec![3, 199_999, 42, 1];
        assert_eq!(roundtrip(&ids, &t), ids);
    }

    #[test]
    fn roundtrips_across_tokenizers() {
        for tok in [Tokenizer::R50k, Tokenizer::Cl100k, Tokenizer::O200k] {
            let t = table(tok);
            let ids = vec![3, 1, 2, 7, 0];
            assert_eq!(roundtrip(&ids, &t), ids, "failed for {}", tok.as_str());
        }
    }

    #[test]
    fn roundtrips_long_sequence() {
        let t = table(Tokenizer::O200k);
        let ids: Vec<u32> = (0..512).map(|i| [3u32, 1, 2, 7][(i % 4) as usize]).collect();
        assert_eq!(roundtrip(&ids, &t), ids);
    }

    #[test]
    fn empty_roundtrips() {
        let t = table(Tokenizer::O200k);
        let mut out = Vec::new();
        encode_ans(&[], &t, &mut out).expect("encode");
        assert!(out.is_empty());
        assert_eq!(decode_ans(&out, 0, &t).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn compresses_frequent_tokens_below_raw_packing() {
        // 512 copies of the most frequent token should cost far less than 3 bytes each.
        let t = table(Tokenizer::O200k);
        let ids = vec![3u32; 512];
        let mut out = Vec::new();
        encode_ans(&ids, &t, &mut out).expect("encode");
        assert!(out.len() < ids.len(), "expected under 1 byte/token, got {} bytes", out.len());
        assert_eq!(decode_ans(&out, ids.len(), &t).unwrap(), ids);
    }

    #[test]
    fn rejects_id_outside_vocabulary() {
        let t = table(Tokenizer::R50k);
        let mut out = Vec::new();
        assert_eq!(
            encode_ans(&[50_257], &t, &mut out),
            Err(AnsError::IdOutOfRange { id: 50_257, vocab: 50_257 })
        );
    }

    #[test]
    fn rejects_misaligned_payload() {
        let t = table(Tokenizer::O200k);
        assert_eq!(
            decode_ans(&[0, 1, 2], 1, &t),
            Err(AnsError::NotWordAligned { len: 3 })
        );
    }

    #[test]
    fn decoding_with_the_wrong_table_does_not_return_the_original() {
        // Not a security property, just a correctness one: the table is part of the
        // contract, which is why the value header names it.
        let t1 = table(Tokenizer::O200k);
        let mut other = vec![9u32; 1000];
        other.extend(vec![11u32; 500]);
        let t2 = AnsTable::train(&other, Tokenizer::O200k);

        let ids = vec![3, 1, 2, 7];
        let mut out = Vec::new();
        encode_ans(&ids, &t1, &mut out).expect("encode");
        let decoded = decode_ans(&out, ids.len(), &t2);
        // It may error or return different IDs; it must not return the originals.
        if let Ok(got) = decoded {
            assert_ne!(got, ids, "wrong table decoded to the original IDs");
        }
    }
}
