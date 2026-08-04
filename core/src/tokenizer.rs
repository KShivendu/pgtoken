//! Tokenizer access, and the text <-> token-ID boundary.
//!
//! `tiktoken-rs` exposes each vocabulary as a `lazy_static` singleton, which gives exactly
//! the initialisation behaviour the design wants inside Postgres: the multi-megabyte BPE
//! table is built on first use in a backend and then shared by every later call in that
//! backend. A pure-agent workload, where the client encodes and the server only stores
//! blobs, never touches a tokenizer here and so never pays for one.
//!
//! Only `encode_ordinary` is used. The `encode_with_special_tokens` variants would let a
//! string like `"<|endoftext|>"` in ordinary user text collapse into a single special-token
//! ID, which then decodes back to that literal — fine for prompting, wrong for storage,
//! because it makes the roundtrip depend on whether the text happens to contain a control
//! string. `encode_ordinary` treats all input as ordinary text.

use tiktoken_rs::{cl100k_base_singleton, o200k_base_singleton, r50k_base_singleton};
use tiktoken_rs::CoreBPE;

use crate::header::Tokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// A token ID has no entry in the vocabulary.
    UnknownToken(u32),
    /// The decoded bytes were not valid UTF-8.
    ///
    /// Reachable only from a token sequence that was not produced by encoding a complete
    /// string, e.g. one truncated mid-character by a caller.
    NotUtf8,
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::UnknownToken(t) => write!(f, "token id {t} is not in the vocabulary"),
            TokenizerError::NotUtf8 => {
                write!(f, "decoded token IDs are not valid UTF-8; the sequence is not a whole text")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

/// Borrow the process-wide tokenizer, building it on first use.
pub fn bpe(tokenizer: Tokenizer) -> &'static CoreBPE {
    match tokenizer {
        Tokenizer::R50k => r50k_base_singleton(),
        Tokenizer::Cl100k => cl100k_base_singleton(),
        Tokenizer::O200k => o200k_base_singleton(),
    }
}

/// Text -> token IDs.
pub fn encode(text: &str, tokenizer: Tokenizer) -> Vec<u32> {
    bpe(tokenizer).encode_ordinary(text)
}

/// Token IDs -> text.
pub fn decode(ids: &[u32], tokenizer: Tokenizer) -> Result<String, TokenizerError> {
    let bytes = bpe(tokenizer)
        .decode_bytes(ids)
        .map_err(|e| TokenizerError::UnknownToken(e.token))?;
    String::from_utf8(bytes).map_err(|_| TokenizerError::NotUtf8)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CASES: &[&str] = &[
        "",
        " ",
        "hello world",
        "Token-native storage",
        // Devanagari: r50k has no merges for this script, the paper's worst case. Must
        // still be lossless, only larger.
        "भारत में वेक्टर डेटाबेस",
        // Emoji and combining characters: multi-byte sequences BPE may split across tokens.
        "🚀 café naïve résumé 👨‍👩‍👧‍👦",
        // Code, including the runs of punctuation that hurt raw packing.
        "def f(x):\n    return {'a': [1, 2, 3], 'b': None}\n",
        // A literal special-token string, which must not collapse into one special ID.
        "<|endoftext|>",
        "tabs\tand\r\nnewlines",
    ];

    #[test]
    fn roundtrips_every_tokenizer() {
        for tok in [Tokenizer::R50k, Tokenizer::Cl100k, Tokenizer::O200k] {
            for case in CASES {
                let ids = encode(case, tok);
                let back = decode(&ids, tok).expect("decode should succeed");
                assert_eq!(&back, case, "roundtrip failed for {} on {case:?}", tok.as_str());
            }
        }
    }

    #[test]
    fn ids_are_within_declared_vocabulary() {
        // The vocab sizes in `Tokenizer::vocab` set the width of every trained table, so an
        // ID past the end would silently corrupt the rank and ANS tables.
        for tok in [Tokenizer::R50k, Tokenizer::Cl100k, Tokenizer::O200k] {
            for case in CASES {
                for id in encode(case, tok) {
                    assert!(
                        id < tok.vocab(),
                        "{} produced id {id}, past declared vocab {}",
                        tok.as_str(),
                        tok.vocab()
                    );
                }
            }
        }
    }

    #[test]
    fn special_token_text_does_not_collapse() {
        // "<|endoftext|>" must encode as ordinary text (several tokens), not as the single
        // special-token ID, or the roundtrip would depend on the text's content.
        for tok in [Tokenizer::R50k, Tokenizer::Cl100k, Tokenizer::O200k] {
            let ids = encode("<|endoftext|>", tok);
            assert!(ids.len() > 1, "{} collapsed the literal to {ids:?}", tok.as_str());
            assert_eq!(decode(&ids, tok).unwrap(), "<|endoftext|>");
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        // Canonicality at the storage layer depends on this: same text, same IDs, therefore
        // same bytes. Without it, `=` and `GROUP BY` on a token-native column are wrong.
        for tok in [Tokenizer::R50k, Tokenizer::Cl100k, Tokenizer::O200k] {
            for case in CASES {
                let a = encode(case, tok);
                for _ in 0..8 {
                    assert_eq!(encode(case, tok), a, "nondeterministic for {}", tok.as_str());
                }
            }
        }
    }

    #[test]
    fn rejects_unknown_token_id() {
        let err = decode(&[u32::MAX], Tokenizer::O200k);
        assert_eq!(err, Err(TokenizerError::UnknownToken(u32::MAX)));
    }

    #[test]
    fn empty_input_gives_empty_output() {
        for tok in [Tokenizer::R50k, Tokenizer::Cl100k, Tokenizer::O200k] {
            assert!(encode("", tok).is_empty());
            assert_eq!(decode(&[], tok).unwrap(), "");
        }
    }

    #[test]
    fn hindi_costs_more_tokens_on_r50k_than_o200k() {
        // Documents the paper's tokenizer-coverage limitation as an executable fact: r50k
        // never learned Devanagari merges, so it splits the script far more finely.
        let hindi = "भारत में वेक्टर डेटाबेस का उपयोग";
        let r50k = encode(hindi, Tokenizer::R50k).len();
        let o200k = encode(hindi, Tokenizer::O200k).len();
        assert!(r50k > o200k, "expected r50k ({r50k}) to need more tokens than o200k ({o200k})");
    }
}
