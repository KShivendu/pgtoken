//! The stored value: header plus encoded payload, and the operations over it.
//!
//! This is the API the Postgres extension and an agent client both call. Everything below
//! is lossless, and every function that reads a stored value validates it first rather than
//! trusting bytes that came off disk or off the wire.

use crate::codec::{ans, freq, raw};
use crate::header::{Codec, Header, HeaderError, Tokenizer};
use crate::tables::{AnsTable, RankTable};
use crate::tokenizer::{self, TokenizerError};

/// Coding tables available to encode or decode a value.
///
/// A raw-codec value needs neither, so both are optional; asking for `+freq` or `+ANS`
/// without the matching table is an error rather than a silent fallback.
#[derive(Debug, Default, Clone, Copy)]
pub struct Tables<'a> {
    pub rank: Option<&'a RankTable>,
    pub ans: Option<&'a AnsTable>,
}

impl<'a> Tables<'a> {
    pub fn none() -> Self {
        Tables { rank: None, ans: None }
    }
    pub fn with_rank(rank: &'a RankTable) -> Self {
        Tables { rank: Some(rank), ans: None }
    }
    pub fn with_ans(ans: &'a AnsTable) -> Self {
        Tables { rank: None, ans: Some(ans) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    Header(HeaderError),
    Tokenizer(TokenizerError),
    Raw(raw::RawError),
    Freq(freq::FreqError),
    Ans(ans::AnsError),
    /// The codec needs a table that was not supplied.
    MissingTable(Codec),
    /// The supplied table is for a different tokenizer than the value.
    TableTokenizerMismatch { table: Tokenizer, value: Tokenizer },
}

impl std::fmt::Display for ValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueError::Header(e) => write!(f, "{e}"),
            ValueError::Tokenizer(e) => write!(f, "{e}"),
            ValueError::Raw(e) => write!(f, "{e}"),
            ValueError::Freq(e) => write!(f, "{e}"),
            ValueError::Ans(e) => write!(f, "{e}"),
            ValueError::MissingTable(c) => {
                write!(f, "codec {} needs a coding table, none was supplied", c.as_str())
            }
            ValueError::TableTokenizerMismatch { table, value } => write!(
                f,
                "table is for {} but the value is {}",
                table.as_str(),
                value.as_str()
            ),
        }
    }
}

impl std::error::Error for ValueError {}

impl From<HeaderError> for ValueError {
    fn from(e: HeaderError) -> Self {
        ValueError::Header(e)
    }
}
impl From<TokenizerError> for ValueError {
    fn from(e: TokenizerError) -> Self {
        ValueError::Tokenizer(e)
    }
}
impl From<raw::RawError> for ValueError {
    fn from(e: raw::RawError) -> Self {
        ValueError::Raw(e)
    }
}
impl From<freq::FreqError> for ValueError {
    fn from(e: freq::FreqError) -> Self {
        ValueError::Freq(e)
    }
}
impl From<ans::AnsError> for ValueError {
    fn from(e: ans::AnsError) -> Self {
        ValueError::Ans(e)
    }
}

/// Resolve a codec name, mapping the convenience name `raw` to the narrowest packing this
/// tokenizer allows.
pub fn resolve_codec(name: &str, tokenizer: Tokenizer) -> Result<Codec, HeaderError> {
    if name == "raw" {
        return Ok(raw::preferred_raw(tokenizer));
    }
    Codec::parse(name)
}

/// Encode token IDs into a stored value.
///
/// This is the agent write path: a model has already produced the IDs, so no tokenizer is
/// touched at all.
pub fn encode_ids(
    ids: &[u32],
    tokenizer: Tokenizer,
    codec: Codec,
    table_id: u16,
    tables: Tables<'_>,
) -> Result<Vec<u8>, ValueError> {
    raw::validate_ids(ids, tokenizer)?;

    let n_tokens = u32::try_from(ids.len()).map_err(|_| {
        ValueError::Header(HeaderError::PayloadLen { want: u32::MAX as usize, got: ids.len() })
    })?;

    let mut out = Vec::with_capacity(crate::header::HEADER_LEN + ids.len() * 2);
    Header::new(tokenizer, codec, table_id, n_tokens).write_to(&mut out);

    match codec {
        Codec::Raw16 => raw::encode16(ids, &mut out)?,
        Codec::Raw24 => raw::encode24(ids, &mut out)?,
        Codec::Freq => {
            let t = tables.rank.ok_or(ValueError::MissingTable(codec))?;
            check_table_tokenizer(t.tokenizer, tokenizer)?;
            freq::encode_freq(ids, t, &mut out)?
        }
        Codec::Ans => {
            let t = tables.ans.ok_or(ValueError::MissingTable(codec))?;
            check_table_tokenizer(t.tokenizer, tokenizer)?;
            ans::encode_ans(ids, t, &mut out)?
        }
    }
    Ok(out)
}

/// Tokenize text and encode it. The human write path; costs a tokenize pass.
pub fn encode_text(
    text: &str,
    tokenizer: Tokenizer,
    codec: Codec,
    table_id: u16,
    tables: Tables<'_>,
) -> Result<Vec<u8>, ValueError> {
    let ids = tokenizer::encode(text, tokenizer);
    encode_ids(&ids, tokenizer, codec, table_id, tables)
}

/// Decode a stored value back to token IDs.
///
/// The agent read path. No tokenizer is touched; `+freq` and `+ANS` need only their table.
pub fn decode_ids(value: &[u8], tables: Tables<'_>) -> Result<Vec<u32>, ValueError> {
    let (h, payload) = Header::parse(value)?;
    let n = h.n_tokens as usize;
    let ids = match h.codec {
        Codec::Raw16 => raw::decode16(payload, n)?,
        Codec::Raw24 => raw::decode24(payload, n)?,
        Codec::Freq => {
            let t = tables.rank.ok_or(ValueError::MissingTable(h.codec))?;
            check_table_tokenizer(t.tokenizer, h.tokenizer)?;
            freq::decode_freq(payload, n, t)?
        }
        Codec::Ans => {
            let t = tables.ans.ok_or(ValueError::MissingTable(h.codec))?;
            check_table_tokenizer(t.tokenizer, h.tokenizer)?;
            ans::decode_ans(payload, n, t)?
        }
    };
    Ok(ids)
}

/// Decode a stored value back to text. The human read path; costs a detokenize pass.
pub fn decode_text(value: &[u8], tables: Tables<'_>) -> Result<String, ValueError> {
    let (h, _) = Header::parse(value)?;
    let ids = decode_ids(value, tables)?;
    Ok(tokenizer::decode(&ids, h.tokenizer)?)
}

/// Re-encode a value under a different codec without going through text.
///
/// Cheaper and safer than detokenize-then-retokenize: the token IDs are the invariant, so
/// this cannot change the text even if the tokenizer's behaviour ever shifts.
pub fn recode(
    value: &[u8],
    to_codec: Codec,
    to_table_id: u16,
    from_tables: Tables<'_>,
    to_tables: Tables<'_>,
) -> Result<Vec<u8>, ValueError> {
    let (h, _) = Header::parse(value)?;
    let ids = decode_ids(value, from_tables)?;
    encode_ids(&ids, h.tokenizer, to_codec, to_table_id, to_tables)
}

/// Header-only inspection. O(1): reads 12 bytes and loads no table.
pub fn describe(value: &[u8]) -> Result<(Header, usize), ValueError> {
    let (h, payload) = Header::parse(value)?;
    Ok((h, payload.len()))
}

fn check_table_tokenizer(table: Tokenizer, value: Tokenizer) -> Result<(), ValueError> {
    if table != value {
        return Err(ValueError::TableTokenizerMismatch { table, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXTS: &[&str] = &[
        "",
        "hello world",
        "Token-native storage: read and write in your agent's language.",
        "भारत में वेक्टर डेटाबेस",
        "🚀 café naïve",
        "def f(x):\n    return {'a': [1, 2, 3]}\n",
    ];

    /// Tables trained on the text under test, which is enough to exercise the codecs.
    fn tables_for(texts: &[&str], tok: Tokenizer) -> (RankTable, AnsTable) {
        let mut ids = Vec::new();
        for t in texts {
            ids.extend(tokenizer::encode(t, tok));
        }
        (RankTable::train(&ids, tok), AnsTable::train(&ids, tok))
    }

    fn all_codecs(tok: Tokenizer) -> Vec<Codec> {
        let mut v = vec![Codec::Raw24, Codec::Freq, Codec::Ans];
        if tok.fits_u16() {
            v.push(Codec::Raw16);
        }
        v
    }

    #[test]
    fn text_roundtrips_across_every_tokenizer_and_codec() {
        for tok in [Tokenizer::R50k, Tokenizer::Cl100k, Tokenizer::O200k] {
            let (rank, ans_t) = tables_for(TEXTS, tok);
            for codec in all_codecs(tok) {
                for text in TEXTS {
                    let (tables, table_id) = match codec {
                        Codec::Freq => (Tables::with_rank(&rank), 1),
                        Codec::Ans => (Tables::with_ans(&ans_t), 1),
                        _ => (Tables::none(), 0),
                    };
                    let v = encode_text(text, tok, codec, table_id, tables)
                        .unwrap_or_else(|e| panic!("encode {} {}: {e}", tok.as_str(), codec.as_str()));
                    let back = decode_text(&v, tables).unwrap_or_else(|e| {
                        panic!("decode {} {}: {e}", tok.as_str(), codec.as_str())
                    });
                    assert_eq!(
                        &back, text,
                        "roundtrip failed: {} / {}",
                        tok.as_str(),
                        codec.as_str()
                    );
                }
            }
        }
    }

    #[test]
    fn encoding_is_canonical() {
        // The property that makes `=`, `GROUP BY`, `DISTINCT` and hash joins correct on a
        // token-native column: identical text must produce byte-identical values.
        let tok = Tokenizer::O200k;
        let (rank, ans_t) = tables_for(TEXTS, tok);
        for codec in all_codecs(tok) {
            let (tables, table_id) = match codec {
                Codec::Freq => (Tables::with_rank(&rank), 1),
                Codec::Ans => (Tables::with_ans(&ans_t), 1),
                _ => (Tables::none(), 0),
            };
            for text in TEXTS {
                let first = encode_text(text, tok, codec, table_id, tables).unwrap();
                for _ in 0..64 {
                    assert_eq!(
                        encode_text(text, tok, codec, table_id, tables).unwrap(),
                        first,
                        "non-canonical output for {}",
                        codec.as_str()
                    );
                }
            }
        }
    }

    #[test]
    fn agent_path_needs_no_tokenizer_and_matches_text_path() {
        let tok = Tokenizer::O200k;
        let (rank, _) = tables_for(TEXTS, tok);
        let text = "Token-native storage";
        let ids = tokenizer::encode(text, tok);

        let via_ids = encode_ids(&ids, tok, Codec::Freq, 1, Tables::with_rank(&rank)).unwrap();
        let via_text = encode_text(text, tok, Codec::Freq, 1, Tables::with_rank(&rank)).unwrap();
        assert_eq!(via_ids, via_text);
        assert_eq!(decode_ids(&via_ids, Tables::with_rank(&rank)).unwrap(), ids);
    }

    #[test]
    fn recode_preserves_ids_across_every_codec_pair() {
        let tok = Tokenizer::O200k;
        let (rank, ans_t) = tables_for(TEXTS, tok);
        let text = "Token-native storage: read and write in your agent's language.";
        let ids = tokenizer::encode(text, tok);

        let both = Tables { rank: Some(&rank), ans: Some(&ans_t) };
        for from in all_codecs(tok) {
            for to in all_codecs(tok) {
                let from_id = if from.needs_table() { 1 } else { 0 };
                let to_id = if to.needs_table() { 1 } else { 0 };
                let v = encode_ids(&ids, tok, from, from_id, both).unwrap();
                let r = recode(&v, to, to_id, both, both).unwrap_or_else(|e| {
                    panic!("recode {} -> {}: {e}", from.as_str(), to.as_str())
                });
                assert_eq!(
                    decode_ids(&r, both).unwrap(),
                    ids,
                    "recode {} -> {} lost ids",
                    from.as_str(),
                    to.as_str()
                );
                assert_eq!(decode_text(&r, both).unwrap(), text);
            }
        }
    }

    #[test]
    fn describe_reads_only_the_header() {
        let tok = Tokenizer::O200k;
        let ids = vec![1u32, 2, 3, 4, 5];
        let v = encode_ids(&ids, tok, Codec::Raw24, 0, Tables::none()).unwrap();
        // No tables passed at all, yet describe works.
        let (h, payload_len) = describe(&v).unwrap();
        assert_eq!(h.n_tokens, 5);
        assert_eq!(h.tokenizer, tok);
        assert_eq!(h.codec, Codec::Raw24);
        assert_eq!(payload_len, 15);
    }

    #[test]
    fn table_codecs_refuse_to_run_without_a_table() {
        let tok = Tokenizer::O200k;
        let ids = vec![1u32, 2, 3];
        assert_eq!(
            encode_ids(&ids, tok, Codec::Freq, 1, Tables::none()),
            Err(ValueError::MissingTable(Codec::Freq))
        );
        assert_eq!(
            encode_ids(&ids, tok, Codec::Ans, 1, Tables::none()),
            Err(ValueError::MissingTable(Codec::Ans))
        );
    }

    #[test]
    fn refuses_a_table_from_a_different_tokenizer() {
        let (rank_r50k, _) = tables_for(TEXTS, Tokenizer::R50k);
        let ids = vec![1u32, 2, 3];
        assert_eq!(
            encode_ids(&ids, Tokenizer::O200k, Codec::Freq, 1, Tables::with_rank(&rank_r50k)),
            Err(ValueError::TableTokenizerMismatch {
                table: Tokenizer::R50k,
                value: Tokenizer::O200k
            })
        );
    }

    #[test]
    fn resolve_codec_maps_raw_to_the_narrowest_packing() {
        assert_eq!(resolve_codec("raw", Tokenizer::R50k).unwrap(), Codec::Raw16);
        assert_eq!(resolve_codec("raw", Tokenizer::O200k).unwrap(), Codec::Raw24);
        assert_eq!(resolve_codec("freq", Tokenizer::O200k).unwrap(), Codec::Freq);
        assert!(resolve_codec("nope", Tokenizer::O200k).is_err());
    }

    #[test]
    fn rejects_out_of_vocabulary_ids_before_writing() {
        let ids = vec![50_257u32];
        assert!(matches!(
            encode_ids(&ids, Tokenizer::R50k, Codec::Raw16, 0, Tables::none()),
            Err(ValueError::Raw(raw::RawError::IdOutOfRange { .. }))
        ));
    }

    #[test]
    fn empty_text_roundtrips_with_a_bare_header() {
        let tok = Tokenizer::O200k;
        let v = encode_text("", tok, Codec::Raw24, 0, Tables::none()).unwrap();
        assert_eq!(v.len(), crate::header::HEADER_LEN);
        assert_eq!(decode_text(&v, Tables::none()).unwrap(), "");
    }
}
