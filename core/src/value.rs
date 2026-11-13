//! The stored value: header plus encoded payload.
//!
//! Everything here operates on token IDs. Turning text into IDs is the caller's job, done
//! with whatever tokenizer they already use — that stays outside this library on purpose.

use std::borrow::Cow;

use crate::codec::{freq, freq_lz4, raw};
#[cfg(test)]
use crate::header::VarintError;
use crate::header::{Codec, Header, HeaderError, MAX_HEADER_LEN, MAX_TOKENS};
use crate::tables::RankTable;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    Header(HeaderError),
    Raw(raw::RawError),
    Freq(freq::FreqError),
    /// The codec needs a coding table that was not supplied.
    MissingTable(Codec),
    /// More IDs than one value may hold ([`MAX_TOKENS`]).
    TooManyTokens(usize),
}

impl std::fmt::Display for ValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueError::Header(e) => write!(f, "{e}"),
            ValueError::Raw(e) => write!(f, "{e}"),
            ValueError::Freq(e) => write!(f, "{e}"),
            ValueError::MissingTable(c) => {
                write!(
                    f,
                    "codec {} needs a coding table, none was supplied",
                    c.as_str()
                )
            }
            ValueError::TooManyTokens(n) => {
                write!(
                    f,
                    "{n} tokens is more than the {MAX_TOKENS} one value may hold"
                )
            }
        }
    }
}

impl std::error::Error for ValueError {}

impl From<HeaderError> for ValueError {
    fn from(e: HeaderError) -> Self {
        ValueError::Header(e)
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

/// Encode token IDs into a stored value.
///
/// Asking for `freq` (or `freq_lz4`) writes whichever of the two is smaller, so the header may
/// say `freq_lz4` though `freq` was asked for. Both decode the same way and compare equal.
pub fn encode(
    ids: &[u32],
    codec: Codec,
    vocabulary_id: u16,
    table: Option<&RankTable>,
) -> Result<Vec<u8>, ValueError> {
    let n_tokens = u32::try_from(ids.len())
        .ok()
        .filter(|&n| n <= MAX_TOKENS)
        .ok_or(ValueError::TooManyTokens(ids.len()))?;
    let mut out = Vec::with_capacity(MAX_HEADER_LEN + ids.len() * 2);

    match codec {
        Codec::Raw8 | Codec::Raw16 | Codec::Raw24 => {
            Header::new(codec, vocabulary_id, n_tokens).write_to(&mut out);
            match codec {
                Codec::Raw8 => raw::encode8(ids, &mut out)?,
                Codec::Raw16 => raw::encode16(ids, &mut out)?,
                _ => raw::encode24(ids, &mut out)?,
            }
        }
        Codec::Freq | Codec::FreqLz4 => {
            let t = table.ok_or(ValueError::MissingTable(codec))?;
            let mut payload = Vec::with_capacity(ids.len() * 2);
            freq::encode_freq(ids, t, &mut payload)?;
            match freq_lz4::compress(&payload) {
                Some(c) => {
                    Header::new(Codec::FreqLz4, vocabulary_id, n_tokens).write_to(&mut out);
                    out.extend_from_slice(&c);
                }
                None => {
                    Header::new(Codec::Freq, vocabulary_id, n_tokens).write_to(&mut out);
                    out.extend_from_slice(&payload);
                }
            }
        }
    }
    Ok(out)
}

/// Decode a stored value back to token IDs.
pub fn decode(value: &[u8], table: Option<&RankTable>) -> Result<Vec<u32>, ValueError> {
    let (h, payload) = Header::parse(value)?;
    Ok(match h.codec {
        Codec::Raw8 => raw::decode8(payload),
        Codec::Raw16 => raw::decode16(payload)?,
        Codec::Raw24 => raw::decode24(payload)?,
        Codec::Freq => {
            let t = table.ok_or(ValueError::MissingTable(h.codec))?;
            freq::decode_freq(payload, h.n_tokens as usize, t)?
        }
        Codec::FreqLz4 => {
            let t = table.ok_or(ValueError::MissingTable(h.codec))?;
            let n = h.n_tokens as usize;
            freq::decode_freq(&freq_lz4::decompress(payload, n)?, n, t)?
        }
    })
}

/// The one byte form of a value that equality and hashing use: a `freq_lz4` value rewritten as
/// the plain `freq` value it decompresses to, anything else as it is.
///
/// Every other codec has exactly one byte form per token list (`encoding_is_canonical`). lz4
/// does not: many blocks decompress to the same bytes, and a newer lz4 library may choose a
/// different one, so equal tokens could otherwise sit in unequal bytes. Needs no ranking table.
pub fn canonical(value: &[u8]) -> Result<Cow<'_, [u8]>, ValueError> {
    let (h, payload) = Header::parse(value)?;
    if h.codec != Codec::FreqLz4 {
        return Ok(Cow::Borrowed(value));
    }
    let freq_payload = freq_lz4::decompress(payload, h.n_tokens as usize)?;
    let mut out = Header::new(Codec::Freq, h.vocabulary_id, h.n_tokens).to_bytes();
    out.extend_from_slice(&freq_payload);
    Ok(Cow::Owned(out))
}

/// Whether two stored values hold the same tokens under the same vocabulary. Identical bytes
/// answer at once; a `freq_lz4` value is decompressed only when the bytes differ and the
/// headers agree on vocabulary, codec family and token count.
pub fn same(a: &[u8], b: &[u8]) -> Result<bool, ValueError> {
    if a == b {
        return Ok(true);
    }
    let (ha, _) = Header::parse(a)?;
    let (hb, _) = Header::parse(b)?;
    if ha.codec != Codec::FreqLz4 && hb.codec != Codec::FreqLz4 {
        return Ok(false);
    }
    if ha.vocabulary_id != hb.vocabulary_id
        || ha.n_tokens != hb.n_tokens
        || ha.codec.family() != hb.codec.family()
    {
        return Ok(false);
    }
    Ok(canonical(a)? == canonical(b)?)
}

/// Re-encode a value under a different codec, without leaving the token-ID domain.
pub fn recode(
    value: &[u8],
    to_codec: Codec,
    to_vocabulary_id: u16,
    from_table: Option<&RankTable>,
    to_table: Option<&RankTable>,
) -> Result<Vec<u8>, ValueError> {
    let ids = decode(value, from_table)?;
    encode(&ids, to_codec, to_vocabulary_id, to_table)
}

/// Header-only inspection: the header and the payload length. O(1), loads no table; the
/// token count comes from the header.
pub fn describe(value: &[u8]) -> Result<(Header, usize), ValueError> {
    let (h, payload) = Header::parse(value)?;
    Ok((h, payload.len()))
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

    const CASES: &[&[u32]] = &[
        &[],
        &[0],
        &[3, 1, 2],
        &[3, 1, 2, 3, 3, 1, 2, 2, 0, 4],
        &[65_535, 65_536, 200_018],
        &[1_000_000, 0, 7],
    ];

    fn codecs_for(ids: &[u32]) -> Vec<Codec> {
        let mut v = vec![Codec::Raw24, Codec::Freq];
        if ids.iter().all(|&i| i <= u16::MAX as u32) {
            v.push(Codec::Raw16);
        }
        if ids.iter().all(|&i| i <= u8::MAX as u32) {
            v.push(Codec::Raw8);
        }
        v
    }

    #[test]
    fn roundtrips_every_codec() {
        let t = table();
        for ids in CASES {
            for codec in codecs_for(ids) {
                let (tid, tbl) = if codec.needs_table() {
                    (1u16, Some(&t))
                } else {
                    (0u16, None)
                };
                let v = encode(ids, codec, tid, tbl)
                    .unwrap_or_else(|e| panic!("encode {} {ids:?}: {e}", codec.as_str()));
                let back = decode(&v, tbl)
                    .unwrap_or_else(|e| panic!("decode {} {ids:?}: {e}", codec.as_str()));
                assert_eq!(&back, ids, "{} lost data", codec.as_str());
            }
        }
    }

    #[test]
    fn encoding_is_canonical() {
        // The property that makes `=`, `GROUP BY`, `DISTINCT` and hash joins correct on a
        // pgtoken column: identical input must give byte-identical output.
        let t = table();
        for ids in CASES {
            for codec in codecs_for(ids) {
                let (tid, tbl) = if codec.needs_table() {
                    (1u16, Some(&t))
                } else {
                    (0u16, None)
                };
                let first = encode(ids, codec, tid, tbl).unwrap();
                for _ in 0..32 {
                    assert_eq!(encode(ids, codec, tid, tbl).unwrap(), first);
                }
            }
        }
    }

    #[test]
    fn recode_preserves_ids_across_every_pair() {
        let t = table();
        let ids: &[u32] = &[3, 1, 2, 0, 65_536, 200_018];
        for from in [Codec::Raw24, Codec::Freq] {
            for to in [Codec::Raw24, Codec::Freq] {
                let (fid, ftbl) = if from.needs_table() {
                    (1u16, Some(&t))
                } else {
                    (0, None)
                };
                let (tid, ttbl) = if to.needs_table() {
                    (1u16, Some(&t))
                } else {
                    (0, None)
                };
                let v = encode(ids, from, fid, ftbl).unwrap();
                let r = recode(&v, to, tid, ftbl, ttbl)
                    .unwrap_or_else(|e| panic!("recode {} -> {}: {e}", from.as_str(), to.as_str()));
                assert_eq!(decode(&r, ttbl).unwrap(), ids);
            }
        }
    }

    #[test]
    fn describe_counts_without_a_table() {
        let v = encode(&[1, 2, 3, 4, 5], Codec::Raw24, 0, None).unwrap();
        let (h, payload_len) = describe(&v).unwrap();
        assert_eq!(h.n_tokens, 5);
        assert_eq!(h.codec, Codec::Raw24);
        assert_eq!(payload_len, 15);

        let v = encode(&[3, 1, 200_018, 0], Codec::Freq, 1, Some(&table())).unwrap();
        let (h, _) = describe(&v).unwrap();
        assert_eq!((h.codec, h.n_tokens), (Codec::Freq, 4));
    }

    #[test]
    fn freq_refuses_to_run_without_a_table() {
        assert_eq!(
            encode(&[1, 2, 3], Codec::Freq, 1, None),
            Err(ValueError::MissingTable(Codec::Freq))
        );
    }

    #[test]
    fn empty_input_is_a_bare_header() {
        let v = encode(&[], Codec::Raw24, 0, None).unwrap();
        assert_eq!(
            v,
            [0x11, 0, 0x00, 0x00, 0x00],
            "header with a zero count only"
        );
        assert_eq!(decode(&v, None).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn raw8_roundtrips_through_a_value() {
        let v = encode(&[0, 7, 255], Codec::Raw8, 0, None).expect("encode");
        assert_eq!(v, [0x13, 0, 0x00, 0x00, 3, 0, 7, 255]);
        assert_eq!(decode(&v, None).unwrap(), vec![0, 7, 255]);
    }

    #[test]
    fn freq_beats_raw_on_a_realistic_distribution() {
        // Zipf-ish: a few tokens dominate, which is what the rank remap exploits.
        let t = table();
        let ids: Vec<u32> = (0..512)
            .map(|i| [3u32, 3, 3, 1, 1, 2][(i % 6) as usize])
            .collect();
        let raw = encode(&ids, Codec::Raw24, 0, None).unwrap();
        let freq = encode(&ids, Codec::Freq, 1, Some(&t)).unwrap();
        assert!(
            freq.len() < raw.len(),
            "freq {} vs raw {}",
            freq.len(),
            raw.len()
        );
        assert_eq!(decode(&freq, Some(&t)).unwrap(), ids);
    }

    #[test]
    fn golden_values_for_every_codec() {
        // ids [3, 1, 200] under vocabulary 5, raw; [3, 1, 200, 0] under vocabulary 65280, freq.
        // In the test table 3 is rank 0, 1 rank 1, 2 rank 2; unseen ids take k + id, k = 3.
        let hex = |v: &[u8]| v.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let raw = [3u32, 1, 200];
        //            tag flags id   n   payload
        let r8 = "13000500".to_string() + "03" + "0301c8";
        assert_eq!(hex(&encode(&raw, Codec::Raw8, 5, None).unwrap()), r8);
        assert_eq!(
            hex(&encode(&raw, Codec::Raw16, 5, None).unwrap()),
            "10000500".to_string() + "03" + "03000100c800"
        );
        assert_eq!(
            hex(&encode(&raw, Codec::Raw24, 5, None).unwrap()),
            "11000500".to_string() + "03" + "030000010000c80000"
        );
        let t = table();
        let v = encode(&[3, 1, 200, 0], Codec::Freq, 65280, Some(&t)).unwrap();
        // 200 -> rank 203 = 0xcb 0x01; 0 -> rank 3.
        assert_eq!(hex(&v), "120000ff".to_string() + "04" + "0001cb0103");
        assert_eq!(decode(&v, Some(&t)).unwrap(), [3, 1, 200, 0]);
    }

    #[test]
    fn rejects_malformed_values() {
        let t = table();
        let cases: &[(&[u8], ValueError)] = &[
            // The pre-release 12-byte header.
            (
                &[0xA7, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 7],
                ValueError::Header(HeaderError::LegacyFormat),
            ),
            (
                &[0x15, 0, 1, 0, 0],
                ValueError::Header(HeaderError::UnknownTag(0x15)),
            ),
            (
                &[0x14, 0, 1, 0, 0],
                ValueError::Header(HeaderError::FreqPayloadLen {
                    n_tokens: 0,
                    got: 0,
                }),
            ),
            (
                &[0x14, 0, 1, 0, 3, 0xFF],
                ValueError::Freq(freq::FreqError::Lz4(
                    "expected another byte, found none".into(),
                )),
            ),
            (
                &[0x21, 0, 1, 0, 0],
                ValueError::Header(HeaderError::UnknownTag(0x21)),
            ),
            (
                &[0xF1, 0, 1, 0, 0],
                ValueError::Header(HeaderError::ReservedEscape(0xF1)),
            ),
            (
                &[0x13, 0x80, 1, 0, 0],
                ValueError::Header(HeaderError::UnknownFlags(0x80)),
            ),
            (
                &[0x13, 0, 1, 0],
                ValueError::Header(HeaderError::TooShort(4)),
            ),
            (
                &[0x13, 0, 1, 0, 0x81, 0x00, 7],
                ValueError::Header(HeaderError::Count(VarintError::NonMinimal)),
            ),
            (
                &[0x12, 0, 1, 0, 2, 0x05, 0x81, 0x00],
                ValueError::Freq(freq::FreqError::NonMinimal { offset: 2 }),
            ),
            (
                &[0x12, 0, 1, 0, 2, 0x05, 0x81],
                ValueError::Freq(freq::FreqError::Truncated),
            ),
            (
                &[0x12, 0, 1, 0, 1, 0x80, 0x80, 0x80, 0x80, 0x10],
                ValueError::Freq(freq::FreqError::Overflow { offset: 4 }),
            ),
            (
                &[0x12, 0, 1, 0, 2, 0x05, 0x06, 0x07],
                ValueError::Freq(freq::FreqError::CountMismatch { want: 2, got: 3 }),
            ),
            (
                &[0x11, 0, 1, 0, 1, 0, 0, 0, 0],
                ValueError::Header(HeaderError::PayloadLen { want: 3, got: 4 }),
            ),
            (
                &[0x10, 0, 1, 0, 1, 0],
                ValueError::Header(HeaderError::PayloadLen { want: 2, got: 1 }),
            ),
        ];
        for (bytes, want) in cases {
            assert_eq!(decode(bytes, Some(&t)).as_ref(), Err(want), "{bytes:02x?}");
        }
    }

    /// A value with lines that repeat, which lz4 shrinks and plain `freq` cannot.
    fn repetitive() -> Vec<u32> {
        [3u32, 1, 2, 0, 200_018, 7].repeat(40)
    }

    #[test]
    fn token_count_is_capped_on_read_and_write() {
        // A 1 MB freq_lz4 value could otherwise claim 255 million tokens. Header only: the cap
        // must fire before any payload is looked at or any buffer is reserved.
        let mut v = Header::new(Codec::FreqLz4, 1, MAX_TOKENS + 1).to_bytes();
        v.extend(std::iter::repeat_n(0u8, 1 << 20));
        assert_eq!(
            Header::parse(&v).unwrap_err(),
            HeaderError::TooManyTokens(MAX_TOKENS + 1)
        );
        let mut v = Header::new(Codec::Raw8, 0, MAX_TOKENS + 1).to_bytes();
        v.extend(std::iter::repeat_n(0u8, MAX_TOKENS as usize + 1));
        assert!(matches!(
            Header::parse(&v),
            Err(HeaderError::TooManyTokens(_))
        ));
        // Exactly MAX_TOKENS still reads, and the encoder refuses one more.
        let ids = vec![0u32; MAX_TOKENS as usize];
        let at_cap = encode(&ids, Codec::Raw8, 0, None).unwrap();
        assert_eq!(decode(&at_cap, None).unwrap().len(), MAX_TOKENS as usize);
        let over = vec![0u32; MAX_TOKENS as usize + 1];
        assert!(matches!(
            encode(&over, Codec::Raw8, 0, None),
            Err(ValueError::TooManyTokens(_))
        ));
    }

    #[test]
    fn freq_writes_lz4_only_when_smaller() {
        let t = table();
        let short = encode(&[3, 1, 2], Codec::Freq, 1, Some(&t)).unwrap();
        assert_eq!(short[0], Codec::Freq.tag(), "3 tokens stay plain freq");
        let ids = repetitive();
        let v = encode(&ids, Codec::Freq, 1, Some(&t)).unwrap();
        assert_eq!(v[0], Codec::FreqLz4.tag());
        let plain = canonical(&v).unwrap();
        assert!(
            v.len() * 4 < plain.len(),
            "{} vs {} bytes",
            v.len(),
            plain.len()
        );
        assert_eq!(decode(&v, Some(&t)).unwrap(), ids);
        assert_eq!(describe(&v).unwrap().0.n_tokens, ids.len() as u32);
    }

    #[test]
    fn equality_ignores_how_lz4_chose_to_compress() {
        // The same tokens, three byte forms: what this build writes, plain freq, and an lz4
        // block that copies nothing (all literals), as a different lz4 version might write.
        let t = table();
        let ids = repetitive();
        let ours = encode(&ids, Codec::Freq, 1, Some(&t)).unwrap();
        let plain = canonical(&ours).unwrap().into_owned();
        let (h, freq_payload) = Header::parse(&plain).unwrap();
        let mut literal_only = Header::new(Codec::FreqLz4, 1, h.n_tokens).to_bytes();
        // One lz4 sequence of only literals: token 0xF0 (15+ literals), length extension
        // bytes, then the bytes themselves.
        let mut rest = freq_payload.len() - 15;
        literal_only.push(0xF0);
        while rest >= 255 {
            literal_only.push(255);
            rest -= 255;
        }
        literal_only.push(rest as u8);
        literal_only.extend_from_slice(freq_payload);
        assert_eq!(decode(&literal_only, Some(&t)).unwrap(), ids);

        for (a, b) in [
            (&ours, &plain),
            (&ours, &literal_only),
            (&plain, &literal_only),
        ] {
            assert_ne!(a, b);
            assert!(same(a, b).unwrap(), "same tokens must compare equal");
            assert_eq!(canonical(a).unwrap(), canonical(b).unwrap());
        }
        let other = encode(&repetitive()[1..], Codec::Freq, 1, Some(&t)).unwrap();
        assert!(!same(&ours, &other).unwrap());
        let other_vocab = encode(&ids, Codec::Freq, 2, Some(&t)).unwrap();
        assert!(!same(&ours, &other_vocab).unwrap());
    }

    #[test]
    fn golden_freq_lz4_value() {
        // Pins this build's lz4 output. If an lz4 upgrade changes it, old values still decode
        // and compare equal (see above); update the golden after checking that.
        let hex = |v: &[u8]| v.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let v = encode(&repetitive(), Codec::Freq, 65280, Some(&table())).unwrap();
        // tag 14 (freq_lz4), flags 00, id 65280, count 0xf0 0x01 (240), then the lz4 block.
        assert_eq!(
            hex(&v),
            "140000fff0018f00010203d59a0c0a0800ff20600203d59a0c0a"
        );
    }
}
