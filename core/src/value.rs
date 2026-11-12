//! The stored value: header plus encoded payload.
//!
//! Everything here operates on token IDs. Turning text into IDs is the caller's job, done
//! with whatever tokenizer they already use — that stays outside this library on purpose.

use crate::codec::{freq, raw};
#[cfg(test)]
use crate::header::VarintError;
use crate::header::{Codec, Header, HeaderError, MAX_HEADER_LEN};
use crate::tables::RankTable;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    Header(HeaderError),
    Raw(raw::RawError),
    Freq(freq::FreqError),
    /// The codec needs a coding table that was not supplied.
    MissingTable(Codec),
    /// More IDs than the 32-bit token count can express.
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
                    "{n} tokens exceeds the {} the format can address",
                    u32::MAX
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
pub fn encode(
    ids: &[u32],
    codec: Codec,
    vocabulary_id: u16,
    table: Option<&RankTable>,
) -> Result<Vec<u8>, ValueError> {
    let n_tokens = u32::try_from(ids.len()).map_err(|_| ValueError::TooManyTokens(ids.len()))?;
    let mut out = Vec::with_capacity(MAX_HEADER_LEN + ids.len() * 2);
    Header::new(codec, vocabulary_id, n_tokens).write_to(&mut out);

    match codec {
        Codec::Raw8 => raw::encode8(ids, &mut out)?,
        Codec::Raw16 => raw::encode16(ids, &mut out)?,
        Codec::Raw24 => raw::encode24(ids, &mut out)?,
        Codec::Freq => {
            let t = table.ok_or(ValueError::MissingTable(codec))?;
            freq::encode_freq(ids, t, &mut out)?
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
    })
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
            "11000500".to_string() + "03" + "0000030000010000c8"
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
                &[0x14, 0, 1, 0, 0],
                ValueError::Header(HeaderError::UnknownTag(0x14)),
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
}
