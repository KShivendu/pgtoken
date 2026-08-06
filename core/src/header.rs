//! The 12-byte self-describing value header.
//!
//! Every stored value names its own codec, coding table and token count. Nothing here refers
//! to a tokenizer: this library compresses sequences of token IDs and has no opinion about
//! which vocabulary produced them. Keeping that knowledge out means any tokenizer works,
//! including ones that did not exist when this was written.
//!
//! ```text
//! off  size  field
//!   0     1  magic 0xA7
//!   1     1  format version (1)
//!   2     1  codec id  0=raw16 1=raw24 2=freq
//!   3     1  reserved, must be zero
//!   4     2  vocabulary id (u16 LE; 0 = none)
//!   6     2  reserved, must be zero
//!   8     4  token count (u32 LE)
//!  12     -  payload
//! ```
//!
//! The token count is not redundant: streamvbyte needs `n` up front to decode.

use core::fmt;

pub const MAGIC: u8 = 0xA7;
pub const VERSION: u8 = 1;
pub const HEADER_LEN: usize = 12;

/// How the token-ID sequence is encoded in the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Codec {
    /// 2 bytes/id, little-endian. IDs must fit in `u16`.
    Raw16 = 0,
    /// 3 bytes/id, big-endian. IDs must fit in 24 bits, which covers every vocabulary in
    /// practical use.
    Raw24 = 1,
    /// Frequency-rank remap, then streamvbyte.
    Freq = 2,
}

impl Codec {
    pub fn from_u8(v: u8) -> Result<Self, HeaderError> {
        match v {
            0 => Ok(Codec::Raw16),
            1 => Ok(Codec::Raw24),
            2 => Ok(Codec::Freq),
            _ => Err(HeaderError::UnknownCodec(v)),
        }
    }

    /// Whether this codec needs a trained ranking, and therefore a nonzero `vocabulary_id`.
    pub fn needs_table(self) -> bool {
        matches!(self, Codec::Freq)
    }

    /// Largest token ID this codec can represent, for the table-free codecs. The
    /// table-driven ones are bounded by their table's vocabulary instead.
    pub fn max_id(self) -> Option<u32> {
        match self {
            Codec::Raw16 => Some(u16::MAX as u32),
            Codec::Raw24 => Some(0x00FF_FFFF),
            Codec::Freq => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Codec::Raw16 => "raw16",
            Codec::Raw24 => "raw24",
            Codec::Freq => "freq",
        }
    }

    /// Parse a codec name. `raw` is resolved by the caller, which knows the ID range.
    pub fn parse(s: &str) -> Result<Self, HeaderError> {
        match s {
            "raw16" => Ok(Codec::Raw16),
            "raw24" => Ok(Codec::Raw24),
            "freq" => Ok(Codec::Freq),
            _ => Err(HeaderError::UnknownCodecName(s.to_owned())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub codec: Codec,
    pub vocabulary_id: u16,
    pub n_tokens: u32,
}

impl Header {
    pub fn new(codec: Codec, vocabulary_id: u16, n_tokens: u32) -> Self {
        Header {
            codec,
            vocabulary_id,
            n_tokens,
        }
    }

    /// Write the header into the first `HEADER_LEN` bytes of a fresh buffer.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.push(MAGIC);
        out.push(VERSION);
        out.push(self.codec as u8);
        out.push(0); // reserved
        out.extend_from_slice(&self.vocabulary_id.to_le_bytes());
        out.extend_from_slice(&[0u8, 0u8]); // reserved
        out.extend_from_slice(&self.n_tokens.to_le_bytes());
        debug_assert_eq!(out.len(), HEADER_LEN);
    }

    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut v = Vec::with_capacity(HEADER_LEN);
        self.write_to(&mut v);
        let mut out = [0u8; HEADER_LEN];
        out.copy_from_slice(&v);
        out
    }

    /// Parse and validate a header, returning it alongside the payload slice.
    ///
    /// Every malformed input is an error rather than a best-effort decode: a value reaching
    /// this function came off disk or off the wire, and silently reinterpreting it would hand
    /// back plausible-looking wrong IDs.
    pub fn parse(buf: &[u8]) -> Result<(Header, &[u8]), HeaderError> {
        if buf.len() < HEADER_LEN {
            return Err(HeaderError::TooShort(buf.len()));
        }
        if buf[0] != MAGIC {
            return Err(HeaderError::BadMagic(buf[0]));
        }
        if buf[1] != VERSION {
            return Err(HeaderError::UnsupportedVersion(buf[1]));
        }
        let codec = Codec::from_u8(buf[2])?;
        // Reserved bytes are checked, not ignored, so a future version that assigns meaning
        // to them can rely on old writers having left them zero.
        if buf[3] != 0 || buf[6] != 0 || buf[7] != 0 {
            return Err(HeaderError::ReservedNotZero);
        }
        let vocabulary_id = u16::from_le_bytes([buf[4], buf[5]]);
        let n_tokens = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);

        // `freq` cannot decode without its ranking, so a zero id is a broken value. The converse
        // is deliberately allowed: a `raw` value names the vocabulary it belongs to even though
        // the codec itself needs nothing from it.
        if codec.needs_table() && vocabulary_id == 0 {
            return Err(HeaderError::MissingTable(codec));
        }

        let payload = &buf[HEADER_LEN..];

        // The raw codecs have an exact expected length, so a truncated or padded payload is
        // caught here rather than producing a short read downstream.
        let expected = match codec {
            Codec::Raw16 => Some(n_tokens as usize * 2),
            Codec::Raw24 => Some(n_tokens as usize * 3),
            Codec::Freq => None,
        };
        if let Some(want) = expected {
            if payload.len() != want {
                return Err(HeaderError::PayloadLen {
                    want,
                    got: payload.len(),
                });
            }
        } else if n_tokens > 0 && payload.is_empty() {
            return Err(HeaderError::PayloadLen { want: 1, got: 0 });
        }

        Ok((
            Header {
                codec,
                vocabulary_id,
                n_tokens,
            },
            payload,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderError {
    TooShort(usize),
    BadMagic(u8),
    UnsupportedVersion(u8),
    UnknownCodec(u8),
    UnknownCodecName(String),
    ReservedNotZero,
    MissingTable(Codec),
    PayloadLen { want: usize, got: usize },
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderError::TooShort(n) => {
                write!(
                    f,
                    "value is {n} bytes, shorter than the {HEADER_LEN}-byte header"
                )
            }
            HeaderError::BadMagic(b) => {
                write!(f, "bad magic byte 0x{b:02X}, expected 0x{MAGIC:02X}")
            }
            HeaderError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported format version {v}, this build understands {VERSION}"
                )
            }
            HeaderError::UnknownCodec(v) => write!(f, "unknown codec id {v}"),
            HeaderError::UnknownCodecName(s) => write!(f, "unknown codec name {s:?}"),
            HeaderError::ReservedNotZero => write!(f, "reserved header bytes are not zero"),
            HeaderError::MissingTable(c) => {
                write!(
                    f,
                    "codec {} requires a table but vocabulary_id is 0",
                    c.as_str()
                )
            }
            HeaderError::PayloadLen { want, got } => {
                write!(f, "payload is {got} bytes, expected {want}")
            }
        }
    }
}

impl std::error::Error for HeaderError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_raw24(n: u32) -> Vec<u8> {
        let mut v = Vec::new();
        Header::new(Codec::Raw24, 0, n).write_to(&mut v);
        v.extend(std::iter::repeat_n(0u8, n as usize * 3));
        v
    }

    #[test]
    fn header_roundtrips() {
        for (codec, table) in [(Codec::Raw16, 0u16), (Codec::Raw24, 0), (Codec::Freq, 7)] {
            let h = Header::new(codec, table, 512);
            let mut buf = Vec::new();
            h.write_to(&mut buf);
            match codec {
                Codec::Raw16 => buf.extend(std::iter::repeat_n(0u8, 1024)),
                Codec::Raw24 => buf.extend(std::iter::repeat_n(0u8, 1536)),
                _ => buf.extend_from_slice(&[1, 2, 3]),
            }
            let (parsed, payload) = Header::parse(&buf).expect("should parse");
            assert_eq!(parsed, h);
            assert!(!payload.is_empty());
        }
    }

    #[test]
    fn header_is_exactly_12_bytes() {
        assert_eq!(
            Header::new(Codec::Freq, 1, 512).to_bytes().len(),
            HEADER_LEN
        );
        assert_eq!(HEADER_LEN, 12);
    }

    #[test]
    fn header_carries_no_tokenizer() {
        // The point of the format: values are interchangeable regardless of which tokenizer
        // produced their IDs, so nothing in the header may encode one.
        let bytes = Header::new(Codec::Raw24, 0, 3).to_bytes();
        assert_eq!(
            bytes[3], 0,
            "byte 3 must stay reserved, not become a tokenizer tag"
        );
    }

    #[test]
    fn rejects_short_buffer() {
        assert_eq!(Header::parse(&[]), Err(HeaderError::TooShort(0)));
        assert_eq!(Header::parse(&[MAGIC; 11]), Err(HeaderError::TooShort(11)));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut v = valid_raw24(2);
        v[0] = 0x00;
        assert_eq!(Header::parse(&v), Err(HeaderError::BadMagic(0x00)));
    }

    #[test]
    fn rejects_unknown_version() {
        let mut v = valid_raw24(2);
        v[1] = 99;
        assert_eq!(Header::parse(&v), Err(HeaderError::UnsupportedVersion(99)));
    }

    #[test]
    fn rejects_unknown_codec() {
        let mut v = valid_raw24(2);
        v[2] = 9;
        assert_eq!(Header::parse(&v), Err(HeaderError::UnknownCodec(9)));
    }

    #[test]
    fn rejects_nonzero_reserved() {
        for idx in [3usize, 6, 7] {
            let mut v = valid_raw24(2);
            v[idx] = 1;
            assert_eq!(
                Header::parse(&v),
                Err(HeaderError::ReservedNotZero),
                "reserved byte {idx} was not checked"
            );
        }
    }

    #[test]
    fn rejects_truncated_and_padded_raw_payload() {
        let mut v = valid_raw24(4);
        v.pop();
        assert_eq!(
            Header::parse(&v),
            Err(HeaderError::PayloadLen { want: 12, got: 11 })
        );

        let mut v = valid_raw24(4);
        v.push(0);
        assert_eq!(
            Header::parse(&v),
            Err(HeaderError::PayloadLen { want: 12, got: 13 })
        );
    }

    #[test]
    fn freq_still_requires_a_vocabulary() {
        let mut v = Vec::new();
        Header::new(Codec::Freq, 0, 1).write_to(&mut v);
        v.push(1);
        assert_eq!(
            Header::parse(&v),
            Err(HeaderError::MissingTable(Codec::Freq))
        );
    }

    #[test]
    fn raw_codecs_may_carry_a_vocabulary_id() {
        // Every column belongs to a vocabulary now, including `raw` ones, and the value must
        // record which — for the cast's type check, and for detokenization later.
        let mut v = Vec::new();
        Header::new(Codec::Raw24, 3, 1).write_to(&mut v);
        v.extend_from_slice(&[0, 0, 0]);
        let (h, _) = Header::parse(&v).expect("raw with a vocabulary id must parse");
        assert_eq!(h.vocabulary_id, 3);
    }

    #[test]
    fn empty_token_sequence_is_valid() {
        let v = valid_raw24(0);
        let (h, payload) = Header::parse(&v).expect("empty should be valid");
        assert_eq!(h.n_tokens, 0);
        assert!(payload.is_empty());
    }

    #[test]
    fn codec_id_ranges() {
        assert_eq!(Codec::Raw16.max_id(), Some(65_535));
        assert_eq!(Codec::Raw24.max_id(), Some(16_777_215));
        assert_eq!(Codec::Freq.max_id(), None);
        assert_eq!(
            Codec::from_u8(3),
            Err(HeaderError::UnknownCodec(3)),
            "ANS was removed"
        );
    }
}
