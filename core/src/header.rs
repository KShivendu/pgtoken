//! The 12-byte self-describing value header.
//!
//! Every stored value names its own tokenizer, codec, and coding table, so a column can
//! hold values from different tokenizers during a migration and no per-column typmod has
//! to be kept in sync. Naming the table in the value is also what lets `decode` be
//! honestly `IMMUTABLE` in Postgres: the bytes fully determine the text.
//!
//! ```text
//! off  size  field
//!   0     1  magic 0xA7
//!   1     1  format version (1)
//!   2     1  tokenizer id   1=r50k 2=cl100k 3=o200k
//!   3     1  codec id       0=raw16 1=raw24 2=freq+svb 3=ANS
//!   4     2  table id (u16 LE; 0 = none, for the raw codecs)
//!   6     2  reserved, must be zero
//!   8     4  token count (u32 LE)
//!  12     -  payload
//! ```
//!
//! The token count is not redundant: both streamvbyte and ANS need `n` up front to
//! decode. This mirrors the Python harness, which prepends `len(ids).to_bytes(4, 'big')`.

use core::fmt;

pub const MAGIC: u8 = 0xA7;
pub const VERSION: u8 = 1;
pub const HEADER_LEN: usize = 12;

/// Which BPE vocabulary the token IDs belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Tokenizer {
    R50k = 1,
    Cl100k = 2,
    O200k = 3,
}

impl Tokenizer {
    pub fn from_u8(v: u8) -> Result<Self, HeaderError> {
        match v {
            1 => Ok(Tokenizer::R50k),
            2 => Ok(Tokenizer::Cl100k),
            3 => Ok(Tokenizer::O200k),
            _ => Err(HeaderError::UnknownTokenizer(v)),
        }
    }

    /// Vocabulary size. These fix the width of the rank and ANS tables, so they must match
    /// the values the Python harness trains against.
    pub fn vocab(self) -> u32 {
        match self {
            Tokenizer::R50k => 50_257,
            Tokenizer::Cl100k => 100_277,
            Tokenizer::O200k => 200_019,
        }
    }

    /// True when every ID fits in a `u16`, i.e. the `raw16` codec is available.
    pub fn fits_u16(self) -> bool {
        self.vocab() <= u16::MAX as u32 + 1
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Tokenizer::R50k => "r50k",
            Tokenizer::Cl100k => "cl100k",
            Tokenizer::O200k => "o200k",
        }
    }

    pub fn parse(s: &str) -> Result<Self, HeaderError> {
        match s {
            "r50k" | "r50k_base" => Ok(Tokenizer::R50k),
            "cl100k" | "cl100k_base" => Ok(Tokenizer::Cl100k),
            "o200k" | "o200k_base" => Ok(Tokenizer::O200k),
            _ => Err(HeaderError::UnknownTokenizerName(s.to_owned())),
        }
    }
}

/// How the token-ID sequence is encoded in the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Codec {
    /// 2 bytes/id, little-endian. Only valid for vocabularies that fit in `u16`.
    Raw16 = 0,
    /// 3 bytes/id, big-endian. Matches `tnbench.pack3`.
    Raw24 = 1,
    /// Frequency-rank remap, then streamvbyte.
    Freq = 2,
    /// Static Laplace-smoothed unigram ANS.
    Ans = 3,
}

impl Codec {
    pub fn from_u8(v: u8) -> Result<Self, HeaderError> {
        match v {
            0 => Ok(Codec::Raw16),
            1 => Ok(Codec::Raw24),
            2 => Ok(Codec::Freq),
            3 => Ok(Codec::Ans),
            _ => Err(HeaderError::UnknownCodec(v)),
        }
    }

    /// Whether this codec needs a trained table, and therefore a nonzero `table_id`.
    pub fn needs_table(self) -> bool {
        matches!(self, Codec::Freq | Codec::Ans)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Codec::Raw16 => "raw16",
            Codec::Raw24 => "raw24",
            Codec::Freq => "freq",
            Codec::Ans => "ans",
        }
    }

    pub fn parse(s: &str) -> Result<Self, HeaderError> {
        match s {
            "raw16" => Ok(Codec::Raw16),
            "raw24" => Ok(Codec::Raw24),
            // `raw` picks the narrowest packing the vocabulary allows; resolved by the
            // caller, which knows the tokenizer. Deliberately not accepted here.
            "freq" => Ok(Codec::Freq),
            "ans" => Ok(Codec::Ans),
            _ => Err(HeaderError::UnknownCodecName(s.to_owned())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub tokenizer: Tokenizer,
    pub codec: Codec,
    pub table_id: u16,
    pub n_tokens: u32,
}

impl Header {
    pub fn new(tokenizer: Tokenizer, codec: Codec, table_id: u16, n_tokens: u32) -> Self {
        Header { tokenizer, codec, table_id, n_tokens }
    }

    /// Write the header into the first `HEADER_LEN` bytes of a fresh buffer.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.push(MAGIC);
        out.push(VERSION);
        out.push(self.tokenizer as u8);
        out.push(self.codec as u8);
        out.extend_from_slice(&self.table_id.to_le_bytes());
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
    /// Every malformed input is an error rather than a best-effort decode: a value that
    /// reaches this function came off disk or off the wire, and silently reinterpreting
    /// it would hand back plausible-looking wrong text.
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
        let tokenizer = Tokenizer::from_u8(buf[2])?;
        let codec = Codec::from_u8(buf[3])?;
        let table_id = u16::from_le_bytes([buf[4], buf[5]]);
        // Reserved bytes are checked, not ignored. A future version that assigns meaning
        // to them can then rely on old writers having left them zero.
        if buf[6] != 0 || buf[7] != 0 {
            return Err(HeaderError::ReservedNotZero);
        }
        let n_tokens = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);

        if codec == Codec::Raw16 && !tokenizer.fits_u16() {
            return Err(HeaderError::Raw16Overflow(tokenizer));
        }
        if codec.needs_table() && table_id == 0 {
            return Err(HeaderError::MissingTable(codec));
        }
        if !codec.needs_table() && table_id != 0 {
            return Err(HeaderError::UnexpectedTable(codec, table_id));
        }

        let payload = &buf[HEADER_LEN..];

        // The raw codecs have an exact expected length, so a truncated or padded payload
        // is caught here rather than producing a short read downstream.
        let expected = match codec {
            Codec::Raw16 => Some(n_tokens as usize * 2),
            Codec::Raw24 => Some(n_tokens as usize * 3),
            Codec::Freq | Codec::Ans => None,
        };
        if let Some(want) = expected {
            if payload.len() != want {
                return Err(HeaderError::PayloadLen { want, got: payload.len() });
            }
        } else if n_tokens > 0 && payload.is_empty() {
            return Err(HeaderError::PayloadLen { want: 1, got: 0 });
        }

        Ok((Header { tokenizer, codec, table_id, n_tokens }, payload))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderError {
    TooShort(usize),
    BadMagic(u8),
    UnsupportedVersion(u8),
    UnknownTokenizer(u8),
    UnknownTokenizerName(String),
    UnknownCodec(u8),
    UnknownCodecName(String),
    ReservedNotZero,
    Raw16Overflow(Tokenizer),
    MissingTable(Codec),
    UnexpectedTable(Codec, u16),
    PayloadLen { want: usize, got: usize },
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderError::TooShort(n) => {
                write!(f, "value is {n} bytes, shorter than the {HEADER_LEN}-byte header")
            }
            HeaderError::BadMagic(b) => {
                write!(f, "bad magic byte 0x{b:02X}, expected 0x{MAGIC:02X}")
            }
            HeaderError::UnsupportedVersion(v) => {
                write!(f, "unsupported format version {v}, this build understands {VERSION}")
            }
            HeaderError::UnknownTokenizer(v) => write!(f, "unknown tokenizer id {v}"),
            HeaderError::UnknownTokenizerName(s) => write!(f, "unknown tokenizer name {s:?}"),
            HeaderError::UnknownCodec(v) => write!(f, "unknown codec id {v}"),
            HeaderError::UnknownCodecName(s) => write!(f, "unknown codec name {s:?}"),
            HeaderError::ReservedNotZero => write!(f, "reserved header bytes are not zero"),
            HeaderError::Raw16Overflow(t) => {
                write!(f, "codec raw16 cannot represent {} (vocab {})", t.as_str(), t.vocab())
            }
            HeaderError::MissingTable(c) => {
                write!(f, "codec {} requires a table but table_id is 0", c.as_str())
            }
            HeaderError::UnexpectedTable(c, id) => {
                write!(f, "codec {} takes no table but table_id is {id}", c.as_str())
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
        Header::new(Tokenizer::O200k, Codec::Raw24, 0, n).write_to(&mut v);
        v.extend(std::iter::repeat_n(0u8, n as usize * 3));
        v
    }

    #[test]
    fn header_roundtrips() {
        for (tok, codec, table) in [
            (Tokenizer::R50k, Codec::Raw16, 0u16),
            (Tokenizer::O200k, Codec::Raw24, 0),
            (Tokenizer::Cl100k, Codec::Freq, 7),
            (Tokenizer::O200k, Codec::Ans, 65535),
        ] {
            let h = Header::new(tok, codec, table, 512);
            let mut buf = Vec::new();
            h.write_to(&mut buf);
            // Give the table-driven codecs a nonempty payload; raw ones need exact length.
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
        let h = Header::new(Tokenizer::O200k, Codec::Freq, 1, 512);
        assert_eq!(h.to_bytes().len(), HEADER_LEN);
        assert_eq!(HEADER_LEN, 12);
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
    fn rejects_unknown_tokenizer_and_codec() {
        let mut v = valid_raw24(2);
        v[2] = 9;
        assert_eq!(Header::parse(&v), Err(HeaderError::UnknownTokenizer(9)));

        let mut v = valid_raw24(2);
        v[3] = 9;
        assert_eq!(Header::parse(&v), Err(HeaderError::UnknownCodec(9)));
    }

    #[test]
    fn rejects_nonzero_reserved() {
        for idx in [6usize, 7] {
            let mut v = valid_raw24(2);
            v[idx] = 1;
            assert_eq!(Header::parse(&v), Err(HeaderError::ReservedNotZero));
        }
    }

    #[test]
    fn rejects_truncated_and_padded_raw_payload() {
        let mut v = valid_raw24(4);
        v.pop();
        assert_eq!(Header::parse(&v), Err(HeaderError::PayloadLen { want: 12, got: 11 }));

        let mut v = valid_raw24(4);
        v.push(0);
        assert_eq!(Header::parse(&v), Err(HeaderError::PayloadLen { want: 12, got: 13 }));
    }

    #[test]
    fn rejects_raw16_for_wide_vocab() {
        let mut v = Vec::new();
        Header::new(Tokenizer::O200k, Codec::Raw16, 0, 1).write_to(&mut v);
        v.extend_from_slice(&[0, 0]);
        assert_eq!(Header::parse(&v), Err(HeaderError::Raw16Overflow(Tokenizer::O200k)));
    }

    #[test]
    fn table_id_presence_must_match_codec() {
        // freq/ans without a table
        let mut v = Vec::new();
        Header::new(Tokenizer::O200k, Codec::Freq, 0, 1).write_to(&mut v);
        v.extend_from_slice(&[1]);
        assert_eq!(Header::parse(&v), Err(HeaderError::MissingTable(Codec::Freq)));

        // raw with a table
        let mut v = Vec::new();
        Header::new(Tokenizer::O200k, Codec::Raw24, 3, 1).write_to(&mut v);
        v.extend_from_slice(&[0, 0, 0]);
        assert_eq!(Header::parse(&v), Err(HeaderError::UnexpectedTable(Codec::Raw24, 3)));
    }

    #[test]
    fn empty_token_sequence_is_valid() {
        let v = valid_raw24(0);
        let (h, payload) = Header::parse(&v).expect("empty should be valid");
        assert_eq!(h.n_tokens, 0);
        assert!(payload.is_empty());
    }

    #[test]
    fn only_r50k_fits_u16() {
        assert!(Tokenizer::R50k.fits_u16());
        assert!(!Tokenizer::Cl100k.fits_u16());
        assert!(!Tokenizer::O200k.fits_u16());
    }
}
