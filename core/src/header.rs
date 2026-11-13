//! The self-describing value header: one tag byte and a varint vocabulary id.
//!
//! Every stored value names its own format version, codec and vocabulary. Nothing here refers
//! to a tokenizer: this library compresses sequences of token IDs and has no opinion about
//! which vocabulary produced them. Keeping that knowledge out means any tokenizer works,
//! including ones that did not exist when this was written.
//!
//! ```text
//! off  size  field
//!   0     1  tag = (version << 4) | codec    version 1; codec 0=raw16 1=raw24 2=freq 3=raw8
//!                                                        4=freq_lz4
//!   1     1  flags, all zero in version 1
//!   2     2  vocabulary id (u16 LE; 0 = none)
//!   4   1-5  token count, unsigned LEB128, minimally encoded, at most u32::MAX
//!   *     -  payload, to the end of the value
//! ```
//!
//! So `12 00 00 ff 03 …` is `freq` under vocabulary 65280 holding 3 tokens, and an empty `raw24`
//! value with no vocabulary is the five bytes `11 00 00 00 00`.
//!
//! **Tag.** Only `0x10..=0x14` are valid. The high nibble `0xF` is reserved as an escape for a
//! future, different header. Values from the pre-release 12-byte layout start with `0xA7` and
//! are refused with an error that says so.
//!
//! **Flags.** A reader refuses any bit it does not know, so a future flag can never be silently
//! ignored by an old reader. `=`, hashing and `DISTINCT` compare a value's canonical bytes (see
//! [`crate::value::canonical`]), so a flag added later must either change what the value means
//! (so unequal bytes are right) or be masked out of equality and hashing.
//!
//! **Count.** Stored for every codec, so `token_count` is O(1) and a large TOASTed value can be
//! sized from a short detoast slice without reading its payload. For the raw codecs it is
//! cross-checked here against the payload length (`count * width`). For `freq` the decoder
//! checks that the payload holds exactly that many varints (see [`crate::codec::freq`]). For
//! `freq_lz4` the count bounds how far the payload may decompress.
//!
//! Every field is minimally encoded, because two encodings of one value would compare unequal.
//! Nothing needs alignment: every field is read with byte loads. (The SQL type still declares
//! `ALIGNMENT = int4`, the narrowest PostgreSQL allows for a variable-length type.)

use core::fmt;

/// Format version, the tag's high nibble.
pub const VERSION: u8 = 1;
/// High nibble reserved for a future header escape.
pub const ESCAPE: u8 = 0xF;
/// First byte of every value in the pre-release 12-byte layout.
pub const LEGACY_MAGIC: u8 = 0xA7;
/// Bytes before the count: tag, flags, vocabulary id.
pub const FIXED_LEN: usize = 4;
/// Shortest valid value: the fixed part and a one-byte count of zero.
pub const MIN_LEN: usize = FIXED_LEN + 1;
/// Longest header: the fixed part and a five-byte count.
pub const MAX_HEADER_LEN: usize = FIXED_LEN + 5;
/// Flag bits this build understands. None yet.
pub const KNOWN_FLAGS: u8 = 0;
/// The most tokens one value may hold: 2^26 = 67,108,864. Far past any model's context window,
/// and its ids as `int[]` (256 MB) stay well under PostgreSQL's 1 GB allocation limit. The real
/// point is `freq_lz4`: lz4 expands up to 255-fold, so without a cap a 1 MB value could claim
/// 255 million tokens and make every reader allocate over a gigabyte. With it, the worst 1 MB
/// value decodes to about 330 MB. Raising it later keeps every stored value readable.
pub const MAX_TOKENS: u32 = 1 << 26;

/// How the token-ID sequence is encoded in the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Codec {
    /// 2 bytes/id, little-endian. IDs must fit in `u16`.
    Raw16 = 0,
    /// 3 bytes/id, little-endian. IDs must fit in 24 bits, which covers every vocabulary in
    /// practical use.
    Raw24 = 1,
    /// Frequency-rank remap, then one LEB128 varint per rank.
    Freq = 2,
    /// 1 byte/id. IDs must fit in `u8`.
    Raw8 = 3,
    /// The `freq` payload compressed with an lz4 block. Never asked for directly: encoding as
    /// `freq` writes this instead when it is smaller (see [`crate::codec::freq_lz4`]).
    FreqLz4 = 4,
}

impl Codec {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Codec::Raw16),
            1 => Some(Codec::Raw24),
            2 => Some(Codec::Freq),
            3 => Some(Codec::Raw8),
            4 => Some(Codec::FreqLz4),
            _ => None,
        }
    }

    /// Whether this codec needs a trained ranking, and therefore a nonzero `vocabulary_id`.
    pub fn needs_table(self) -> bool {
        matches!(self, Codec::Freq | Codec::FreqLz4)
    }

    /// The codec a column chooses that writes this one: `freq` for `freq_lz4`, itself otherwise.
    /// `freq` and `freq_lz4` values can sit in the same column and compare equal.
    pub fn family(self) -> Codec {
        match self {
            Codec::FreqLz4 => Codec::Freq,
            c => c,
        }
    }

    /// Bytes per token for the fixed-width codecs, `None` for `freq`.
    pub fn width(self) -> Option<usize> {
        match self {
            Codec::Raw8 => Some(1),
            Codec::Raw16 => Some(2),
            Codec::Raw24 => Some(3),
            Codec::Freq | Codec::FreqLz4 => None,
        }
    }

    /// Largest token ID this codec can represent, for the table-free codecs. The
    /// table-driven ones are bounded by their table's vocabulary instead.
    pub fn max_id(self) -> Option<u32> {
        match self {
            Codec::Raw16 => Some(u16::MAX as u32),
            Codec::Raw24 => Some(0x00FF_FFFF),
            Codec::Freq | Codec::FreqLz4 => None,
            Codec::Raw8 => Some(u8::MAX as u32),
        }
    }

    /// The tag byte a value under this codec starts with.
    pub fn tag(self) -> u8 {
        (VERSION << 4) | self as u8
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Codec::Raw16 => "raw16",
            Codec::Raw24 => "raw24",
            Codec::Freq => "freq",
            Codec::Raw8 => "raw8",
            Codec::FreqLz4 => "freq_lz4",
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

    /// Append the header to `out`.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.push(self.codec.tag());
        out.push(0); // flags
        out.extend_from_slice(&self.vocabulary_id.to_le_bytes());
        write_varint(self.n_tokens, out);
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(MAX_HEADER_LEN);
        self.write_to(&mut v);
        v
    }

    /// Parse and validate a header, returning it alongside the payload slice. O(1).
    ///
    /// Checks the tag, the flags, the count's encoding, that `freq` names a vocabulary, and, for
    /// the raw codecs, that the payload is exactly `count * width` bytes. A `freq` payload is
    /// checked against the count when it is decoded, since that takes a scan.
    ///
    /// Every malformed input is an error rather than a best-effort decode: a value reaching
    /// this function came off disk or off the wire, and silently reinterpreting it would hand
    /// back plausible-looking wrong IDs.
    pub fn parse(buf: &[u8]) -> Result<(Header, &[u8]), HeaderError> {
        let Some(&tag) = buf.first() else {
            return Err(HeaderError::TooShort(0));
        };
        // The tag first, so an old 12-byte value is named as such however long it is.
        let codec = parse_tag(tag)?;
        if buf.len() < MIN_LEN {
            return Err(HeaderError::TooShort(buf.len()));
        }
        let flags = buf[1];
        if flags & !KNOWN_FLAGS != 0 {
            return Err(HeaderError::UnknownFlags(flags));
        }
        let vocabulary_id = u16::from_le_bytes([buf[2], buf[3]]);
        let (n_tokens, used) = read_varint(&buf[FIXED_LEN..]).map_err(HeaderError::Count)?;
        if n_tokens > MAX_TOKENS {
            return Err(HeaderError::TooManyTokens(n_tokens));
        }

        // `freq` cannot decode without its ranking, so a zero id is a broken value. The converse
        // is deliberately allowed: a `raw` value names the vocabulary it belongs to even though
        // the codec itself needs nothing from it.
        if codec.needs_table() && vocabulary_id == 0 {
            return Err(HeaderError::MissingTable(codec));
        }

        let payload = &buf[FIXED_LEN + used..];
        match codec.width() {
            Some(width) => {
                let want = n_tokens as usize * width;
                if payload.len() != want {
                    return Err(HeaderError::PayloadLen {
                        want,
                        got: payload.len(),
                    });
                }
            }
            // An lz4 block expands at most 255-fold, and each rank is at least one byte.
            None if codec == Codec::FreqLz4 => {
                if n_tokens == 0 || n_tokens as usize > payload.len().saturating_mul(255) {
                    return Err(HeaderError::FreqPayloadLen {
                        n_tokens,
                        got: payload.len(),
                    });
                }
            }
            // Every rank takes at least one byte and at most five.
            None => {
                let n = n_tokens as usize;
                if payload.len() < n || payload.len() > n.saturating_mul(5) {
                    return Err(HeaderError::FreqPayloadLen {
                        n_tokens,
                        got: payload.len(),
                    });
                }
            }
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

/// Append `v` as unsigned LEB128: 7 bits per byte, low bits first, high bit set on all but the
/// last byte.
#[inline]
pub fn write_varint(mut v: u32, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Why a varint was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarintError {
    /// The input ends on a continuation byte.
    Truncated,
    /// A multi-byte varint ends in `0x00`, so a shorter encoding of the same value exists.
    NonMinimal,
    /// The value does not fit `u32`.
    Overflow,
}

impl fmt::Display for VarintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            VarintError::Truncated => "ends inside a varint",
            VarintError::NonMinimal => "is not minimally encoded",
            VarintError::Overflow => "is larger than 32 bits",
        })
    }
}

/// Read one minimal unsigned LEB128 `u32` from the start of `buf`, returning it and the bytes
/// it took.
#[inline]
pub fn read_varint(buf: &[u8]) -> Result<(u32, usize), VarintError> {
    let mut v: u32 = 0;
    for (i, &b) in buf.iter().enumerate() {
        if i == 4 && b > 0x0F {
            // The fifth byte holds only the top 4 bits of a u32; anything more, including a
            // continuation bit, overflows.
            return Err(VarintError::Overflow);
        }
        v |= ((b & 0x7F) as u32) << (7 * i);
        if b & 0x80 == 0 {
            if b == 0 && i > 0 {
                return Err(VarintError::NonMinimal);
            }
            return Ok((v, i + 1));
        }
    }
    Err(VarintError::Truncated)
}

fn parse_tag(tag: u8) -> Result<Codec, HeaderError> {
    if tag == LEGACY_MAGIC {
        return Err(HeaderError::LegacyFormat);
    }
    let version = tag >> 4;
    if version == ESCAPE {
        return Err(HeaderError::ReservedEscape(tag));
    }
    if version != VERSION {
        return Err(HeaderError::UnknownTag(tag));
    }
    Codec::from_u8(tag & 0x0F).ok_or(HeaderError::UnknownTag(tag))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderError {
    /// Shorter than the smallest header.
    TooShort(usize),
    /// A value in the pre-release 12-byte layout, which started with `0xA7`.
    LegacyFormat,
    /// A tag outside `0x10..=0x14`.
    UnknownTag(u8),
    /// A tag whose high nibble is `0xF`, reserved for a future header.
    ReservedEscape(u8),
    /// A flag bit this build does not know is set.
    UnknownFlags(u8),
    /// The token count is malformed.
    Count(VarintError),
    MissingTable(Codec),
    /// A raw payload that is not `count * width` bytes.
    PayloadLen {
        want: usize,
        got: usize,
    },
    /// A token count above [`MAX_TOKENS`].
    TooManyTokens(u32),
    /// A `freq` payload that cannot hold `n_tokens` varints of 1 to 5 bytes, or a `freq_lz4`
    /// payload that cannot decompress to that many.
    FreqPayloadLen {
        n_tokens: u32,
        got: usize,
    },
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderError::TooShort(n) => write!(
                f,
                "value is {n} bytes, shorter than the {MIN_LEN}-byte minimum header"
            ),
            HeaderError::LegacyFormat => write!(
                f,
                "unknown value format: tag 0xA7 is the pre-release 12-byte header, which this \
                 build no longer reads; re-insert the data"
            ),
            HeaderError::UnknownTag(t) => write!(
                f,
                "unknown value format tag 0x{t:02X}, this build reads format version {VERSION} \
                 (tags 0x10-0x14)"
            ),
            HeaderError::ReservedEscape(t) => write!(
                f,
                "value format tag 0x{t:02X} uses the reserved 0xF escape, which this build does \
                 not read"
            ),
            HeaderError::UnknownFlags(b) => write!(
                f,
                "value has flags 0x{b:02X}, which this build does not understand"
            ),
            HeaderError::Count(e) => write!(f, "token count {e}"),
            HeaderError::TooManyTokens(n) => write!(
                f,
                "value claims {n} tokens, more than the {MAX_TOKENS} one value may hold"
            ),
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
            HeaderError::FreqPayloadLen { n_tokens, got } => write!(
                f,
                "freq payload is {got} bytes, which cannot hold {n_tokens} tokens"
            ),
        }
    }
}

impl std::error::Error for HeaderError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_raw24(n: u32) -> Vec<u8> {
        let mut v = Header::new(Codec::Raw24, 0, n).to_bytes();
        v.extend(std::iter::repeat_n(0u8, n as usize * 3));
        v
    }

    #[test]
    fn header_roundtrips() {
        for (codec, id, n) in [
            (Codec::Raw16, 0u16, 0u32),
            (Codec::Raw24, 0, 1),
            (Codec::Freq, 7, 2),
            (Codec::Raw8, 128, 300),
            (Codec::Freq, 65535, 5),
        ] {
            let h = Header::new(codec, id, n);
            let mut buf = h.to_bytes();
            let fill = codec.width().unwrap_or(1) * n as usize;
            buf.extend(std::iter::repeat_n(1u8, fill));
            let (parsed, payload) = Header::parse(&buf).expect("should parse");
            assert_eq!(parsed, h);
            assert_eq!(payload.len(), fill);
        }
    }

    #[test]
    fn golden_headers() {
        // Tag, flags 0, id u16 LE, then the count as LEB128.
        assert_eq!(
            Header::new(Codec::Raw16, 0, 0).to_bytes(),
            [0x10, 0, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            Header::new(Codec::Raw24, 1, 127).to_bytes(),
            [0x11, 0, 0x01, 0x00, 0x7F]
        );
        assert_eq!(
            Header::new(Codec::Freq, 65280, 128).to_bytes(),
            [0x12, 0, 0x00, 0xFF, 0x80, 0x01]
        );
        assert_eq!(
            Header::new(Codec::Raw8, 300, 2000).to_bytes(),
            [0x13, 0, 0x2C, 0x01, 0xD0, 0x0F]
        );
        assert_eq!(
            Header::new(Codec::Freq, 1, u32::MAX).to_bytes(),
            [0x12, 0, 0x01, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]
        );
    }

    #[test]
    fn rejects_short_buffer() {
        assert_eq!(Header::parse(&[]), Err(HeaderError::TooShort(0)));
        assert_eq!(
            Header::parse(&[0x11, 0, 0, 0]),
            Err(HeaderError::TooShort(4))
        );
    }

    #[test]
    fn rejects_the_old_12_byte_header() {
        let old = [0xA7, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 7];
        assert_eq!(Header::parse(&old), Err(HeaderError::LegacyFormat));
        assert_eq!(Header::parse(&old[..1]), Err(HeaderError::LegacyFormat));
        assert!(HeaderError::LegacyFormat
            .to_string()
            .starts_with("unknown value format"));
    }

    #[test]
    fn rejects_every_unknown_tag() {
        for tag in 0u8..=255 {
            let buf = [tag, 0, 0, 0, 0];
            let r = Header::parse(&buf);
            match tag {
                0x10 | 0x11 | 0x13 => assert!(r.is_ok(), "tag 0x{tag:02X}"),
                0x12 => assert_eq!(r, Err(HeaderError::MissingTable(Codec::Freq))),
                0x14 => assert_eq!(r, Err(HeaderError::MissingTable(Codec::FreqLz4))),
                0xA7 => assert_eq!(r, Err(HeaderError::LegacyFormat)),
                0xF0..=0xFF => assert_eq!(r, Err(HeaderError::ReservedEscape(tag))),
                _ => assert_eq!(r, Err(HeaderError::UnknownTag(tag)), "tag 0x{tag:02X}"),
            }
        }
    }

    #[test]
    fn rejects_every_unknown_flag() {
        for bit in 0..8 {
            let flags = 1u8 << bit;
            assert_eq!(
                Header::parse(&[0x11, flags, 0, 0, 0]),
                Err(HeaderError::UnknownFlags(flags))
            );
        }
    }

    #[test]
    fn rejects_a_malformed_count() {
        // 1 as two bytes; 0 as two bytes; ends mid-varint; 2^32.
        assert_eq!(
            Header::parse(&[0x13, 0, 1, 0, 0x81, 0x00, 7]),
            Err(HeaderError::Count(VarintError::NonMinimal))
        );
        assert_eq!(
            Header::parse(&[0x13, 0, 1, 0, 0x80, 0x00]),
            Err(HeaderError::Count(VarintError::NonMinimal))
        );
        assert_eq!(
            Header::parse(&[0x13, 0, 1, 0, 0x80]),
            Err(HeaderError::Count(VarintError::Truncated))
        );
        assert_eq!(
            Header::parse(&[0x13, 0, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x10]),
            Err(HeaderError::Count(VarintError::Overflow))
        );
    }

    #[test]
    fn rejects_a_raw_payload_that_disagrees_with_the_count() {
        let mut v = valid_raw24(4);
        v.pop();
        assert_eq!(
            Header::parse(&v),
            Err(HeaderError::PayloadLen { want: 12, got: 11 })
        );
        let mut v = valid_raw24(4);
        v.extend_from_slice(&[0, 0, 0]);
        assert_eq!(
            Header::parse(&v),
            Err(HeaderError::PayloadLen { want: 12, got: 15 })
        );
    }

    #[test]
    fn rejects_a_freq_payload_that_cannot_hold_the_count() {
        assert_eq!(
            Header::parse(&[0x12, 0, 1, 0, 3, 0, 0]),
            Err(HeaderError::FreqPayloadLen {
                n_tokens: 3,
                got: 2
            })
        );
        assert_eq!(
            Header::parse(&[0x12, 0, 1, 0, 0, 0]),
            Err(HeaderError::FreqPayloadLen {
                n_tokens: 0,
                got: 1
            })
        );
    }

    #[test]
    fn freq_still_requires_a_vocabulary() {
        assert_eq!(
            Header::parse(&[0x12, 0, 0, 0, 1, 1]),
            Err(HeaderError::MissingTable(Codec::Freq))
        );
    }

    #[test]
    fn raw_codecs_may_carry_a_vocabulary_id() {
        // Every column belongs to a vocabulary now, including `raw` ones, and the value must
        // record which — for the cast's type check, and for detokenization later.
        let (h, _) = Header::parse(&[0x11, 0, 0x03, 0x00, 1, 0, 0, 0]).expect("must parse");
        assert_eq!(h.vocabulary_id, 3);
    }

    #[test]
    fn empty_token_sequence_is_valid() {
        let v = valid_raw24(0);
        assert_eq!(v, [0x11, 0, 0, 0, 0]);
        let (h, payload) = Header::parse(&v).expect("empty should be valid");
        assert_eq!(h.n_tokens, 0);
        assert!(payload.is_empty());
    }

    #[test]
    fn codec_id_ranges() {
        assert_eq!(Codec::Raw16.max_id(), Some(65_535));
        assert_eq!(Codec::Raw24.max_id(), Some(16_777_215));
        assert_eq!(Codec::Freq.max_id(), None);
        assert_eq!(Codec::Raw8.max_id(), Some(255));
        assert_eq!(Codec::FreqLz4.max_id(), None);
        assert_eq!(Codec::from_u8(4), Some(Codec::FreqLz4));
        assert_eq!(Codec::from_u8(5), None);
    }
}
