//! Token-native storage codecs.
//!
//! Stores text as its BPE token IDs rather than UTF-8 bytes. See the paper in
//! `research/token-native-storage/` for why; the short version is that the systems reading
//! and writing this text work in token IDs, so a token-native store hands them the IDs
//! directly instead of re-tokenizing on every read.
//!
//! Layering: [`header`] defines the self-describing value envelope, [`codec`] the four
//! payload encodings. Neither depends on Postgres.

pub mod codec;
pub mod header;
pub mod tables;
pub mod tokenizer;
pub mod value;

pub use header::{Codec, Header, HeaderError, Tokenizer, HEADER_LEN, MAGIC, VERSION};
pub use tables::{AnsTable, RankTable, TableError};
pub use value::{
    decode_ids, decode_text, describe, encode_ids, encode_text, recode, resolve_codec, Tables,
    ValueError,
};
