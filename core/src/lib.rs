//! Compress sequences of token IDs.
//!
//! Text is stored as the token IDs a model already works in, rather than UTF-8, which is both
//! smaller and means a reader never has to re-tokenize. See the blog and paper linked from the
//! README for why.
//!
//! **This library has no tokenizer.** It takes `&[u32]` and gives back `&[u32]`. Turning text
//! into IDs is the caller's job, with whatever tokenizer they already run — a database has no
//! business holding an opinion about that, and staying out of it means any tokenizer works,
//! including ones that do not exist yet. Nothing here needs a vocabulary size either; see
//! [`tables::RankTable`] for how that is avoided.
//!
//! Layering: [`header`] is the self-describing value envelope, [`codec`] the payload
//! encodings, [`tables`] the trained frequency table, [`value`] the API most callers want.

pub mod codec;
pub mod header;
pub mod tables;
pub mod value;

pub use header::{Codec, Header, HeaderError, HEADER_LEN, MAGIC, VERSION};
pub use tables::{RankTable, TableError};
pub use value::{decode, describe, encode, recode, ValueError};
