//! The payload encodings.
//!
//! [`raw`] packs token IDs at a fixed width with no table. [`freq`] remaps them to
//! frequency ranks and packs each as a LEB128 varint, and [`freq_lz4`] runs that through lz4
//! when it helps. All are lossless, and neither knows anything
//! about tokenizers or vocabulary sizes.

pub mod freq;
pub mod freq_lz4;
pub mod raw;
