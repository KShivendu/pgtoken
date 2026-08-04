//! The payload encodings.
//!
//! [`raw`] packs token IDs at a fixed width with no table. [`freq`] remaps them to
//! frequency ranks and packs with streamvbyte. Both are lossless, and neither knows anything
//! about tokenizers or vocabulary sizes.

pub mod freq;
pub mod raw;
