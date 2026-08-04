//! The four payload encodings.
//!
//! [`raw`] packs token IDs at a fixed width with no table. [`freq`] remaps them to
//! frequency ranks and packs with streamvbyte. [`ans`] entropy-codes them against a static
//! unigram model. All are lossless: the token IDs come back exactly, and the tokenizer maps
//! them back to the source text.

pub mod ans;
pub mod freq;
pub mod raw;
