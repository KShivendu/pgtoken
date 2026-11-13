//! `freq_lz4`: the `freq` payload (one LEB128 rank per token) compressed as one lz4 block.
//!
//! `freq` codes each token on its own, so a line repeated ten times costs ten times as much.
//! lz4 finds those repeats. On real o200k code it made 512-token values 1.33x smaller and
//! 2000-token values 1.51x smaller, for about 3 ns/token more to encode and 0.5 ns/token more to
//! decode; on English prose it saved 1-5% (`core/examples/lz_on_freq.rs`). PostgreSQL's own
//! compression cannot do this for most values, since it only runs on rows over about 2 KB.
//!
//! An encoder writes this codec only when it is strictly smaller than plain `freq`, so short or
//! repeat-free values stay `freq` and decode exactly as before.
//!
//! **Equality.** lz4 output is not unique: many blocks decompress to the same bytes, and a newer
//! lz4 library may pick a different one. So two `freq_lz4` values, or a `freq` and a `freq_lz4`
//! value, can hold the same tokens in different bytes. `=` and hashing therefore compare the
//! decompressed `freq` bytes, which are unique per token list (see [`crate::value::canonical`]).

use super::freq::FreqError;

/// lz4-compress a `freq` payload, if that makes it strictly smaller.
pub fn compress(freq: &[u8]) -> Option<Vec<u8>> {
    let c = lz4_flex::block::compress(freq);
    (c.len() < freq.len()).then_some(c)
}

/// Decompress a `freq_lz4` payload holding `n_tokens` ranks back to the `freq` payload. The
/// result is at most `5 * n_tokens` bytes (a rank is 1 to 5 bytes), so a block that would grow
/// past that is refused rather than allocated. The ranks themselves are checked by the `freq`
/// decoder.
pub fn decompress(payload: &[u8], n_tokens: usize) -> Result<Vec<u8>, FreqError> {
    let cap = n_tokens
        .saturating_mul(5)
        .min(payload.len().saturating_mul(255));
    let mut out = vec![0u8; cap];
    let n = lz4_flex::block::decompress_into(payload, &mut out)
        .map_err(|e| FreqError::Lz4(e.to_string()))?;
    out.truncate(n);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_only_a_smaller_block() {
        assert_eq!(compress(&[1, 2, 3]), None, "too short to gain anything");
        let repeated: Vec<u8> = [5u8, 9, 0x81, 0x01].repeat(50);
        let c = compress(&repeated).expect("a repeated pattern compresses");
        assert!(c.len() < repeated.len() / 4, "{} bytes", c.len());
        assert_eq!(decompress(&c, 150).unwrap(), repeated);
    }

    #[test]
    fn refuses_garbage_and_overlong_output() {
        assert!(decompress(&[0xFF, 0xFF, 0xFF], 10).is_err());
        let repeated: Vec<u8> = [5u8, 9].repeat(200);
        let c = compress(&repeated).unwrap();
        // 400 bytes of output, but 10 tokens can be at most 50 bytes of ranks.
        assert!(matches!(decompress(&c, 10), Err(FreqError::Lz4(_))));
    }
}
