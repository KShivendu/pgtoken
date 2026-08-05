//! Codec cost in isolation, with no database and no client library in the way.
//!
//! The Python client in `benchmarks/` is vectorised but still pays numpy overhead on
//! 512-element arrays, which inflates `freq` well beyond what the codec actually costs. This
//! measures the Rust implementation directly, which is what the extension runs and what a Rust
//! client would get.
//!
//! ```sh
//! cd core && cargo run --release --example codec_bench
//! ```
//!
//! Release mode matters: a debug build reports several times the real cost.

use std::time::Instant;

use pgtoken_core::header::Codec;
use pgtoken_core::tables::RankTable;
use pgtoken_core::value;

const CHUNK: usize = 512;
const CHUNKS: usize = 2000;
const VOCAB: u32 = 200_019;

/// A Zipf-ish token stream, which is what real text looks like to these codecs: a few tokens
/// dominate and the tail is long. Speed depends on the magnitude of the values being packed,
/// so the shape of the distribution is what matters, not that it be real text.
fn synth(n: usize) -> Vec<u32> {
    // xorshift, so the sequence is reproducible without pulling in a rand dependency.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n)
        .map(|_| {
            let r = (next() % 1_000_000) as f64 / 1_000_000.0;
            // Heavy head, long tail: ~1/x over the vocabulary.
            let id = (VOCAB as f64).powf(r) as u32;
            id.min(VOCAB - 1)
        })
        .collect()
}

fn time<F: FnMut()>(label: &str, iters: usize, per_iter: usize, mut f: F) -> f64 {
    f(); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!(
        "  {label:<28} {us:>8.2} us/chunk  ({:>5.2} us per 1k tokens)",
        us * 1000.0 / per_iter as f64
    );
    us
}

fn main() {
    let corpus = synth(CHUNK * CHUNKS);
    let table = RankTable::train(&corpus, None).expect("train");
    println!(
        "codec cost, {CHUNK}-token chunks, Zipf-ish over {VOCAB} ids, {} ranked\n",
        table.k()
    );

    let chunk: Vec<u32> = corpus[..CHUNK].to_vec();

    for (name, codec, tid, tbl) in [
        ("raw16", Codec::Raw16, 0u16, None),
        ("raw24", Codec::Raw24, 0, None),
        ("freq", Codec::Freq, 1, Some(&table)),
    ] {
        // raw16 cannot hold this vocabulary; show it on a narrowed stream instead so the
        // comparison is still meaningful.
        let ids: Vec<u32> = if codec == Codec::Raw16 {
            chunk.iter().map(|&i| i % 60_000).collect()
        } else {
            chunk.clone()
        };

        let encoded = value::encode(&ids, codec, tid, tbl).expect("encode");
        let ratio = (ids.len() * 4) as f64 / encoded.len() as f64;
        println!(
            "{name}  ({} B/chunk, {:.2} B/token, {ratio:.2}x vs int32)",
            encoded.len(),
            encoded.len() as f64 / ids.len() as f64
        );

        time("encode", 2000, ids.len(), || {
            std::hint::black_box(value::encode(&ids, codec, tid, tbl).unwrap());
        });
        time("decode", 2000, ids.len(), || {
            std::hint::black_box(value::decode(&encoded, tbl).unwrap());
        });
        println!();
    }
}
