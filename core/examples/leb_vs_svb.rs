//! Decode speed of the `freq` payload: streamvbyte (today) against LEB128 (proposed).
//!
//! Both pack the same frequency ranks of real o200k text, so the only difference is the
//! varint scheme. Timed per token on short (4-20 tokens, like chat turns), medium (128) and
//! long (2000) values, since per-value overhead dominates short ones. Decode includes mapping
//! ranks back to ids, as the extension does.
//!
//! ```sh
//! uv run python benchmarks/export_rank_streams.py <dir>
//! cd core && cargo run --release --example leb_vs_svb -- <dir>
//! ```

use std::hint::black_box;
use std::time::Instant;

use pgtoken_core::tables::RankTable;
use stream_vbyte::{decode::decode, encode::encode, scalar::Scalar};

fn leb_encode(ranks: &[u32], out: &mut Vec<u8>) {
    for &r in ranks {
        let mut v = r;
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
}

/// Count is not stored: it is the number of terminating bytes (high bit clear).
fn leb_decode(payload: &[u8], table: &RankTable, out: &mut Vec<u32>) {
    let mut v = 0u32;
    let mut shift = 0;
    for &b in payload {
        v |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            out.push(table.token(v));
            v = 0;
            shift = 0;
        } else {
            shift += 7;
        }
    }
}

fn svb_encode(ranks: &[u32]) -> Vec<u8> {
    let mut buf = vec![0u8; ranks.len() * 5 + 16];
    let n = encode::<Scalar>(ranks, &mut buf);
    buf.truncate(n);
    buf
}

fn svb_decode(
    payload: &[u8],
    n: usize,
    table: &RankTable,
    scratch: &mut [u32],
    out: &mut Vec<u32>,
) {
    decode::<Scalar>(payload, n, scratch);
    out.extend(scratch[..n].iter().map(|&r| table.token(r)));
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: leb_vs_svb <dir from export_rank_streams.py>");
    let table =
        RankTable::from_bytes(&std::fs::read(format!("{dir}/o200k.tntt")).unwrap()).unwrap();
    let raw = std::fs::read(format!("{dir}/docs.bin")).unwrap();
    let mut docs: Vec<Vec<u32>> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let n = u32::from_le_bytes(raw[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        docs.push(
            (0..n)
                .map(|k| u32::from_le_bytes(raw[i + 4 * k..i + 4 * k + 4].try_into().unwrap()))
                .collect(),
        );
        i += 4 * n;
    }
    println!("{} docs loaded", docs.len());

    for (label, lo, hi) in [
        ("short 4-20", 4, 20),
        ("medium 128", 128, 128),
        ("long 2000", 2000, 2000),
    ] {
        let set: Vec<&Vec<u32>> = docs
            .iter()
            .filter(|d| d.len() >= lo && d.len() <= hi)
            .collect();
        let ranks: Vec<Vec<u32>> = set
            .iter()
            .map(|d| d.iter().map(|&t| table.rank(t).unwrap()).collect())
            .collect();
        let svb: Vec<Vec<u8>> = ranks.iter().map(|r| svb_encode(r)).collect();
        let leb: Vec<Vec<u8>> = ranks
            .iter()
            .map(|r| {
                let mut o = Vec::new();
                leb_encode(r, &mut o);
                o
            })
            .collect();
        let tokens: usize = set.iter().map(|d| d.len()).sum();
        let (sb, lb): (usize, usize) = (
            svb.iter().map(Vec::len).sum(),
            leb.iter().map(Vec::len).sum(),
        );

        // Correctness first: both must give back the original ids.
        let mut scratch = vec![0u32; 2048];
        for (k, d) in set.iter().enumerate() {
            let mut a = Vec::new();
            svb_decode(&svb[k], d.len(), &table, &mut scratch, &mut a);
            let mut b = Vec::new();
            leb_decode(&leb[k], &table, &mut b);
            assert_eq!(&a, *d);
            assert_eq!(&b, *d);
        }

        let reps = (4_000_000 / tokens).max(3);
        let mut best = [f64::MAX; 2];
        for _round in 0..5 {
            let t = Instant::now();
            for _ in 0..reps {
                for (k, d) in set.iter().enumerate() {
                    let mut out = Vec::with_capacity(d.len());
                    svb_decode(black_box(&svb[k]), d.len(), &table, &mut scratch, &mut out);
                    black_box(&out);
                }
            }
            best[0] = best[0].min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for _ in 0..reps {
                for (k, d) in set.iter().enumerate() {
                    let mut out = Vec::with_capacity(d.len());
                    leb_decode(black_box(&leb[k]), &table, &mut out);
                    black_box(&out);
                }
            }
            best[1] = best[1].min(t.elapsed().as_secs_f64());
        }
        let ns = |s: f64| s * 1e9 / (reps * tokens) as f64;
        println!(
            "{label:<11} {:>5} values {:>7} tokens | streamvbyte {:.3} B/tok {:6.2} ns/tok | leb128 {:.3} B/tok {:6.2} ns/tok | leb/svb time {:.2}",
            set.len(), tokens, sb as f64 / tokens as f64, ns(best[0]), lb as f64 / tokens as f64, ns(best[1]), best[1] / best[0]
        );
    }
}
