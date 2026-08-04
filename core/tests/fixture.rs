//! Cross-language validation against the Python reference implementation.
//!
//! The unit tests prove the Rust codecs are self-consistent. This proves they agree with the
//! reference behind the paper, which lives in the separate token-storage repo along with the
//! corpora. Generate the fixture with that repo's `harness/export_fixture.py` and symlink or
//! copy it to `../harness/fixture` here.
//!
//! These tests skip rather than fail when the fixture is absent, so a fresh clone runs
//! `cargo test` without needing a 9 MB fixture or a Python environment.
//!
//! Four checks, in increasing strength:
//!
//! 1. Python-written TNTT tables load in Rust, byte-identically on re-serialize. Validates
//!    the on-disk format from both ends.
//! 2. Rust tokenization agrees with Python's token counts. Checked before any size
//!    comparison, because a mismatch here would make all of them meaningless.
//! 3. Every chunk roundtrips text -> value -> text across all three corpora, all
//!    tokenizers, all codecs. The lossless claim, including r50k-on-Hindi, the worst case.
//! 4. Rust decodes Python's actual payload bytes back to the original text.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;

use pgtoken_core::codec::{ans as ans_codec, raw as raw_codec};
use pgtoken_core::header::{Codec, Tokenizer};
use pgtoken_core::tables::{AnsTable, RankTable};
use pgtoken_core::{tokenizer, value};

/// `freq` sizes are compared as an aggregate ratio within this relative tolerance, because
/// the Rust `stream-vbyte` crate and Python's `pyfastpfor` use different container layouts.
///
/// Measured spread across the nine (corpus, tokenizer) cases is under 1.2%, with Rust
/// consistently the *better* of the two, so 3% leaves headroom without letting a real
/// regression through. A Rust ratio that came out materially *worse* than Python's would
/// mean the rank remap or the varint packing had broken, so the direction is asserted too.
const FREQ_RATIO_TOLERANCE: f64 = 0.03;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/fixture")
}

/// Returns None when the fixture has not been generated.
fn manifest() -> Option<(PathBuf, Json)> {
    let dir = fixture_dir();
    let text = fs::read_to_string(dir.join("manifest.json")).ok()?;
    Some((dir, serde_json::from_str(&text).expect("manifest.json should be valid JSON")))
}

macro_rules! skip_without_fixture {
    () => {
        match manifest() {
            Some(m) => m,
            None => {
                eprintln!(
                    "skipping: no fixture at {}; run \
                     the export_fixture.py script in the token-storage repo",
                    fixture_dir().display()
                );
                return;
            }
        }
    };
}

struct Case {
    domain: String,
    tokenizer: Tokenizer,
    rank: RankTable,
    rank_bytes: Vec<u8>,
    ans: AnsTable,
    texts: Vec<String>,
    n_tokens: Vec<usize>,
    raw_utf8: Vec<usize>,
    python_sizes: HashMap<String, Vec<usize>>,
    python_payloads: HashMap<String, Vec<Vec<u8>>>,
}

impl Case {
    fn tag(&self) -> String {
        format!("{}/{}", self.domain, self.tokenizer.as_str())
    }
}

fn str_at<'a>(v: &'a Json, key: &str) -> &'a str {
    v.get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("manifest case is missing string field {key:?}"))
}

fn usize_list(v: &Json, key: &str) -> Vec<usize> {
    v.get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("manifest case is missing array field {key:?}"))
        .iter()
        .map(|x| x.as_u64().expect("expected a number") as usize)
        .collect()
}

fn read_chunks(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("chunk file {}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Json = serde_json::from_str(l).expect("chunk line should be valid JSON");
            str_at(&v, "text").to_owned()
        })
        .collect()
}

/// Read the length-prefixed payload blob file written by the exporter:
/// u32 LE record count, then per record a u32 LE length followed by that many bytes.
fn read_payloads(path: &Path) -> Vec<Vec<u8>> {
    let buf = fs::read(path).unwrap_or_else(|e| panic!("payload file {}: {e}", path.display()));
    assert!(buf.len() >= 4, "payload file {} is truncated", path.display());
    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut off = 4usize;
    for i in 0..count {
        assert!(off + 4 <= buf.len(), "record {i} length is truncated");
        let len = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        assert!(off + len <= buf.len(), "record {i} payload is truncated");
        out.push(buf[off..off + len].to_vec());
        off += len;
    }
    out
}

fn load_cases(dir: &Path, m: &Json) -> Vec<Case> {
    m.get("cases")
        .and_then(Json::as_array)
        .expect("manifest needs a cases array")
        .iter()
        .map(|c| {
            let domain = str_at(c, "domain").to_owned();
            let tname = str_at(c, "tokenizer");
            let tok = Tokenizer::parse(tname).expect("manifest names a known tokenizer");

            let rank_bytes = fs::read(dir.join(str_at(c, "rank_table"))).expect("rank table");
            let ans_bytes = fs::read(dir.join(str_at(c, "ans_table"))).expect("ans table");
            let rank = RankTable::from_bytes(&rank_bytes, tok)
                .unwrap_or_else(|e| panic!("{domain}/{tname} rank table: {e}"));
            let ans = AnsTable::from_bytes(&ans_bytes, tok)
                .unwrap_or_else(|e| panic!("{domain}/{tname} ans table: {e}"));

            let mut python_sizes = HashMap::new();
            if let Some(obj) = c.get("python_payload_bytes").and_then(Json::as_object) {
                for (k, v) in obj {
                    python_sizes.insert(
                        k.clone(),
                        v.as_array()
                            .expect("size list")
                            .iter()
                            .map(|x| x.as_u64().unwrap() as usize)
                            .collect::<Vec<_>>(),
                    );
                }
            }
            let mut python_payloads = HashMap::new();
            if let Some(obj) = c.get("python_payload_files").and_then(Json::as_object) {
                for (k, v) in obj {
                    python_payloads.insert(
                        k.clone(),
                        read_payloads(&dir.join(v.as_str().expect("payload path"))),
                    );
                }
            }

            Case {
                domain,
                tokenizer: tok,
                rank,
                rank_bytes,
                ans,
                texts: read_chunks(&dir.join(str_at(c, "chunks"))),
                n_tokens: usize_list(c, "n_tokens"),
                raw_utf8: usize_list(c, "raw_utf8_bytes"),
                python_sizes,
                python_payloads,
            }
        })
        .collect()
}

fn codecs_for(tok: Tokenizer) -> Vec<(&'static str, Codec)> {
    let mut v = vec![("raw24", Codec::Raw24), ("freq", Codec::Freq), ("ans", Codec::Ans)];
    if tok.fits_u16() {
        v.push(("raw16", Codec::Raw16));
    }
    v
}

fn median(mut v: Vec<f64>) -> f64 {
    assert!(!v.is_empty(), "median of an empty sample");
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[test]
fn python_written_tables_load_in_rust() {
    let (dir, m) = skip_without_fixture!();
    let cases = load_cases(&dir, &m);
    assert!(!cases.is_empty(), "fixture has no cases");
    for c in &cases {
        assert_eq!(c.rank.tokenizer, c.tokenizer, "{}: rank table tokenizer", c.tag());
        assert_eq!(c.ans.tokenizer, c.tokenizer, "{}: ans table tokenizer", c.tag());
        assert_eq!(c.rank.vocab(), c.tokenizer.vocab(), "{}: vocab", c.tag());
        // Re-serializing must reproduce the file, which pins the format rather than just
        // proving it parses.
        assert_eq!(
            c.rank.to_bytes(),
            c.rank_bytes,
            "{}: rank table did not survive a Rust round trip byte-identically",
            c.tag()
        );
    }
}

#[test]
fn tokenization_matches_python() {
    let (dir, m) = skip_without_fixture!();
    for c in load_cases(&dir, &m) {
        for (i, text) in c.texts.iter().enumerate() {
            let got = tokenizer::encode(text, c.tokenizer).len();
            assert_eq!(
                got,
                c.n_tokens[i],
                "{} chunk {i}: Rust produced {got} tokens, Python {}",
                c.tag(),
                c.n_tokens[i]
            );
        }
    }
}

#[test]
fn every_chunk_roundtrips_losslessly() {
    let (dir, m) = skip_without_fixture!();
    let mut checked = 0usize;
    for c in load_cases(&dir, &m) {
        let tables = value::Tables { rank: Some(&c.rank), ans: Some(&c.ans) };
        for (name, codec) in codecs_for(c.tokenizer) {
            let table_id = if codec.needs_table() { 1 } else { 0 };
            for (i, text) in c.texts.iter().enumerate() {
                let v = value::encode_text(text, c.tokenizer, codec, table_id, tables)
                    .unwrap_or_else(|e| panic!("{}/{name} chunk {i} encode: {e}", c.tag()));
                let back = value::decode_text(&v, tables)
                    .unwrap_or_else(|e| panic!("{}/{name} chunk {i} decode: {e}", c.tag()));
                assert_eq!(&back, text, "{}/{name} chunk {i} is not lossless", c.tag());
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "fixture produced no comparisons");
    eprintln!("roundtripped {checked} (corpus, tokenizer, codec, chunk) combinations");
}

#[test]
fn payload_sizes_match_python() {
    let (dir, m) = skip_without_fixture!();
    // Collect every mismatch before failing. Asserting inline would stop at the first
    // problem and hide whether the rest of the grid agrees, which is the thing worth
    // knowing when a codec regresses.
    let mut failures: Vec<String> = Vec::new();
    let mut exact = 0usize;

    for c in load_cases(&dir, &m) {
        let tables = value::Tables { rank: Some(&c.rank), ans: Some(&c.ans) };
        for (name, codec) in codecs_for(c.tokenizer) {
            let Some(py) = c.python_sizes.get(name) else { continue };
            let table_id = if codec.needs_table() { 1 } else { 0 };

            let rust: Vec<usize> = c
                .texts
                .iter()
                .map(|text| {
                    let v =
                        value::encode_text(text, c.tokenizer, codec, table_id, tables).unwrap();
                    // Python sizes exclude the 12-byte value header.
                    v.len() - pgtoken_core::HEADER_LEN
                })
                .collect();

            let tag = format!("{}/{name}", c.tag());
            match codec {
                // Deterministic fixed-width packing, and the same entropy coder from the
                // same library: all three must agree byte-for-byte in size.
                Codec::Raw16 | Codec::Raw24 | Codec::Ans => {
                    if rust == *py {
                        exact += 1;
                    } else {
                        let n_diff = rust.iter().zip(py).filter(|(a, b)| a != b).count();
                        let first = rust
                            .iter()
                            .zip(py)
                            .enumerate()
                            .find(|(_, (a, b))| a != b)
                            .map(|(i, (a, b))| format!("chunk {i}: Rust {a} B vs Python {b} B"))
                            .unwrap_or_else(|| "length mismatch".into());
                        failures.push(format!(
                            "{tag}: {n_diff}/{} chunks differ ({first})",
                            rust.len()
                        ));
                    }
                }
                // Different varint container, so compare the aggregate ratio rather than
                // bytes, and record the direction.
                Codec::Freq => {
                    let ratio = |sizes: &[usize]| {
                        median(
                            c.raw_utf8
                                .iter()
                                .zip(sizes)
                                .map(|(&raw, &enc)| raw as f64 / enc as f64)
                                .collect(),
                        )
                    };
                    let (r, p) = (ratio(&rust), ratio(py));
                    let rel = (r - p).abs() / p;
                    if rel > FREQ_RATIO_TOLERANCE {
                        failures.push(format!(
                            "{tag}: median ratio {r:.4}x vs Python {p:.4}x, {:.2}% apart \
                             (tolerance {:.0}%)",
                            rel * 100.0,
                            FREQ_RATIO_TOLERANCE * 100.0
                        ));
                    } else if r < p * (1.0 - FREQ_RATIO_TOLERANCE) {
                        failures.push(format!(
                            "{tag}: Rust ratio {r:.4}x is materially worse than Python's \
                             {p:.4}x; the rank remap or varint packing has regressed"
                        ));
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "payload sizes disagree with Python in {} case(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    eprintln!("{exact} (corpus, tokenizer, codec) cases matched Python's sizes exactly");
}

#[test]
fn rust_decodes_python_payloads() {
    // The strongest cross-language check: decode the bytes Python actually produced. Skips
    // `freq`, whose container genuinely differs between the two varint libraries.
    let (dir, m) = skip_without_fixture!();
    let mut checked = 0usize;
    for c in load_cases(&dir, &m) {
        for (name, codec) in codecs_for(c.tokenizer) {
            if codec == Codec::Freq {
                continue;
            }
            let Some(payloads) = c.python_payloads.get(name) else { continue };
            for (i, text) in c.texts.iter().enumerate() {
                let (n, payload) = (c.n_tokens[i], &payloads[i]);
                let ids = match codec {
                    Codec::Raw16 => raw_codec::decode16(payload, n).unwrap(),
                    Codec::Raw24 => raw_codec::decode24(payload, n).unwrap(),
                    Codec::Ans => {
                        ans_codec::decode_ans(payload, n, &c.ans).unwrap_or_else(|e| {
                            panic!(
                                "{}/{name} chunk {i}: Rust could not decode Python's ANS \
                                 payload: {e}",
                                c.tag()
                            )
                        })
                    }
                    Codec::Freq => unreachable!("filtered above"),
                };
                let back = tokenizer::decode(&ids, c.tokenizer).unwrap();
                assert_eq!(
                    &back, text,
                    "{}/{name} chunk {i}: Python's bytes decoded by Rust give different text",
                    c.tag()
                );
                checked += 1;
            }
        }
    }
    eprintln!("decoded {checked} Python-produced payloads in Rust");
}

#[test]
fn reports_ratios_against_the_paper() {
    // A report, not an assertion: prints the median ratio per (corpus, tokenizer, codec) so
    // the numbers can be compared against Table 1 of the paper. Run with --nocapture.
    let (dir, m) = skip_without_fixture!();
    println!("\nRust median compression ratio vs raw UTF-8 (includes the 12-byte header)");
    println!("{:<8} {:<8} {:>8} {:>8} {:>8} {:>8}", "corpus", "tok", "raw16", "raw24", "freq", "ans");
    for c in load_cases(&dir, &m) {
        let tables = value::Tables { rank: Some(&c.rank), ans: Some(&c.ans) };
        let mut cell: HashMap<&str, String> = HashMap::new();
        for (name, codec) in codecs_for(c.tokenizer) {
            let table_id = if codec.needs_table() { 1 } else { 0 };
            let ratios: Vec<f64> = c
                .texts
                .iter()
                .zip(&c.raw_utf8)
                .map(|(text, &raw)| {
                    let v =
                        value::encode_text(text, c.tokenizer, codec, table_id, tables).unwrap();
                    raw as f64 / v.len() as f64
                })
                .collect();
            cell.insert(name, format!("{:.2}x", median(ratios)));
        }
        let get = |k: &str| cell.get(k).map(String::as_str).unwrap_or("-").to_owned();
        println!(
            "{:<8} {:<8} {:>8} {:>8} {:>8} {:>8}",
            c.domain,
            c.tokenizer.as_str(),
            get("raw16"),
            get("raw24"),
            get("freq"),
            get("ans"),
        );
    }
}
