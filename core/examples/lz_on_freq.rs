//! Does LZ compression on top of the `freq` payload pay off, and how fast is it?
//!
//! `freq` (frequency-rank remap, one LEB128 varint per token) codes each token on its own, so a
//! repeated line costs as much the tenth time as the first. Postgres only compresses a row over
//! about 2 KB, so most tokens values never get LZ. This measures doing it inside the value:
//!
//!   freq                      today's codec (the control)
//!   freq + lz4                LEB128 bytes through lz4, kept only when smaller
//!   freq + lz4 + dict         the same, lz4 primed with 64 KB of training payloads
//!   freq + zstd-3             zstd level 3, no dictionary, kept only when smaller
//!   freq + zstd-3 + dict      zstd level 3 with a 112 KB dictionary trained on freq payloads
//!   freq + zstd-19 + dict     the ceiling: slow to encode, best size
//!   text + lz4 / zstd + dict  the same compressors on the UTF-8 text, for reference
//!
//! Sizes include pgtoken's value header (4 bytes + LEB128 count) for token arms; text arms
//! count only compressed bytes. Times include everything the extension would do: id -> rank
//! and LEB128 on encode, the reverse on decode. Every arm is checked to round-trip first.
//!
//! ```sh
//! uv run python benchmarks/export_codec_corpus.py bench_data/lz_on_freq
//! cd core && cargo run --release --example lz_on_freq -- ../bench_data/lz_on_freq
//! ```

use std::hint::black_box;
use std::time::Instant;

use pgtoken_core::codec::freq::{decode_freq, encode_freq};
use pgtoken_core::tables::{ByteMap, RankTable};
use zstd::bulk::{Compressor, Decompressor};
use zstd::dict::{DecoderDictionary, EncoderDictionary};
use zstd::zstd_safe::{CParameter, DParameter, FrameFormat};

const DOMAINS: [&str; 4] = ["prose", "code", "hindi", "repeat"];

fn read_records(path: &str) -> Vec<(u8, Vec<u32>)> {
    let raw = std::fs::read(path).unwrap();
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let dom = raw[i];
        let n = u32::from_le_bytes(raw[i + 1..i + 5].try_into().unwrap()) as usize;
        i += 5;
        out.push((
            dom,
            (0..n)
                .map(|k| u32::from_le_bytes(raw[i + 4 * k..i + 4 * k + 4].try_into().unwrap()))
                .collect(),
        ));
        i += 4 * n;
    }
    out
}

fn leb_len(mut n: usize) -> usize {
    let mut l = 1;
    while n >= 0x80 {
        n >>= 7;
        l += 1;
    }
    l
}

/// Stored bytes for a token arm: the value header (tag, flags, u16 vocabulary id, LEB128 count)
/// plus the payload. `payload[0]` is a marker the codec tag would carry, so it is not counted.
fn stored(n: usize, payload: &[u8]) -> usize {
    4 + leb_len(n) + payload.len() - 1
}

type Enc<'a> = Box<dyn FnMut(&[u32], &[u8], &mut Vec<u8>) + 'a>;
type Dec<'a> = Box<dyn FnMut(&[u8], usize, &mut Vec<u8>) -> Vec<u32> + 'a>;

/// One way to store a value. `enc` gets the ids and their text and writes a payload whose first
/// byte is a marker (0 = plain freq, 1 = compressed); `dec` returns the ids (text arms return
/// the text's bytes, widened, so the round-trip check still works).
struct Arm<'a> {
    name: &'static str,
    text: bool,
    enc: Enc<'a>,
    dec: Dec<'a>,
}

fn zcomp<'a>(level: i32, dict: Option<&'a EncoderDictionary<'static>>) -> Compressor<'a> {
    let mut c = match dict {
        Some(d) => Compressor::with_prepared_dictionary(d).unwrap(),
        None => Compressor::new(level).unwrap(),
    };
    c.set_parameter(CParameter::Format(FrameFormat::Magicless)).unwrap();
    c.set_parameter(CParameter::ChecksumFlag(false)).unwrap();
    c.set_parameter(CParameter::ContentSizeFlag(false)).unwrap();
    c.set_parameter(CParameter::DictIdFlag(false)).unwrap();
    c
}

fn zdecomp<'a>(dict: Option<&'a DecoderDictionary<'static>>) -> Decompressor<'a> {
    let mut d = match dict {
        Some(d) => Decompressor::with_prepared_dictionary(d).unwrap(),
        None => Decompressor::new().unwrap(),
    };
    d.set_parameter(DParameter::Format(FrameFormat::Magicless)).unwrap();
    d
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: lz_on_freq <dir from export_codec_corpus.py>");
    let table = RankTable::from_bytes(&std::fs::read(format!("{dir}/o200k.tntt")).unwrap()).unwrap();
    let map = ByteMap::from_bytes(&std::fs::read(format!("{dir}/o200k.tnmap")).unwrap()).unwrap();
    let text_of = |ids: &[u32]| -> Vec<u8> { ids.iter().flat_map(|&t| map.get(t).unwrap_or(&[]).iter().copied()).collect() };
    let freq_of = |ids: &[u32]| -> Vec<u8> {
        let mut o = Vec::new();
        encode_freq(ids, &table, &mut o).unwrap();
        o
    };

    // Dictionaries, trained on the train split only (4500 windows of 512 ids, all domains).
    let train = read_records(&format!("{dir}/train.bin"));
    let t0 = Instant::now();
    let freq_samples: Vec<Vec<u8>> = train.iter().map(|(_, ids)| freq_of(ids)).collect();
    let text_samples: Vec<Vec<u8>> = train.iter().map(|(_, ids)| text_of(ids)).collect();
    let zdict = zstd::dict::from_samples(&freq_samples, 112 * 1024).unwrap();
    let tdict = zstd::dict::from_samples(&text_samples, 112 * 1024).unwrap();
    // lz4 has no trainer; its dictionary is just a 64 KB prefix it may copy from. Interleave
    // training payloads across domains so each is represented.
    let mut ldict = Vec::new();
    for k in 0.. {
        let s = &freq_samples[(k * 7919) % freq_samples.len()];
        if ldict.len() + s.len() > 64 * 1024 {
            break;
        }
        ldict.extend_from_slice(s);
    }
    println!(
        "dictionaries: zstd on freq {} B, zstd on text {} B, lz4 prefix {} B ({:.1} s)",
        zdict.len(),
        tdict.len(),
        ldict.len(),
        t0.elapsed().as_secs_f64()
    );

    let ed3 = EncoderDictionary::copy(&zdict, 3);
    let ed19 = EncoderDictionary::copy(&zdict, 19);
    let dd = DecoderDictionary::copy(&zdict);
    let ted = EncoderDictionary::copy(&tdict, 3);
    let tdd = DecoderDictionary::copy(&tdict);

    // Plain freq, or `compressed` behind marker 1 when that is smaller.
    fn keep_smaller(freq: &[u8], compressed: &[u8], out: &mut Vec<u8>) {
        out.clear();
        if compressed.len() < freq.len() {
            out.push(1);
            out.extend_from_slice(compressed);
        } else {
            out.push(0);
            out.extend_from_slice(freq);
        }
    }
    let t = &table;
    let freq_dec = move |p: &[u8], n: usize, scratch: &mut Vec<u8>, inflate: &mut dyn FnMut(&[u8], &mut Vec<u8>)| -> Vec<u32> {
        if p[0] == 0 {
            decode_freq(&p[1..], n, t).unwrap()
        } else {
            inflate(&p[1..], scratch);
            decode_freq(scratch, n, t).unwrap()
        }
    };

    let mut arms: Vec<Arm> = Vec::new();
    arms.push(Arm {
        name: "freq (today)",
        text: false,
        enc: Box::new(|ids, _, out| {
            out.clear();
            out.push(0);
            encode_freq(ids, t, out).unwrap();
        }),
        dec: Box::new(move |p, n, _| decode_freq(&p[1..], n, t).unwrap()),
    });
    {
        let mut f = Vec::new();
        arms.push(Arm {
            name: "freq + lz4",
            text: false,
            enc: Box::new(move |ids, _, out| {
                f.clear();
                encode_freq(ids, t, &mut f).unwrap();
                keep_smaller(&f, &lz4_flex::block::compress(&f), out);
            }),
            dec: Box::new(move |p, n, s| {
                freq_dec(p, n, s, &mut |c, s| {
                    s.resize(n * 5, 0);
                    let k = lz4_flex::block::decompress_into(c, s).unwrap();
                    s.truncate(k);
                })
            }),
        });
    }
    {
        let (mut f, d1, d2) = (Vec::new(), ldict.clone(), ldict.clone());
        arms.push(Arm {
            name: "freq + lz4 + dict",
            text: false,
            enc: Box::new(move |ids, _, out| {
                f.clear();
                encode_freq(ids, t, &mut f).unwrap();
                keep_smaller(&f, &lz4_flex::block::compress_with_dict(&f, &d1), out);
            }),
            dec: Box::new(move |p, n, s| {
                freq_dec(p, n, s, &mut |c, s| {
                    s.resize(n * 5, 0);
                    let k = lz4_flex::block::decompress_into_with_dict(c, s, &d2).unwrap();
                    s.truncate(k);
                })
            }),
        });
    }
    for (name, level, enc_dict) in [
        ("freq + zstd-3", 3, None),
        ("freq + zstd-3 + dict", 3, Some(&ed3)),
        ("freq + zstd-19 + dict", 19, Some(&ed19)),
    ] {
        let mut c = zcomp(level, enc_dict);
        let mut d = zdecomp(if enc_dict.is_some() { Some(&dd) } else { None });
        let mut f = Vec::new();
        let mut zbuf = vec![0u8; 64 * 1024];
        arms.push(Arm {
            name,
            text: false,
            enc: Box::new(move |ids, _, out| {
                f.clear();
                encode_freq(ids, t, &mut f).unwrap();
                let k = c.compress_to_buffer(&f, &mut zbuf).unwrap();
                keep_smaller(&f, &zbuf[..k], out);
            }),
            dec: Box::new(move |p, n, s| {
                freq_dec(p, n, s, &mut |cmp, s| {
                    s.clear();
                    s.reserve(n * 5);
                    d.decompress_to_buffer(cmp, s).unwrap();
                })
            }),
        });
    }
    // Text arms. Marker byte kept for symmetry; dec returns text bytes widened to u32.
    arms.push(Arm {
        name: "text + lz4",
        text: true,
        enc: Box::new(|_, txt, out| {
            out.clear();
            out.push(1);
            out.extend_from_slice(&lz4_flex::block::compress(txt));
        }),
        dec: Box::new(|p, n, s| {
            s.resize(n * 64, 0);
            let k = lz4_flex::block::decompress_into(&p[1..], s).unwrap();
            s[..k].iter().map(|&b| b as u32).collect()
        }),
    });
    {
        let mut c = zcomp(3, Some(&ted));
        let mut d = zdecomp(Some(&tdd));
        let mut zbuf = vec![0u8; 256 * 1024];
        arms.push(Arm {
            name: "text + zstd-3 + dict",
            text: true,
            enc: Box::new(move |_, txt, out| {
                let k = c.compress_to_buffer(txt, &mut zbuf).unwrap();
                out.clear();
                out.push(1);
                out.extend_from_slice(&zbuf[..k]);
            }),
            dec: Box::new(move |p, n, s| {
                s.clear();
                s.reserve(n * 64);
                d.decompress_to_buffer(&p[1..], s).unwrap();
                s.iter().map(|&b| b as u32).collect()
            }),
        });
    }

    let test = read_records(&format!("{dir}/test.bin"));
    let texts: Vec<Vec<u8>> = test.iter().map(|(_, ids)| text_of(ids)).collect();
    println!("{} test values; sizes include the 4-byte header + LEB128 count for token arms", test.len());
    println!(
        "{:<15} {:<22} {:>9} {:>9} {:>7} {:>7} {:>6} {:>9} {:>9} {:>9}",
        "values", "method", "B/value", "p50 B", "vs text", "vs freq", "plain", "enc ns/t", "dec ns/t", "dec µs/v"
    );

    let mut groups: Vec<(u8, usize)> = Vec::new();
    for (d, ids) in &test {
        if !groups.contains(&(*d, ids.len())) {
            groups.push((*d, ids.len()));
        }
    }
    // Per arm, per value: stored bytes, for the examples printed after the table.
    let mut sizes: Vec<Vec<usize>> = vec![vec![0; test.len()]; arms.len()];
    for &(dom, len) in &groups {
        let idx: Vec<usize> = (0..test.len()).filter(|&i| test[i].0 == dom && test[i].1.len() == len).collect();
        let tokens: usize = idx.len() * len;
        let text_bytes: usize = idx.iter().map(|&i| texts[i].len()).sum();
        let mut freq_total = 0usize;
        let label = format!("{} {}x{}", DOMAINS[dom as usize], idx.len(), len);
        for (a, arm) in arms.iter_mut().enumerate() {
            let mut payloads: Vec<Vec<u8>> = Vec::new();
            let mut scratch = Vec::new();
            for &i in &idx {
                let mut p = Vec::new();
                (arm.enc)(&test[i].1, &texts[i], &mut p);
                let back = (arm.dec)(&p, len, &mut scratch);
                if arm.text {
                    assert!(back.iter().map(|&b| b as u8).eq(texts[i].iter().copied()), "{} round trip", arm.name);
                } else {
                    assert_eq!(back, test[i].1, "{} round trip", arm.name);
                }
                sizes[a][i] = if arm.text { p.len() - 1 } else { stored(len, &p) };
                payloads.push(p);
            }
            let mut per: Vec<usize> = idx.iter().map(|&i| sizes[a][i]).collect();
            let total: usize = per.iter().sum();
            per.sort_unstable();
            if a == 0 {
                freq_total = total;
            }
            let plain = payloads.iter().filter(|p| p[0] == 0).count();

            let reps = (2_000_000 / tokens).max(2);
            let (mut be, mut bd) = (f64::MAX, f64::MAX);
            let mut out = Vec::new();
            for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..reps {
                    for &i in &idx {
                        (arm.enc)(black_box(&test[i].1), black_box(&texts[i]), &mut out);
                        black_box(&out);
                    }
                }
                be = be.min(t0.elapsed().as_secs_f64());
                let t0 = Instant::now();
                for _ in 0..reps {
                    for p in &payloads {
                        black_box((arm.dec)(black_box(p), len, &mut scratch));
                    }
                }
                bd = bd.min(t0.elapsed().as_secs_f64());
            }
            let ns = |s: f64| s * 1e9 / (reps * tokens) as f64;
            println!(
                "{:<15} {:<22} {:>9.0} {:>9} {:>6.2}x {:>6.2}x {:>6} {:>9.2} {:>9.2} {:>9.2}",
                if a == 0 { label.as_str() } else { "" },
                arm.name,
                total as f64 / idx.len() as f64,
                per[per.len() / 2],
                text_bytes as f64 / total as f64,
                freq_total as f64 / total as f64,
                if arm.text { "-".to_string() } else { plain.to_string() },
                ns(be),
                ns(bd),
                bd * 1e6 / (reps * idx.len()) as f64,
            );
        }
    }

    // Examples: in each code group, the value lz4 helps most and the one it helps least.
    let snippet = |i: usize| -> String {
        let s = String::from_utf8_lossy(&texts[i]).replace('\n', "⏎");
        s.chars().take(90).collect()
    };
    println!("\nexamples (freq -> freq + lz4 -> freq + zstd-3 + dict, stored bytes; text = UTF-8 bytes):");
    for &(dom, len) in &groups {
        if dom != 1 && dom != 3 {
            continue;
        }
        let mut idx: Vec<usize> = (0..test.len()).filter(|&i| test[i].0 == dom && test[i].1.len() == len).collect();
        idx.sort_by(|&x, &y| (sizes[1][x] as f64 / sizes[0][x] as f64).total_cmp(&(sizes[1][y] as f64 / sizes[0][y] as f64)));
        for (tag, i) in [("most", idx[0]), ("least", idx[idx.len() - 1])] {
            println!(
                "  {} {:>4} ids, lz4 helps {:<5}: text {:>5}, {:>5} -> {:>5} -> {:>5}  {:?}",
                DOMAINS[dom as usize], len, tag, texts[i].len(), sizes[0][i], sizes[1][i], sizes[4][i], snippet(i)
            );
            if dom == 3 {
                break;
            }
        }
    }
}
