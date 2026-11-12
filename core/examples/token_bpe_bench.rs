//! BPE over token IDs, encoded straight on `u32`s: how fast is it without a text tokenizer?
//!
//! The merge table is one trained by `benchmarks/bench_token_bpe.py` and exported with
//! `benchmarks/export_token_bpe_fixture.py`. Every chunk's output is checked against
//! HuggingFace tokenizers v1's encoding of the same chunk, so this is the same BPE, only without
//! strings: a merge is looked up by its (left, right) id pair packed into one u64.
//!
//! The algorithm is the standard one (and the one tokenizers uses): symbols in a linked list, a
//! min-heap of mergeable neighbour pairs keyed by (merge rank, position), apply until empty.
//! `freq` and `raw16` from this crate are timed on the same chunks for comparison.
//!
//!     uv run python benchmarks/export_token_bpe_fixture.py
//!     cargo run --release --example token_bpe_bench

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use pgtoken_core::{decode, encode, Codec, RankTable};

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../benchmarks/fixture/token_bpe"
);

/// Hash for a packed (left << 32 | right) pair key. A bare multiply is not enough: its low bits
/// depend only on the key's low bits (the right id), and the table picks buckets from the low
/// bits, so every pair sharing a right id would collide. Folding the high half back in fixes it.
#[derive(Default)]
struct PairHasher(u64);
impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys")
    }
    fn write_u64(&mut self, k: u64) {
        let h = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 32);
    }
}
type PairMap = HashMap<u64, (u32, u32), BuildHasherDefault<PairHasher>>;

fn read_u32s(path: &str) -> Vec<u32> {
    let b = std::fs::read(path)
        .unwrap_or_else(|e| panic!("{path}: {e} (run export_token_bpe_fixture.py)"));
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Bpe {
    base: Vec<u32>,        // gpt2 id -> symbol id
    merges: PairMap,       // (left, right) -> (rank, result)
    expand: Vec<Vec<u32>>, // symbol id -> the gpt2 ids it stands for
}

impl Bpe {
    fn load(name: &str) -> Self {
        let v = read_u32s(&format!("{DIR}/{name}.merges.u32"));
        let (n_base, n_merges) = (v[0] as usize, v[1] as usize);
        let base = v[2..2 + n_base].to_vec();
        let n_syms = 1 + *base.iter().max().unwrap() as usize + n_merges;
        let mut expand = vec![Vec::new(); n_syms];
        for (g, &s) in base.iter().enumerate() {
            expand[s as usize] = vec![g as u32];
        }
        let mut merges = PairMap::default();
        for (r, m) in v[2 + n_base..].chunks_exact(3).enumerate() {
            let (a, b, c) = (m[0], m[1], m[2]);
            merges.insert(((a as u64) << 32) | b as u64, (r as u32, c));
            let mut e = expand[a as usize].clone();
            e.extend_from_slice(&expand[b as usize]);
            expand[c as usize] = e;
        }
        Bpe {
            base,
            merges,
            expand,
        }
    }

    #[inline]
    fn lookup(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        self.merges.get(&(((a as u64) << 32) | b as u64)).copied()
    }

    #[inline]
    fn lookup_counted(&self, a: u32, b: u32, count: &mut [u64; 4]) -> Option<(u32, u32)> {
        count[0] += 1;
        self.lookup(a, b)
    }

    /// Encode gpt2 ids into symbols. `sym/prev/next/heap` are reused scratch buffers.
    fn encode(&self, ids: &[u32], s: &mut Scratch, out: &mut Vec<u32>) {
        let n = ids.len();
        s.sym.clear();
        s.prev.clear();
        s.next.clear();
        s.heap.clear();
        for (i, &g) in ids.iter().enumerate() {
            s.sym.push(self.base[g as usize]);
            s.prev.push(i as i32 - 1);
            s.next.push(if i + 1 < n { i as i32 + 1 } else { -1 });
        }
        for i in 0..n.saturating_sub(1) {
            if let Some((rank, _)) = self.lookup_counted(s.sym[i], s.sym[i + 1], &mut s.count) {
                s.count[1] += 1;
                s.heap.push(Reverse((rank, i as u32)));
            }
        }
        const DEAD: u32 = u32::MAX;
        while let Some(Reverse((rank, pos))) = s.heap.pop() {
            s.count[2] += 1;
            let p = pos as usize;
            if s.sym[p] == DEAD || s.next[p] < 0 {
                s.count[3] += 1;
                continue;
            }
            let q = s.next[p] as usize;
            match self.lookup_counted(s.sym[p], s.sym[q], &mut s.count) {
                Some((r, new)) if r == rank => {
                    s.sym[p] = new;
                    s.sym[q] = DEAD;
                    let nq = s.next[q];
                    s.next[p] = nq;
                    if nq >= 0 {
                        s.prev[nq as usize] = p as i32;
                    }
                    let pp = s.prev[p];
                    if pp >= 0 {
                        if let Some((r2, _)) =
                            self.lookup_counted(s.sym[pp as usize], new, &mut s.count)
                        {
                            s.count[1] += 1;
                            s.heap.push(Reverse((r2, pp as u32)));
                        }
                    }
                    if nq >= 0 {
                        if let Some((r2, _)) =
                            self.lookup_counted(new, s.sym[nq as usize], &mut s.count)
                        {
                            s.count[1] += 1;
                            s.heap.push(Reverse((r2, p as u32)));
                        }
                    }
                }
                _ => {
                    s.count[3] += 1; // stale entry: the pair changed since it was queued
                    continue;
                }
            }
        }
        out.clear();
        out.extend(s.sym.iter().copied().filter(|&x| x != DEAD));
    }

    fn decode(&self, syms: &[u32], out: &mut Vec<u32>) {
        out.clear();
        for &x in syms {
            out.extend_from_slice(&self.expand[x as usize]);
        }
    }
}

/// A prefix tree over every symbol's gpt2-id expansion, for encoders that need not reproduce
/// BPE exactly. Storage only needs *a* split into known symbols: any split decodes losslessly.
/// The first step is a dense array (every gpt2 id is a one-id symbol); deeper steps are hashed.
struct Trie {
    root: Vec<u32>,   // gpt2 id -> node for that one-id prefix
    child: PairMap,   // (node << 32 | gpt2 id) -> (unused, child node)
    symbol: Vec<u32>, // node -> symbol id, or NONE if that prefix is not a symbol
}

const NONE: u32 = u32::MAX;

impl Trie {
    fn build(bpe: &Bpe) -> Self {
        let mut root = vec![NONE; bpe.base.len()];
        let mut child = PairMap::default();
        let mut symbol: Vec<u32> = Vec::new();
        for (sym, exp) in bpe.expand.iter().enumerate() {
            if exp.is_empty() {
                continue;
            }
            let mut node = root[exp[0] as usize];
            if node == NONE {
                node = symbol.len() as u32;
                symbol.push(NONE);
                root[exp[0] as usize] = node;
            }
            for &g in &exp[1..] {
                let key = ((node as u64) << 32) | g as u64;
                node = match child.get(&key) {
                    Some(&(_, c)) => c,
                    None => {
                        let c = symbol.len() as u32;
                        symbol.push(NONE);
                        child.insert(key, (0, c));
                        c
                    }
                };
            }
            symbol[node as usize] = sym as u32;
        }
        Trie {
            root,
            child,
            symbol,
        }
    }

    #[inline]
    fn step(&self, node: u32, g: u32) -> Option<u32> {
        self.child
            .get(&(((node as u64) << 32) | g as u64))
            .map(|&(_, c)| c)
    }

    /// Greedy: at each position take the longest symbol that matches.
    fn greedy(&self, ids: &[u32], out: &mut Vec<u32>) {
        out.clear();
        let mut i = 0;
        while i < ids.len() {
            let mut node = self.root[ids[i] as usize];
            let (mut best, mut end) = (self.symbol[node as usize], i + 1);
            let mut j = i + 1;
            while j < ids.len() {
                match self.step(node, ids[j]) {
                    Some(c) => {
                        node = c;
                        j += 1;
                        if self.symbol[c as usize] != NONE {
                            best = self.symbol[c as usize];
                            end = j;
                        }
                    }
                    None => break,
                }
            }
            out.push(best);
            i = end;
        }
    }

    /// Fewest symbols: dynamic programming over every symbol that starts at each position.
    fn optimal(&self, ids: &[u32], cost: &mut Vec<(u32, u32, u32)>, out: &mut Vec<u32>) {
        let n = ids.len();
        cost.clear();
        cost.resize(n + 1, (u32::MAX, 0, 0)); // (symbols so far, previous position, symbol)
        cost[0].0 = 0;
        for i in 0..n {
            let base = cost[i].0;
            let mut node = self.root[ids[i] as usize];
            let mut j = i + 1;
            loop {
                let s = self.symbol[node as usize];
                if s != NONE && base + 1 < cost[j].0 {
                    cost[j] = (base + 1, i as u32, s);
                }
                if j == n {
                    break;
                }
                match self.step(node, ids[j]) {
                    Some(c) => {
                        node = c;
                        j += 1;
                    }
                    None => break,
                }
            }
        }
        out.clear();
        let mut j = n;
        while j > 0 {
            out.push(cost[j].2);
            j = cost[j].1 as usize;
        }
        out.reverse();
    }
}

#[derive(Default)]
struct Scratch {
    /// Work counters, summed over every chunk encoded (lookups, pushes, pops, stale pops).
    count: [u64; 4],
    sym: Vec<u32>,
    prev: Vec<i32>,
    next: Vec<i32>,
    heap: BinaryHeap<Reverse<(u32, u32)>>,
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

fn main() {
    for dom in ["prose", "code"] {
        let name = format!("token_bpe_{dom}_65536");
        let bpe = Bpe::load(&name);
        let test = read_u32s(&format!("{DIR}/{dom}_test.u32"));
        let chunks: Vec<&[u32]> = test.chunks_exact(512).collect();
        let reference = read_u32s(&format!("{DIR}/{name}.v1_out.u32"));

        // Correctness: identical symbols to tokenizers v1, and lossless back to gpt2 ids.
        let (mut s, mut out, mut back) = (Scratch::default(), Vec::new(), Vec::new());
        let mut at = 0usize;
        let mut n_syms = 0usize;
        for c in &chunks {
            bpe.encode(c, &mut s, &mut out);
            let len = reference[at] as usize;
            assert_eq!(
                out,
                &reference[at + 1..at + 1 + len],
                "{dom}: differs from tokenizers v1"
            );
            at += 1 + len;
            n_syms += out.len();
            bpe.decode(&out, &mut back);
            assert_eq!(&back[..], *c, "{dom}: decode is not lossless");
        }

        let per = |k: usize| s.count[k] as f64 / chunks.len() as f64;
        println!(
            "{dom} work per chunk: {:.0} pair lookups, {:.0} heap pushes, {:.0} heap pops ({:.0} stale), {:.0} merges",
            per(0), per(1), per(2), per(3), per(2) - per(3)
        );
        let lookups_only = {
            let t0 = Instant::now();
            let mut hits = 0u64;
            for c in &chunks {
                for w in c.windows(2) {
                    hits += bpe
                        .lookup(bpe.base[w[0] as usize], bpe.base[w[1] as usize])
                        .is_some() as u64;
                }
            }
            std::hint::black_box(hits);
            t0.elapsed().as_secs_f64() * 1e9 / (chunks.len() * 511) as f64
        };
        println!(
            "{dom}: one pair lookup alone costs {lookups_only:.1} ns (511 per chunk, sequential)"
        );

        // Trie encoders: lossless, and how many symbols they need against exact BPE.
        let trie = Trie::build(&bpe);
        let (mut g_out, mut o_out, mut dp) = (Vec::new(), Vec::new(), Vec::new());
        let (mut n_greedy, mut n_opt) = (0usize, 0usize);
        for c in &chunks {
            trie.greedy(c, &mut g_out);
            bpe.decode(&g_out, &mut back);
            assert_eq!(&back[..], *c, "{dom}: greedy is not lossless");
            trie.optimal(c, &mut dp, &mut o_out);
            bpe.decode(&o_out, &mut back);
            assert_eq!(&back[..], *c, "{dom}: optimal is not lossless");
            n_greedy += g_out.len();
            n_opt += o_out.len();
        }
        println!(
            "{dom}: symbols per chunk: exact BPE {:.1}, greedy longest-match {:.1}, fewest possible {:.1}",
            n_syms as f64 / chunks.len() as f64,
            n_greedy as f64 / chunks.len() as f64,
            n_opt as f64 / chunks.len() as f64
        );

        let table = RankTable::train(&test, None).unwrap();
        let mut t_greedy = Vec::new();
        let mut t_opt = Vec::new();
        let mut t_bpe = Vec::new();
        let mut t_dec = Vec::new();
        let mut t_freq = Vec::new();
        let mut t_raw = Vec::new();
        let mut t_freq_dec = Vec::new();
        for _round in 0..3 {
            for c in &chunks {
                let t0 = Instant::now();
                bpe.encode(c, &mut s, &mut out);
                t_bpe.push(t0.elapsed().as_secs_f64() * 1e6);
                let t0 = Instant::now();
                bpe.decode(&out, &mut back);
                t_dec.push(t0.elapsed().as_secs_f64() * 1e6);
                let t0 = Instant::now();
                trie.greedy(c, &mut g_out);
                t_greedy.push(t0.elapsed().as_secs_f64() * 1e6);
                let t0 = Instant::now();
                trie.optimal(c, &mut dp, &mut o_out);
                t_opt.push(t0.elapsed().as_secs_f64() * 1e6);
                let t0 = Instant::now();
                let f = encode(c, Codec::Freq, 1, Some(&table)).unwrap();
                t_freq.push(t0.elapsed().as_secs_f64() * 1e6);
                let t0 = Instant::now();
                let d = decode(&f, Some(&table)).unwrap();
                t_freq_dec.push(t0.elapsed().as_secs_f64() * 1e6);
                std::hint::black_box(d);
                let t0 = Instant::now();
                let r = encode(c, Codec::Raw16, 1, None).unwrap();
                t_raw.push(t0.elapsed().as_secs_f64() * 1e6);
                std::hint::black_box(r);
            }
        }
        println!(
            "{dom}: {} chunks of 512 ids, all identical to tokenizers v1 and lossless; \
             {:.1} symbols per chunk on average",
            chunks.len(),
            n_syms as f64 / chunks.len() as f64
        );
        println!(
            "  {:<28}{:>9}{:>9}{:>12}",
            "", "p50 µs", "p90 µs", "ns per id"
        );
        for (label, v) in [
            ("token BPE encode (Rust)", &mut t_bpe),
            ("greedy longest-match encode", &mut t_greedy),
            ("fewest-symbols encode", &mut t_opt),
            ("token BPE decode (expand)", &mut t_dec),
            ("freq encode", &mut t_freq),
            ("freq decode", &mut t_freq_dec),
            ("raw16 encode", &mut t_raw),
        ] {
            let p50 = pct(v, 0.5);
            let p90 = pct(v, 0.9);
            println!(
                "  {label:<28}{p50:>9.2}{p90:>9.2}{:>12.1}",
                p50 * 1000.0 / 512.0
            );
        }
    }
}
