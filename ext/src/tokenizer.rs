//! Server-side tokenizers: turning text into a vocabulary's token ids inside PostgreSQL.
//!
//! A vocabulary may or may not have one. [`crate::registry::tokenizer`] is the single lookup the
//! write paths ask, by vocabulary id: text input, the `text -> tokens` cast, comparing a column
//! with a text literal, and `pgtoken.tokenize`. None of them knows which tokenizer it gets. Today
//! the only implementation is the o200k preset's; a later one (a Hugging Face `tokenizer.json`
//! stored as an artefact function, like a ranking) plugs into the same lookup.
//!
//! Token ids remain what a column stores and what `pg_dump` writes, so a dump never needs a
//! tokenizer to restore.

use fancy_regex::Regex;
use rustc_hash::FxHashMap;

use pgtoken_core::tables::ByteMap;

/// Text to token ids, under one vocabulary.
pub trait Tokenizer {
    /// Tokenize `text`. Special tokens are not recognised: `<|endoftext|>` in the text is plain
    /// text, as in tiktoken's `encode_ordinary`.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String>;
}

/// A tiktoken-style byte-level BPE: a pre-tokenizer regex splits the text into pieces, and each
/// piece is merged pair by pair, lowest rank first.
///
/// Built from the vocabulary's own `token_id -> bytes` mapping, because for tiktoken encodings a
/// token's id *is* its merge rank. So the preset ships no second copy of its vocabulary: the
/// mapping `pgtoken.text` already uses is the whole merge table.
pub struct TiktokenBpe {
    ranks: FxHashMap<Vec<u8>, u32>,
    pattern: Regex,
}

impl TiktokenBpe {
    /// `mergeable` bounds the ids that take part in merges: special tokens sit above it and are
    /// left out, so text can never turn into one.
    pub fn from_mapping(map: &ByteMap, mergeable: u32, pattern: &str) -> Result<Self, String> {
        let mut ranks = FxHashMap::default();
        ranks.reserve(mergeable as usize);
        for id in 0..mergeable {
            if let Some(bytes) = map.get(id) {
                ranks.insert(bytes.to_vec(), id);
            }
        }
        let pattern = Regex::new(pattern).map_err(|e| format!("bad pre-tokenizer regex: {e}"))?;
        Ok(TiktokenBpe { ranks, pattern })
    }

    fn rank(&self, bytes: &[u8]) -> u32 {
        self.ranks.get(bytes).copied().unwrap_or(u32::MAX)
    }

    /// One pre-tokenized piece to ids, with the same result as tiktoken's `byte_pair_encode`:
    /// the pair with the lowest rank merges first, and of equal ranks the leftmost wins.
    ///
    /// The parts form a linked list and the candidate pairs sit in a min-heap keyed by
    /// (rank, start offset), so a piece of n bytes costs O(n log n). A plain rescan after every
    /// merge costs O(n²), which made one long run such as a million `!` take minutes. Heap
    /// entries go stale when a neighbour merges. They are skipped on pop when the part is gone or
    /// its pair rank has changed.
    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) -> Result<(), String> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        if let Some(&id) = self.ranks.get(piece) {
            out.push(id);
            return Ok(());
        }
        let n = piece.len();
        if n == 0 {
            return Ok(());
        }
        // Part i starts at byte i. `next[i]` is the start of the part after it (n at the end),
        // `prev[i]` the start of the part before it. `rank[i]` is the rank of merging part i
        // with the part after it, u32::MAX when there is no such token. `alive[i]` is false
        // once part i has been merged into the part before it.
        let mut next: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive = vec![true; n];
        let mut rank = vec![u32::MAX; n];
        let mut heap = BinaryHeap::with_capacity(n);
        for i in 0..n - 1 {
            rank[i] = self.rank(&piece[i..i + 2]);
            if rank[i] != u32::MAX {
                heap.push(Reverse((rank[i], i)));
            }
        }
        let pair_rank = |next: &[usize], i: usize| {
            let j = next[i];
            if j < n {
                self.rank(&piece[i..next[j]])
            } else {
                u32::MAX
            }
        };
        let mut merges: u32 = 0;
        while let Some(Reverse((r, i))) = heap.pop() {
            if !alive[i] || rank[i] != r {
                continue;
            }
            merges = merges.wrapping_add(1);
            if merges.is_multiple_of(4096) {
                interrupt_point();
            }
            // Merge part i with the part after it.
            let j = next[i];
            alive[j] = false;
            next[i] = next[j];
            if next[i] < n {
                prev[next[i]] = i;
            }
            rank[i] = pair_rank(&next, i);
            if rank[i] != u32::MAX {
                heap.push(Reverse((rank[i], i)));
            }
            let p = prev[i];
            if p != usize::MAX {
                rank[p] = pair_rank(&next, p);
                if rank[p] != u32::MAX {
                    heap.push(Reverse((rank[p], p)));
                }
            }
        }
        let mut i = 0;
        while i < n {
            let bytes = &piece[i..next[i]];
            match self.ranks.get(bytes) {
                Some(&id) => out.push(id),
                None => return Err(format!("no token for bytes {bytes:?}")),
            }
            i = next[i];
        }
        Ok(())
    }
}

impl Tokenizer for TiktokenBpe {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut out = Vec::with_capacity(text.len() / 4 + 1);
        for m in self.pattern.find_iter(text) {
            let m = m.map_err(|e| format!("pre-tokenizer failed: {e}"))?;
            interrupt_point();
            self.encode_piece(m.as_str().as_bytes(), &mut out)?;
        }
        Ok(out)
    }
}

/// Let PostgreSQL stop a long tokenize: `statement_timeout`, `pg_cancel_backend` and Ctrl-C all
/// land here. Plain `cargo test` binaries run outside a server and have no interrupt flag to read.
#[inline]
fn interrupt_point() {
    #[cfg(not(test))]
    pgrx::check_for_interrupts!();
}

/// The pre-tokenizer of tiktoken's `o200k_base`, verbatim (`tiktoken_ext/openai_public.py`).
pub const O200K_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

#[cfg(test)]
mod unit_tests {
    use crate::preset::O200K;

    fn o200k(text: &str) -> Vec<u32> {
        O200K.tokenizer().unwrap().tokenize(text).unwrap()
    }

    #[test]
    fn matches_tiktoken_on_known_strings() {
        // Expected ids from tiktoken 0.13 `o200k_base`, `encode_ordinary`.
        assert_eq!(o200k("Hello, world!"), [13225, 11, 2375, 0]);
        assert_eq!(o200k("🦜"), [4103, 99, 250]);
        assert_eq!(o200k(""), Vec::<u32>::new());
    }

    /// The old O(n²) merge, kept as the reference: tiktoken's `byte_pair_encode` step for step.
    fn reference_merge(bpe: &super::TiktokenBpe, piece: &[u8]) -> Vec<u32> {
        let rank = |b: &[u8]| bpe.ranks.get(b).copied().unwrap_or(u32::MAX);
        if let Some(&id) = bpe.ranks.get(piece) {
            return vec![id];
        }
        let mut parts: Vec<(usize, u32)> = (0..piece.len() - 1)
            .map(|i| (i, rank(&piece[i..i + 2])))
            .collect();
        parts.push((piece.len() - 1, u32::MAX));
        parts.push((piece.len(), u32::MAX));
        let pair = |parts: &[(usize, u32)], i: usize| {
            if i + 3 < parts.len() {
                rank(&piece[parts[i].0..parts[i + 3].0])
            } else {
                u32::MAX
            }
        };
        loop {
            let (r, i) = parts[..parts.len() - 1]
                .iter()
                .enumerate()
                .map(|(j, &(_, r))| (r, j))
                .min()
                .unwrap();
            if r == u32::MAX {
                break;
            }
            if i > 0 {
                parts[i - 1].1 = pair(&parts, i - 1);
            }
            parts[i].1 = pair(&parts, i);
            parts.remove(i + 1);
        }
        parts
            .windows(2)
            .map(|w| bpe.ranks[&piece[w[0].0..w[1].0]])
            .collect()
    }

    #[test]
    fn heap_merge_matches_the_reference_merge() {
        let spec = O200K.tokenizer.as_ref().unwrap();
        let bpe = &super::TiktokenBpe::from_mapping(&O200K.mapping(), spec.mergeable, spec.pattern)
            .unwrap();
        // Repeated runs, where ties between equal ranks decide the result, plus mixed text.
        let mut pieces: Vec<Vec<u8>> = ["!", "!!", "=", "-", " ", "ab", "\u{e9}", "🦜", "aaab"]
            .iter()
            .flat_map(|u| (1..300).map(move |k| u.repeat(k).into_bytes()))
            .collect();
        // Deterministic pseudo-random bytes from a small alphabet, so many merges apply.
        let alphabet = b"abcde !?.-_=\n\t()";
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for len in 1..400 {
            let mut p = Vec::with_capacity(len);
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                p.push(alphabet[(x % alphabet.len() as u64) as usize]);
            }
            pieces.push(p);
        }
        for p in &pieces {
            let mut got = Vec::new();
            bpe.encode_piece(p, &mut got).unwrap();
            assert_eq!(
                got,
                reference_merge(bpe, p),
                "piece {:?}",
                String::from_utf8_lossy(p)
            );
        }
    }

    #[test]
    fn a_long_run_of_one_character_is_fast() {
        // One pre-tokenizer piece of 500,000 bytes (fancy-regex refuses a million). In a debug
        // build the old O(n²) merge took 82 s on 100,000 bytes and the heap merge 0.17 s.
        let text = "!".repeat(500_000);
        let t = std::time::Instant::now();
        let ids = o200k(&text);
        let took = t.elapsed();
        assert!(!ids.is_empty());
        assert!(took.as_secs() < 10, "took {took:?}");
    }

    #[test]
    fn special_tokens_are_plain_text() {
        let ids = o200k("<|endoftext|>");
        assert!(!ids.contains(&199_999), "{ids:?}");
        assert!(ids.len() > 1);
    }
}
