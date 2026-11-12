#!/usr/bin/env python3
"""BPE over the model's token IDs: learn merges of frequent ID sequences from the corpus.

A second BPE pass on top of gpt2 ids. Each new symbol stands for a run of gpt2 ids (for
example " of" + " the"), so decoding is a table lookup that hands the model back its own ids,
with no tokenizer run. The question is whether this beats freq (frequency-ranked varints) and
zstd with a trained dictionary on size, and what it costs to encode and decode.

How: each gpt2 id is mapped to one private-use Unicode character, so HuggingFace's BPE trainer
(tokenizers 0.x; v1 dropped training) can learn merges over whole 512-token chunks with no
pre-tokenizer. New ids are stored either as raw16 (when the merged vocabulary fits in 65536) or
with freq, ranked by how often each new symbol appears in the training split.

Ratio = UTF-8 bytes / stored bytes, including pgtoken's 12-byte header, on 200 held-out test
chunks per domain. Ids are r50k (= gpt2).

Run in its own environment, since it needs the 0.x trainer:
    uv run --no-project --with 'tokenizers<0.22' --with numpy --with tiktoken \\
        --with ./py python benchmarks/bench_token_bpe.py
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time

import numpy as np
import pgtoken_rs as R
import tiktoken
from tokenizers import Tokenizer, models, trainers

CORPUS = os.path.join(os.environ.get("TOKEN_STORAGE_REPO", os.path.expanduser("~/projects/token-storage")),
                      "data", "corpus")
BASE = 0xF0000  # Supplementary Private Use Area-A: 65534 code points, room for 50257 ids
N_GPT2 = 50257


def load(name):
    return np.load(os.path.join(CORPUS, f"{name}.npy")).astype(np.int64)


def as_str(ids) -> str:
    return "".join(chr(BASE + int(i)) for i in ids)


def chunks_of(arr, n, rng):
    m = len(arr) // 512
    starts = rng.choice(m, size=min(n, m), replace=False) * 512
    return [arr[s:s + 512] for s in starts]


def rank_table(ids: np.ndarray, size: int) -> R.Table:
    counts = np.bincount(ids, minlength=size)
    seen = np.nonzero(counts)[0]
    order = seen[np.argsort(-counts[seen], kind="stable")].tolist()
    return R.Table.from_bytes(b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", len(order))
                              + struct.pack(f"<{len(order)}I", *order))


def best_of(fn, n=5):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


class TokenBPE:
    def __init__(self, train_ids: np.ndarray, vocab_size: int):
        tok = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=[chr(BASE + i) for i in range(N_GPT2)],
            limit_alphabet=N_GPT2,
            show_progress=False,
        )
        words = [as_str(train_ids[i:i + 512]) for i in range(0, len(train_ids) - 511, 512)]
        t0 = time.perf_counter()
        tok.train_from_iterator(words, trainer=trainer)
        self.train_s = time.perf_counter() - t0
        self.tok = tok
        vocab = tok.get_vocab()
        self.size = max(vocab.values()) + 1
        # Expansion table: new id -> the gpt2 ids it stands for, flat with offsets.
        exp = [None] * self.size
        for s, i in vocab.items():
            exp[i] = [ord(c) - BASE for c in s]
        self.lens = np.array([len(e) for e in exp], dtype=np.int64)
        self.offs = np.concatenate([[0], np.cumsum(self.lens)])
        self.flat = np.array([x for e in exp for x in e], dtype=np.uint32)
        self.exp = exp

    def encode(self, ids) -> np.ndarray:
        return np.asarray(self.tok.encode(as_str(ids)).ids, dtype=np.uint32)

    def decode(self, syms: np.ndarray) -> np.ndarray:
        lens = self.lens[syms]
        starts = self.offs[syms]
        out_start = np.cumsum(lens) - lens
        idx = np.repeat(starts - out_start, lens) + np.arange(lens.sum())
        return self.flat[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-chunks", type=int, default=8000)
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--vocab", type=int, nargs="+", default=[65536, 131072])
    ap.add_argument("--save-dir", default=None,
                    help="also save each trained tokenizer.json here, e.g. to time it under tokenizers v1")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_bpe_results.json"))
    args = ap.parse_args()
    enc = tiktoken.get_encoding("r50k_base")

    rng = np.random.default_rng(9012)
    tests = {d: [c.astype(np.uint32) for c in chunks_of(load(f"{d}_test"), args.chunks, rng)] for d in ("prose", "code")}
    utf8 = {d: [len(enc.decode(c.tolist()).encode()) for c in cs] for d, cs in tests.items()}
    res = {"baselines": {}, "bpe": {}, "examples": {}}

    for d, cs in tests.items():
        train = load(f"{d}_train")[: args.train_chunks * 512]
        ft = rank_table(train, N_GPT2)
        res["baselines"][d] = {
            "gpt2 raw16": float(np.median([u / len(R.encode(c, "raw16", 1)) for u, c in zip(utf8[d], cs)])),
            "gpt2 freq": float(np.median([u / len(R.encode(c, "freq", 1, ft)) for u, c in zip(utf8[d], cs)])),
        }
        print(f"{d}: gpt2 raw16 {res['baselines'][d]['gpt2 raw16']:.2f}  gpt2 freq "
              f"{res['baselines'][d]['gpt2 freq']:.2f}", flush=True)

    for d_train in ("prose", "code"):
        train = load(f"{d_train}_train")[: args.train_chunks * 512]
        for V in args.vocab:
            bpe = TokenBPE(train, V)
            train_syms = np.concatenate([bpe.encode(train[i:i + 512]) for i in range(0, len(train) - 511, 512)])
            ft = rank_table(train_syms.astype(np.int64), bpe.size)
            key = f"{d_train} merges, vocab {V}"
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                bpe.tok.save(os.path.join(args.save_dir, f"token_bpe_{d_train}_{V}.json"))
            res["bpe"][key] = {"train_seconds": bpe.train_s}
            print(f"\n{key}: trained in {bpe.train_s:.0f} s on {len(train)} gpt2 ids "
                  f"({len(train) / len(train_syms):.2f} gpt2 ids per symbol on train)", flush=True)
            for d_test, cs in tests.items():
                syms = [bpe.encode(c) for c in cs]
                for c, s in zip(cs, syms):
                    assert np.array_equal(bpe.decode(s), c), "lossless check failed"
                row = {"symbols_per_chunk_p50": float(np.median([len(s) for s in syms]))}
                if bpe.size <= 65536:
                    row["ratio raw16"] = float(np.median([u / len(R.encode(s, "raw16", 1)) for u, s in zip(utf8[d_test], syms)]))
                row["ratio freq"] = float(np.median([u / len(R.encode(s, "freq", 1, ft)) for u, s in zip(utf8[d_test], syms)]))
                row["encode_us"] = float(np.median([best_of(lambda c=c: bpe.encode(c)) for c in cs[:50]]))
                blobs = [R.encode(s, "freq", 1, ft) for s in syms[:50]]
                row["decode_us (freq + expand)"] = float(np.median([best_of(lambda b=b: bpe.decode(R.decode(b, ft).astype(np.int64))) for b in blobs]))
                res["bpe"][key][d_test] = row
                print(f"  test {d_test:<6} " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()), flush=True)
            # Concrete merges: the symbols that save the most gpt2 ids on this domain's test set.
            syms_all = np.concatenate([bpe.encode(c) for c in tests[d_train]]).astype(np.int64)
            counts = np.bincount(syms_all, minlength=bpe.size)
            saved = counts * (bpe.lens - 1)
            top = np.argsort(-saved)[:8]
            res["examples"][key] = [
                {"text": enc.decode(bpe.exp[i]), "gpt2_ids": bpe.exp[i], "uses": int(counts[i]), "ids_saved": int(saved[i])}
                for i in top
            ]
            print("  biggest savers on its own domain:")
            for e in res["examples"][key]:
                print(f"    {e['text']!r:<34} = {len(e['gpt2_ids'])} gpt2 ids, used {e['uses']}x, saves {e['ids_saved']} ids")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), **res}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
