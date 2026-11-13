#!/usr/bin/env python3
"""Tokenizer speed, measured fairly: same text, one thread, warmed up, best of 3.

  o200k        tiktoken
  bpe          Hugging Face tokenizers, the exp138 BPE model (same pretokenizer as ISBPE)
  bpe batch    the same, encode_batch (Hugging Face runs this on all cores)
  isbpe        isbpe_rs, exp138 dual model

Two shapes of input: 4 KB documents (like rows in a table) and 1 MB pieces.

Usage: same `uv run --with ...` line as isbpe_storage.py.
"""

from __future__ import annotations

import os
import time

import isbpe_rs
import tiktoken
import tokenizers
from tokenizers import Tokenizer

import isbpe_storage as S

TEXT_BYTES = 4_000_000


def best_of(fn, docs, reps=3):
    fn(docs[:20])  # warm up
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        n = fn(docs)
        best = min(best, time.perf_counter() - t0)
    return best, n


def main() -> int:
    print(f"tokenizers {tokenizers.__version__}, tiktoken {tiktoken.__version__}, "
          f"isbpe_rs {isbpe_rs.__version__}, {os.cpu_count()} cpus", flush=True)
    for dom, prefix in S.DOMAINS.items():
        text = S.utf8_prefix(open(f"{S.RESEARCH}/data/{prefix}_tr.txt", "rb").read(), TEXT_BYTES)
        o200k = tiktoken.get_encoding("o200k_base")
        bpe = Tokenizer.from_file(f"{S.RESEARCH}/models/exp138/bpe_{dom}_same_65536.json")
        isb = isbpe_rs.ISBPE.load(f"{S.RESEARCH}/models/exp138/isbpe_{dom}_dual_65536.json")
        arms = {
            "o200k": lambda d: sum(len(o200k.encode_ordinary(x)) for x in d),
            "bpe": lambda d: sum(len(bpe.encode(x, add_special_tokens=False).ids) for x in d),
            "bpe batch": lambda d: sum(len(e.ids) for e in
                                       bpe.encode_batch(d, add_special_tokens=False)),
            "isbpe": lambda d: sum(len(isb.encode(x)) for x in d),
        }
        for size in (4096, 1 << 20):
            docs = [c.decode("utf-8") for c in S.chunks(text, size, 10**9)]
            mb = sum(len(c.encode()) for c in docs) / 1e6
            for name, fn in arms.items():
                t, n = best_of(fn, docs)
                print(f"[{dom}] {size:>7} B docs  {name:10s} {mb / t:7.1f} MB/s  "
                      f"({n:,} tokens)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
