#!/usr/bin/env python3
"""Export real o200k values for core/examples/lz_on_freq.rs (does LZ compression on top of
`freq` pay off, and at what speed).

Writes to <out>:
  o200k.tntt, o200k.tnmap   the bundled ranking and mapping, decompressed
  test.bin                  values to measure, from token-storage's held-out *_test splits
  train.bin                 512-id windows from the *_train splits, for training dictionaries

Each record: u8 domain (0 prose, 1 code, 2 hindi, 3 one repeated code line), u32 LE id count, that many u32 LE o200k ids.
test.bin holds, per domain, 400 values of 128 ids, 300 of 512 and 60 of 2000, each a random
window of the text. token-storage stores r50k ids, so text is decoded and re-encoded with o200k.

Usage:
    uv run python benchmarks/export_codec_corpus.py <out_dir>
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np
import tiktoken
import zstandard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.expanduser("~/projects/token-storage/data/corpus")
DOMAINS = ("prose", "code", "hindi")
TEST_SIZES = ((128, 400), (512, 300), (2000, 60))
TRAIN_WINDOWS = 1500  # per domain, 512 ids each


def to_o200k(r50k_ids, src, dst):
    out = []
    for i in range(0, len(r50k_ids), 4096):
        out.extend(dst.encode(src.decode(r50k_ids[i:i + 4096].tolist()), disallowed_special=()))
    return out


def write(path, records):
    with open(path, "wb") as f:
        for dom, ids in records:
            f.write(struct.pack("<BI", dom, len(ids)) + np.asarray(ids, dtype="<u4").tobytes())


def main() -> int:
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name in ("o200k.tntt", "o200k.tnmap"):
        with open(os.path.join(ROOT, "ext/presets", name + ".zst"), "rb") as f:
            open(os.path.join(out, name), "wb").write(zstandard.ZstdDecompressor().decompress(f.read()))
    r50k, o200k = tiktoken.get_encoding("r50k_base"), tiktoken.get_encoding("o200k_base")
    rng = np.random.default_rng(0)
    test, train = [], []
    for dom, d in enumerate(DOMAINS):
        ids = to_o200k(np.load(os.path.join(CORPUS, f"{d}_test.npy")).astype(np.int64)[:800_000], r50k, o200k)
        for n, count in TEST_SIZES:
            for _ in range(count):
                s = int(rng.integers(0, len(ids) - n))
                test.append((dom, ids[s:s + n]))
        tr = to_o200k(np.load(os.path.join(CORPUS, f"{d}_train.npy")).astype(np.int64)[:2_000_000], r50k, o200k)
        for _ in range(TRAIN_WINDOWS):
            s = int(rng.integers(0, len(tr) - 512))
            train.append((dom, tr[s:s + 512]))
        print(f"{d}: {len(ids)} test ids, {len(tr)} train ids", flush=True)
    # Domain 3: one line of real code repeated, the case where the README benchmark saw tokens
    # lose most (1466 ids: 3121 B as tokens against 871 B as pglz-compressed text).
    line = 'CThostFtdcLimitPosiSField["TotalVolume"] = "int"\n'
    rep = o200k.encode(line * 200, disallowed_special=())[:1466]
    test.append((3, rep))
    write(os.path.join(out, "test.bin"), test)
    write(os.path.join(out, "train.bin"), train)
    print(f"wrote {len(test)} test values and {len(train)} train windows to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
