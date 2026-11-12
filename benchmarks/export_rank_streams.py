#!/usr/bin/env python3
"""Export o200k token ids of real text, for core/examples/leb_vs_svb.rs.

Writes <out>/o200k.tntt (the bundled ranking, decompressed) and <out>/docs.bin: a sequence of
documents, each a u32 LE token count followed by that many u32 LE o200k ids. Documents come
in three sizes so per-value overhead shows up: short (the first 4-20 ids of a window, like a
chat turn), medium (128 ids) and long (2000 ids), from token-storage's prose, code and Hindi
test splits (stored as r50k ids, so each window is decoded and re-encoded with o200k).

Usage:
    uv run python benchmarks/export_rank_streams.py <out_dir>
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


def main() -> int:
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(ROOT, "ext/presets/o200k.tntt.zst"), "rb") as f:
        open(os.path.join(out, "o200k.tntt"), "wb").write(zstandard.ZstdDecompressor().decompress(f.read()))
    r50k, o200k = tiktoken.get_encoding("r50k_base"), tiktoken.get_encoding("o200k_base")
    rng = np.random.default_rng(0)
    docs = []
    for d in ("prose", "code", "hindi"):
        src = np.load(os.path.join(CORPUS, f"{d}_test.npy")).astype(np.int64)
        ids = o200k.encode(r50k.decode(src[:600_000].tolist()), disallowed_special=())
        for size, count in (("short", 3000), ("medium", 600), ("long", 60)):
            for _ in range(count):
                n = int(rng.integers(4, 21)) if size == "short" else (128 if size == "medium" else 2000)
                s = int(rng.integers(0, len(ids) - n))
                docs.append(ids[s:s + n])
        print(f"{d}: {len(ids)} o200k ids, {len(docs)} docs so far", flush=True)
    with open(os.path.join(out, "docs.bin"), "wb") as f:
        for doc in docs:
            f.write(struct.pack("<I", len(doc)) + np.asarray(doc, dtype="<u4").tobytes())
    print(f"wrote {len(docs)} docs to {out}/docs.bin", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
