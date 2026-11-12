#!/usr/bin/env python3
"""How big would built-in tokenizer presets be? Ranking + id->bytes mapping, raw and compressed.

The ranking is a permutation of ids (most frequent first). A real one comes from a corpus; here a
seeded random permutation stands in, which is the worst case for compression and close to a real
one (a frequency order has no structure a compressor can use beyond the ids' bit width).
The mapping is the tokenizer's own bytes per id, as pgtoken's .tnmap stores them: a u32 offset per
id plus the concatenated token bytes.

Usage:
    uv run python scripts/estimate_preset_sizes.py
"""

from __future__ import annotations

import gzip
import struct

import numpy as np
import tiktoken
import zstandard as zstd

ENCODINGS = ["r50k_base", "cl100k_base", "o200k_base"]


def sizes(blob: bytes) -> tuple[int, int, int]:
    return len(blob), len(gzip.compress(blob, 9)), len(zstd.ZstdCompressor(level=19).compress(blob))


def main() -> int:
    print(f"{'tokenizer':<12}{'ids':>8}  {'part':<9}{'raw KB':>9}{'gzip-9 KB':>11}{'zstd-19 KB':>12}", flush=True)
    total = np.zeros(3)
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        n = enc.n_vocab
        toks = []
        for i in range(n):
            try:
                toks.append(enc.decode_single_token_bytes(i))
            except KeyError:  # gaps in the id space (e.g. before special tokens)
                toks.append(b"")
        offs = np.cumsum([0] + [len(t) for t in toks]).astype("<u4")
        mapping = offs.tobytes() + b"".join(toks)
        order = np.random.default_rng(0).permutation(n).astype("<u4")
        ranking = b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", n) + order.tobytes()
        for part, blob in (("ranking", ranking), ("mapping", mapping)):
            r, g, z = sizes(blob)
            total += (r, g, z)
            print(f"{name:<12}{n:>8}  {part:<9}{r / 1024:>9.0f}{g / 1024:>11.0f}{z / 1024:>12.0f}", flush=True)
        print(f"{'':<12}{'':>8}  e.g. id 1000 = {toks[1000]!r}, longest token {max(len(t) for t in toks)} bytes",
              flush=True)
    print(f"{'all three':<20}  {'both':<9}{total[0] / 1024:>9.0f}{total[1] / 1024:>11.0f}{total[2] / 1024:>12.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
