#!/usr/bin/env python3
"""zstd with a trained dictionary over raw token-ID bytes: ratio vs encode and decode cost.

token-storage's 10_code_dict found this beats freq on ratio (3.15x vs 2.67x on prose, 3.27x vs
1.51x on code) and decodes in 2-5 us, but it only timed decoding, at zstd level 22. A database
pays the encode on every write, so this sweeps the level and times both directions.

Per domain: a 112 KB dictionary is trained on 512-token windows of the train split, packed as
raw16 (2 bytes a token, r50k = gpt2 ids). Then 200 held-out test chunks are compressed with it.
Times are medians over chunks, each the best of 5 warm calls, with one reused compressor (so the
dictionary is digested once, as a server-side cache would). freq and raw16 run through
pgtoken_rs, the extension's own Rust codec, for reference. Ratio = UTF-8 bytes / stored bytes,
including pgtoken's 12-byte header for freq and raw16 (zstd's frame header is in its own size).

Usage:
    uv run python benchmarks/bench_zstd_dict.py
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
import zstandard as zstd

import pgcommon as C  # noqa: F401  (puts token-storage on sys.path)
import tnbench as T

LEVELS = [1, 3, 9, 19, 22]


def best_of(fn, n=5):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def rank_table(ids: np.ndarray) -> R.Table:
    counts = np.bincount(ids, minlength=50257)
    seen = np.nonzero(counts)[0]
    order = seen[np.argsort(-counts[seen], kind="stable")].tolist()
    return R.Table.from_bytes(b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", len(order))
                              + struct.pack(f"<{len(order)}I", *order))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--dict-kb", type=int, default=112)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "zstd_dict_results.json"))
    args = ap.parse_args()
    enc = tiktoken.get_encoding("r50k_base")
    res = {}

    for dom in ("prose", "code"):
        train = T.load_ids(f"{dom}_train")
        samples = [train[i:i + 512].astype("<u2").tobytes() for i in range(0, min(len(train), 15000 * 512), 512)]
        zdict = zstd.train_dictionary(args.dict_kb * 1024, samples)
        tbl = rank_table(train[:2_000_000])
        rng = np.random.default_rng(9012)
        chunks = [c.astype(np.uint32) for c in T.make_chunks(T.load_ids(f"{dom}_test"), 512, args.chunks, rng)]
        utf8 = [len(enc.decode(c.tolist()).encode()) for c in chunks]
        raw = [c.astype("<u2").tobytes() for c in chunks]
        print(f"\n{dom}: dictionary {len(zdict.as_bytes()) // 1024} KB, {len(chunks)} test chunks of 512 tokens")
        print(f"  {'method':<22}{'ratio':>7}{'encode us':>11}{'decode us':>11}")
        rows = {}

        def report(name, sizes, e_us, d_us):
            r = float(np.median([u / s for u, s in zip(utf8, sizes)]))
            rows[name] = {"ratio": r, "encode_us": float(np.median(e_us)), "decode_us": float(np.median(d_us))}
            print(f"  {name:<22}{r:>7.2f}{np.median(e_us):>11.1f}{np.median(d_us):>11.1f}", flush=True)

        for codec, t in (("raw16", None), ("freq", tbl)):
            blobs = [R.encode(c, codec, 1, t) for c in chunks]
            report(f"pgtoken {codec}", [len(b) for b in blobs],
                   [best_of(lambda c=c: R.encode(c, codec, 1, t)) for c in chunks],
                   [best_of(lambda b=b: R.decode(b, t)) for b in blobs])

        zd = zstd.ZstdDecompressor(dict_data=zdict)
        for lvl in LEVELS:
            zc = zstd.ZstdCompressor(level=lvl, dict_data=zdict, write_content_size=True,
                                     write_checksum=False, write_dict_id=True)
            blobs = [zc.compress(b) for b in raw]
            for b, c in zip(blobs, chunks):  # lossless check, and decode back to ids
                assert np.array_equal(np.frombuffer(zd.decompress(b), "<u2"), c)
            report(f"zstd-{lvl} + dict", [len(b) for b in blobs],
                   [best_of(lambda b=b: zc.compress(b)) for b in raw],
                   [best_of(lambda b=b: np.frombuffer(zd.decompress(b), "<u2")) for b in blobs])
        res[dom] = rows

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": res}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
