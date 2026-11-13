#!/usr/bin/env python3
"""Build the bundled o200k preset: a frozen frequency ranking and the id -> bytes mapping.

Ranking: o200k ids counted over an equal mix of prose, code and Hindi (token-storage's train
splits, stored there as r50k ids, so each 4096-id window is decoded to text and re-encoded with
o200k). Ids ordered by count, ties by id; ids never seen are left out and the `freq` codec gives
them rank k + id. Mapping: tiktoken's own bytes for every id.

Both are written in pgtoken's artefact formats (core/src/tables.rs), zstd-19 compressed, to
ext/presets/. A bundled ranking is frozen forever: rows encoded with it must decode the same in
every later release. A better ranking ships under a new name.

Also prints bytes per token on each domain's held-out test split: freq with this ranking against
raw24 (o200k ids need 3 bytes stored plain) and against a ranking matched to that domain.

Usage:
    uv run python scripts/build_o200k_preset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct

import numpy as np
import pgtoken_rs as R
import tiktoken
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(os.environ.get("TOKEN_STORAGE_REPO", os.path.expanduser("~/projects/token-storage")),
                      "data", "corpus")
OUT = os.path.join(ROOT, "ext", "presets")
DOMAINS = ("prose", "code", "hindi")


def to_o200k(r50k_ids: np.ndarray, src, dst) -> np.ndarray:
    out = []
    for i in range(0, len(r50k_ids), 4096):
        out.extend(dst.encode(src.decode(r50k_ids[i:i + 4096].tolist()), disallowed_special=()))
    return np.asarray(out, dtype=np.int64)


def ranking(ids: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(ids, minlength=n)
    seen = np.nonzero(counts)[0]
    return seen[np.argsort(-counts[seen], kind="stable")].astype("<u4")


def tntt(order: np.ndarray) -> bytes:
    return b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", len(order)) + order.tobytes()


def tnmap(enc) -> bytes:
    toks = []
    for i in range(enc.n_vocab):
        try:
            toks.append(enc.decode_single_token_bytes(i))
        except KeyError:
            toks.append(b"")
    offs = np.cumsum([0] + [len(t) for t in toks]).astype("<u4")
    return b"TNTT" + bytes([1, 2, 0, 0]) + struct.pack("<I", enc.n_vocab) + offs.tobytes() + b"".join(toks)


def bytes_per_token(chunks, codec, table=None) -> float:
    return sum(len(R.encode(c, codec, 1, table)) for c in chunks) / sum(len(c) for c in chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r50k-tokens", type=int, default=4_000_000, help="per domain, read from each train split")
    ap.add_argument("--test-chunks", type=int, default=300)
    args = ap.parse_args()
    src, dst = tiktoken.get_encoding("r50k_base"), tiktoken.get_encoding("o200k_base")
    n = dst.n_vocab

    train = {}
    for d in DOMAINS:
        raw = np.load(os.path.join(CORPUS, f"{d}_train.npy")).astype(np.int64)[: args.r50k_tokens]
        train[d] = to_o200k(raw, src, dst)
        print(f"{d}: {len(raw)} r50k ids -> {len(train[d])} o200k ids", flush=True)
    per = min(len(t) for t in train.values())
    mixed = np.concatenate([t[:per] for t in train.values()])
    order = ranking(mixed, n)
    print(f"mixed: {per} o200k ids per domain, {len(order)} of {n} ids seen and ranked", flush=True)
    print("top 12 ranks: " + "  ".join(repr(dst.decode([int(i)])) for i in order[:12]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    blobs = {"o200k.tntt": tntt(order), "o200k.tnmap": tnmap(dst)}
    meta = {"tokenizer": "o200k_base", "vocab_size": n, "ranked": int(len(order)),
            "training": {d: per for d in DOMAINS}, "source": "token-storage data/corpus *_train.npy",
            "files": {}}
    for name, blob in blobs.items():
        z = zstd.ZstdCompressor(level=19).compress(blob)
        with open(os.path.join(OUT, name + ".zst"), "wb") as f:
            f.write(z)
        meta["files"][name] = {"bytes": len(blob), "zst_bytes": len(z), "sha256": hashlib.sha256(blob).hexdigest()}
        print(f"wrote ext/presets/{name}.zst: {len(blob) / 1024:.0f} KB -> {len(z) / 1024:.0f} KB", flush=True)
    with open(os.path.join(OUT, "o200k.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    mixed_tbl = R.Table.from_bytes(blobs["o200k.tntt"])
    rng = np.random.default_rng(7)
    print(f"\nbytes per token on held-out test text ({args.test_chunks} chunks of 512 o200k ids):", flush=True)
    print(f"  {'domain':<8}{'raw24':>8}{'freq, this':>12}{'freq, matched':>15}", flush=True)
    for d in DOMAINS:
        test = to_o200k(np.load(os.path.join(CORPUS, f"{d}_test.npy")).astype(np.int64)[:1_000_000], src, dst)
        m = len(test) // 512
        chunks = [test[s:s + 512].astype(np.uint32) for s in rng.choice(m, min(args.test_chunks, m), replace=False) * 512]
        matched = R.Table.from_bytes(tntt(ranking(train[d], n)))
        print(f"  {d:<8}{bytes_per_token(chunks, 'raw24'):>8.3f}{bytes_per_token(chunks, 'freq', mixed_tbl):>12.3f}"
              f"{bytes_per_token(chunks, 'freq', matched):>15.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
