#!/usr/bin/env python3
"""How much does `freq` lose when its ranking was trained on different data?

A vocabulary's ranking is write-once. If the data drifts away from what it was trained on, new
values still encode (unseen tokens fall back to rank k + id), just wider. This measures how much
wider, to decide whether per-partition or versioned rankings are worth building.

For each training domain (prose, code, hindi) a ranking is fit on that domain's train split,
then 512-token test chunks from every domain are encoded with it, using the extension's own
Rust codec (pgtoken_rs). raw16 (2 bytes a token, no ranking) is the floor a ranking must beat.
It also prints the tokens whose cost changes most between a matched and a mismatched ranking.

Ids are r50k, which is the gpt2 vocabulary (50257 ids).

Usage:
    uv run python benchmarks/bench_ranking_drift.py
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import pgtoken_rs as R
import tiktoken

TOKEN_STORAGE = os.environ.get("TOKEN_STORAGE_REPO", os.path.expanduser("~/projects/token-storage"))
sys.path.insert(0, TOKEN_STORAGE)
import tnbench as T  # noqa: E402

DOMAINS = ["prose", "code", "hindi"]


def ranking(ids: np.ndarray) -> list[int]:
    """Token ids by descending frequency, the order `pgtoken.train` stores."""
    counts = np.bincount(ids, minlength=50257)
    seen = np.nonzero(counts)[0]
    return seen[np.argsort(-counts[seen], kind="stable")].tolist()


def table(order: list[int]) -> R.Table:
    return R.Table.from_bytes(
        b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", len(order)) + struct.pack(f"<{len(order)}I", *order)
    )


def varint_len(rank: np.ndarray) -> np.ndarray:
    return 1 + (rank >= 256) + (rank >= 65536) + (rank >= 1 << 24)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--train-tokens", type=int, default=2_000_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ranking_drift_results.json"))
    args = ap.parse_args()
    enc = tiktoken.get_encoding("r50k_base")

    orders, tables = {}, {}
    for d in DOMAINS:
        tr = T.load_ids(f"{d}_train")[: args.train_tokens]
        orders[d] = ranking(tr)
        tables[d] = table(orders[d])
        print(f"ranking {d}: trained on {len(tr)} tokens, {len(orders[d])} distinct ids", flush=True)
    # A mixed ranking: what one shared vocabulary trained on everything would look like.
    mixed = np.concatenate([T.load_ids(f"{d}_train")[: args.train_tokens // 3] for d in DOMAINS])
    orders["mixed"] = ranking(mixed)
    tables["mixed"] = table(orders["mixed"])

    rng = np.random.default_rng(9012)
    chunks = {d: [c.astype(np.uint32) for c in T.make_chunks(T.load_ids(f"{d}_test"), 512, args.chunks, rng)]
              for d in DOMAINS}

    res = {}
    print(f"\nbytes per token, median over {args.chunks} test chunks of 512 tokens (raw16 = 2.02)")
    print(f"  {'data \\ ranking':<16}" + "".join(f"{r:>9}" for r in DOMAINS + ["mixed"]) + f"{'raw16':>9}")
    for d in DOMAINS:
        row = {}
        for r in DOMAINS + ["mixed"]:
            row[r] = float(np.median([len(R.encode(c, "freq", 1, tables[r])) / len(c) for c in chunks[d]]))
        row["raw16"] = float(np.median([len(R.encode(c, "raw16", 1)) / len(c) for c in chunks[d]]))
        res[d] = row
        print(f"  {d:<16}" + "".join(f"{row[r]:>9.3f}" for r in DOMAINS + ["mixed", "raw16"]), flush=True)

    # Concrete tokens: which ones get more expensive when code is stored with a prose ranking.
    def rank_of(order):
        m = np.full(50257, -1, dtype=np.int64)
        m[np.asarray(order)] = np.arange(len(order))
        return m

    examples = {}
    for data, wrong in (("code", "prose"), ("hindi", "prose"), ("prose", "code")):
        ids = np.concatenate(chunks[data]).astype(np.int64)
        right_r, wrong_r = rank_of(orders[data]), rank_of(orders[wrong])
        kw = len(orders[wrong])
        wr = np.where(wrong_r[ids] >= 0, wrong_r[ids], kw + ids)
        extra = varint_len(wr) - varint_len(right_r[ids])
        uniq, inv = np.unique(ids, return_inverse=True)
        cost = np.bincount(inv, weights=extra)
        top = np.argsort(-cost)[:6]
        rows = []
        for i in top:
            t = int(uniq[i])
            rows.append({"token": enc.decode([t]), "id": t, "uses": int((ids == t).sum()),
                         "rank_matched": int(right_r[t]), "rank_mismatched": int(wrong_r[t]) if wrong_r[t] >= 0 else None,
                         "extra_bytes": int(cost[i])})
        examples[f"{data}_with_{wrong}_ranking"] = {"total_extra_bytes": int(extra.sum()), "tokens": len(ids), "top": rows}
        print(f"\n{data} stored with the {wrong} ranking: +{extra.sum()} bytes over {len(ids)} tokens. Costliest tokens:")
        for x in rows:
            print(f"  {x['token']!r:<14} id {x['id']:>5}  used {x['uses']:>5}x  rank {x['rank_matched']:>5} -> "
                  f"{x['rank_mismatched'] if x['rank_mismatched'] is not None else 'unseen':>6}  +{x['extra_bytes']} B")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "bytes_per_token": res, "examples": examples}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
