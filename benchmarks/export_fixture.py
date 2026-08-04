#!/usr/bin/env python3
"""Export a cross-language validation fixture for the Rust codec port.

The Rust port in ../core is only trustworthy if it reproduces the Python harness's
compression ratios on the real corpora. To make that comparison exact rather than
statistical, this script pins down everything both sides could otherwise disagree about:

  * the coding tables, written in the Rust side's own on-disk TNTT format so both
    languages load byte-identical tables (this also validates the format from both ends);
  * the test chunk texts, so both measure the same documents;
  * Python's own encoded sizes per codec, so the Rust test has a target to hit.

Protocol matches 07_kalcher_baseline/bench_kalcher_table1_matched.py: seed 9012, chunks of
512 *r50k* tokens sampled from each domain's held-out test split, decoded to text, then
re-encoded with the target tokenizer. Tables train on the train split only.

Tie-breaking note: the harness's `build_rank_table` uses `np.argsort(-counts)`, whose
default quicksort is unstable, so its zero-count tail order is arbitrary. Here we use
lexsort to break ties by ascending token ID, matching the Rust implementation, which lets
the two sides produce byte-identical rank tables instead of merely equivalent ones.

Usage:
    uv run python 12_postgres/harness/export_fixture.py [--out DIR] [--chunks N]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import tiktoken

from pgcommon import TOKEN_STORAGE  # noqa: E402  — also checks the repo is reachable

sys.path.insert(0, TOKEN_STORAGE)

import tnbench as T  # noqa: E402

DOMAINS = ["prose", "code", "hindi"]

# Tokenizer id and vocab must match core/src/header.rs `Tokenizer`.
TOKENIZERS = {
    "r50k": (1, "r50k_base", 50_257),
    "cl100k": (2, "cl100k_base", 100_277),
    "o200k": (3, "o200k_base", 200_019),
}

TABLE_MAGIC = b"TNTT"
TABLE_VERSION = 1
KIND_RANK = 1
KIND_ANS = 2

CHUNK_R50K_TOKENS = 512
SEED = 9012


def tntt_bytes(kind: int, tok_id: int, vocab: int, payload: np.ndarray) -> bytes:
    """Serialize a table in the Rust side's on-disk format (see core/src/tables.rs)."""
    vals = np.asarray(payload, dtype=np.uint32)
    assert len(vals) == vocab, f"payload has {len(vals)} entries, expected {vocab}"
    head = TABLE_MAGIC + struct.pack("<BBBBII", TABLE_VERSION, kind, tok_id, 0, vocab, vocab)
    assert len(head) == 16, len(head)
    return head + vals.tobytes()


def rank_table(ids: np.ndarray, vocab: int) -> np.ndarray:
    """token_of_rank, descending by count then ascending by token id.

    Deterministic across numpy versions, unlike the harness's `argsort(-counts)`.
    """
    counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab)[:vocab]
    # lexsort's last key is primary: sort by -counts, tie-break on ascending token id.
    return np.lexsort((np.arange(vocab, dtype=np.int64), -counts.astype(np.int64)))


def ans_counts(ids: np.ndarray, vocab: int) -> np.ndarray:
    """Laplace +1 counts, matching tnbench.build_ans_model."""
    counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab)[:vocab]
    return counts.astype(np.int64) + 1


def pack_raw16(ids: np.ndarray) -> bytes:
    return np.asarray(ids, dtype=np.uint16).tobytes()


def pack_raw24(ids: np.ndarray) -> bytes:
    return T.pack3(np.asarray(ids, dtype=np.int64))


def encode_freq(ids: np.ndarray, rank_of: np.ndarray) -> bytes:
    """Rank remap + streamvbyte, via pyfastpfor.

    The container layout differs from the Rust `stream-vbyte` crate's, so sizes are
    compared with a tolerance rather than for equality. That is expected and documented.
    """
    return T.svb_encode_arr(rank_of[np.asarray(ids, dtype=np.int64)])


def encode_ans(ids: np.ndarray, counts: np.ndarray) -> bytes:
    import constriction

    probs = counts.astype(np.float64) / counts.sum()
    model = constriction.stream.model.Categorical(probs, perfect=False)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(np.asarray(ids, dtype=np.int32), model)
    return coder.get_compressed().tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture"),
        help="output directory for the fixture",
    )
    ap.add_argument("--chunks", type=int, default=40, help="test chunks per domain")
    ap.add_argument(
        "--tokenizers",
        default="r50k,cl100k,o200k",
        help="comma-separated subset of r50k,cl100k,o200k",
    )
    args = ap.parse_args()

    toks = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
    for t in toks:
        if t not in TOKENIZERS:
            ap.error(f"unknown tokenizer {t!r}")

    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "payloads"), exist_ok=True)

    r50k = tiktoken.get_encoding("r50k_base")
    manifest: dict = {
        "protocol": {
            "seed": SEED,
            "chunk_r50k_tokens": CHUNK_R50K_TOKENS,
            "n_chunks": args.chunks,
            "note": (
                "Chunks are 512 r50k tokens sampled from the held-out test split, decoded "
                "to text, then re-encoded with the target tokenizer. Tables train on the "
                "train split only."
            ),
        },
        "cases": [],
    }

    for domain in DOMAINS:
        # One fresh RNG per domain so the draw does not depend on domain ordering.
        rng = np.random.default_rng(SEED)
        test_r50k = T.load_ids(f"{domain}_test")
        sampled = T.make_chunks(test_r50k, CHUNK_R50K_TOKENS, args.chunks, rng)
        texts = [r50k.decode(c.tolist()) for c in sampled]

        chunk_path = os.path.join(args.out, "chunks", f"{domain}.jsonl")
        with open(chunk_path, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        raw_utf8 = [len(t.encode("utf-8")) for t in texts]
        print(f"[{domain}] {len(texts)} chunks, median raw UTF-8 {int(np.median(raw_utf8))} B")

        train_text = r50k.decode(T.load_ids(f"{domain}_train").tolist())

        for tname in toks:
            tok_id, enc_name, vocab = TOKENIZERS[tname]
            enc = tiktoken.get_encoding(enc_name)

            train_ids = np.array(enc.encode(train_text, disallowed_special=()), dtype=np.int64)
            train_ids = train_ids[train_ids < vocab]

            tor = rank_table(train_ids, vocab)
            rank_of = np.empty(vocab, dtype=np.int64)
            rank_of[tor] = np.arange(vocab, dtype=np.int64)
            counts = ans_counts(train_ids, vocab)

            rank_file = os.path.join(args.out, "tables", f"{domain}_{tname}_rank.tntt")
            ans_file = os.path.join(args.out, "tables", f"{domain}_{tname}_ans.tntt")
            with open(rank_file, "wb") as f:
                f.write(tntt_bytes(KIND_RANK, tok_id, vocab, tor))
            with open(ans_file, "wb") as f:
                f.write(tntt_bytes(KIND_ANS, tok_id, vocab, counts))

            sizes = {"raw16": [], "raw24": [], "freq": [], "ans": []}
            blobs: dict[str, list[bytes]] = {k: [] for k in sizes}
            n_tokens = []
            for text in texts:
                ids = np.array(enc.encode(text, disallowed_special=()), dtype=np.int64)
                assert ids.max(initial=0) < vocab, "token id past declared vocab"
                n_tokens.append(len(ids))
                if vocab <= 65_536:
                    blobs["raw16"].append(pack_raw16(ids))
                blobs["raw24"].append(pack_raw24(ids))
                blobs["freq"].append(encode_freq(ids, rank_of))
                blobs["ans"].append(encode_ans(ids, counts))
            for k, v in blobs.items():
                sizes[k] = [len(b) for b in v]

            # Write the actual payload bytes so the Rust side can decode Python's output,
            # not merely compare lengths against it. Format: u32 LE count, then per record
            # a u32 LE length followed by that many bytes.
            payload_files = {}
            for k, recs in blobs.items():
                if not recs:
                    continue
                path = os.path.join(args.out, "payloads", f"{domain}_{tname}_{k}.bin")
                with open(path, "wb") as f:
                    f.write(struct.pack("<I", len(recs)))
                    for b in recs:
                        f.write(struct.pack("<I", len(b)))
                        f.write(b)
                payload_files[k] = os.path.relpath(path, args.out)

            case = {
                "domain": domain,
                "tokenizer": tname,
                "tokenizer_id": tok_id,
                "vocab": vocab,
                "rank_table": os.path.relpath(rank_file, args.out),
                "ans_table": os.path.relpath(ans_file, args.out),
                "chunks": os.path.relpath(chunk_path, args.out),
                "raw_utf8_bytes": raw_utf8,
                "n_tokens": n_tokens,
                "python_payload_files": payload_files,
                # Python payload sizes, excluding the 12-byte value header the Rust side
                # adds. The Rust test must subtract the header before comparing.
                "python_payload_bytes": {k: v for k, v in sizes.items() if v},
                "python_median_ratio": {
                    k: float(np.median(np.array(raw_utf8) / np.array(v)))
                    for k, v in sizes.items()
                    if v
                },
            }
            manifest["cases"].append(case)
            ratios = " ".join(
                f"{k}={case['python_median_ratio'][k]:.2f}x"
                for k in ("raw16", "raw24", "freq", "ans")
                if k in case["python_median_ratio"]
            )
            print(f"  {tname:>7}: {ratios}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nfixture written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
