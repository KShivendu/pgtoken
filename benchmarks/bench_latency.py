#!/usr/bin/env python3
"""End-to-end agent read latency: a UTF-8 text column vs a token-native column.

What an agent actually pays to get token IDs out of Postgres, measured through a real client
at realistic RAG fan-out:

  text path   SELECT body            -> server decompresses pglz -> wire -> client tokenizes
  token path  SELECT body            -> server hands over the blob -> wire -> client decodes

The text path's tokenize is mandatory: a model cannot consume characters. That is the cost
token-native storage removes, and it recurs on every read.

Two protocol rules, both of which will silently invalidate the result if ignored:

  * Tokenize is measured serving-cold. A back-to-back tokenize loop keeps tiktoken's
    multi-megabyte rank table resident and reports roughly half the real cost. The repo
    README calls this out; here every iteration sweeps the cache first, so both paths face
    the same cache state and neither is flattered.
  * Pin to a performance core. On a hybrid CPU an unpinned run can migrate to an LP-E core
    and inflate every cell ~1.6x:

        taskset -c 4 uv run python 12_postgres/harness/bench_latency.py

Codec coverage note: `+freq` is excluded from the client-decode path because the Rust
`stream-vbyte` container differs from Python's `pyfastpfor` one, so this Python client cannot
decode what the extension wrote. `raw24` and `ans` are byte-compatible across the two (the
Rust test suite verifies it) and bracket the interesting range: raw24 is the fastest decode,
ans the highest ratio. `+freq`'s server side is byte-for-byte the same memcpy as raw24; only
its client decode is unmeasured here.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tiktoken

import pgcommon as C
import tnbench as T

# Codecs the Python client can decode from the extension's output. See the module docstring.
CLIENT_CODECS = ("raw24", "ans")


def build_tables(conn, texts: list[str], codec: str, table_id: int, embedding_bytes: int):
    """Load the token-native table under one codec, plus the two text baselines."""
    conn.execute("DROP TABLE IF EXISTS bench_text, bench_text_lz4, bench_tnt")
    conn.execute(
        "CREATE TABLE bench_text (id integer PRIMARY KEY, embedding bytea, body text)"
    )
    conn.execute(
        "CREATE TABLE bench_text_lz4 "
        "(id integer PRIMARY KEY, embedding bytea, body text COMPRESSION lz4)"
    )
    conn.execute(
        "CREATE TABLE bench_tnt (id integer PRIMARY KEY, embedding bytea, body bytea)"
    )
    conn.execute("ALTER TABLE bench_tnt ALTER COLUMN body SET STORAGE EXTERNAL")
    for t in ("bench_text", "bench_text_lz4", "bench_tnt"):
        conn.execute(f"ALTER TABLE {t} ALTER COLUMN embedding SET STORAGE EXTERNAL")

    embedding = b"\x00" * embedding_bytes

    # Encode server-side once at load time. In production an agent would send already-encoded
    # blobs, so this is setup cost, not part of any measurement below.
    encoded: list[bytes] = []
    with conn.cursor() as cur:
        for start in range(0, len(texts), 500):
            batch = texts[start : start + 500]
            cur.execute(
                "SELECT pgtoken.encode(t, %s, %s, %s) FROM unnest(%s::text[]) t",
                (C.DEFAULT_TOKENIZER, codec, table_id, batch),
            )
            encoded.extend(r[0] for r in cur.fetchall())

    with conn.cursor() as cur:
        with cur.copy("COPY bench_text (id, embedding, body) FROM STDIN") as cp:
            for i, t in enumerate(texts):
                cp.write_row((i, embedding, t))
        with cur.copy("COPY bench_text_lz4 (id, embedding, body) FROM STDIN") as cp:
            for i, t in enumerate(texts):
                cp.write_row((i, embedding, t))
        with cur.copy("COPY bench_tnt (id, embedding, body) FROM STDIN") as cp:
            for i, b in enumerate(encoded):
                cp.write_row((i, embedding, b))

    for t in ("bench_text", "bench_text_lz4", "bench_tnt"):
        conn.execute(f"VACUUM (ANALYZE) {t}")
    return encoded


def decode_client(blobs: list[bytes], codec: str, enc, ans_model) -> int:
    """Decode stored blobs to token IDs, client-side. Returns a token count so the work
    cannot be optimised away."""
    total = 0
    for b in blobs:
        # 12-byte header: magic, version, tokenizer, codec, table_id u16, reserved, n u32 LE.
        n = int.from_bytes(b[8:12], "little")
        payload = b[12:]
        if codec == "raw24":
            ids = T.unpack3(payload, n)
        elif codec == "ans":
            import constriction

            buf = np.frombuffer(payload, dtype=np.uint32).copy()
            ids = constriction.stream.stack.AnsCoder(buf).decode(ans_model, n)
        else:
            raise ValueError(f"no Python client decoder for {codec}")
        total += len(ids)
    return total


def bench(conn, ids_pool: list[int], fanout: int, reps: int, codec: str, enc, ans_model) -> dict:
    """Time one read of `fanout` rows, for each of the three storage shapes."""
    rng = np.random.default_rng(4242)
    out: dict[str, list[float]] = {
        "text_total": [], "text_fetch": [], "text_tokenize": [],
        "lz4_total": [], "lz4_fetch": [], "lz4_tokenize": [],
        "tnt_total": [], "tnt_fetch": [], "tnt_decode": [],
    }
    wire = {"text": [], "lz4": [], "tnt": []}

    with conn.cursor(binary=True) as cur:
        for _ in range(reps):
            picks = rng.choice(len(ids_pool), size=fanout, replace=False).tolist()

            # Same cache state for both paths: evict the tokenizer's rank table so the text
            # path pays its real serving-cold tokenize cost.
            T.cache_pollute()
            t0 = time.perf_counter()
            cur.execute("SELECT body FROM bench_text WHERE id = ANY(%s)", (picks,))
            rows = [r[0] for r in cur.fetchall()]
            t1 = time.perf_counter()
            enc.encode_ordinary_batch(rows) if hasattr(enc, "encode_ordinary_batch") else [
                enc.encode(r, disallowed_special=()) for r in rows
            ]
            t2 = time.perf_counter()
            out["text_fetch"].append((t1 - t0) * 1e6)
            out["text_tokenize"].append((t2 - t1) * 1e6)
            out["text_total"].append((t2 - t0) * 1e6)
            wire["text"].append(sum(len(r.encode("utf-8")) for r in rows))

            T.cache_pollute()
            t0 = time.perf_counter()
            cur.execute("SELECT body FROM bench_text_lz4 WHERE id = ANY(%s)", (picks,))
            rows = [r[0] for r in cur.fetchall()]
            t1 = time.perf_counter()
            [enc.encode(r, disallowed_special=()) for r in rows]
            t2 = time.perf_counter()
            out["lz4_fetch"].append((t1 - t0) * 1e6)
            out["lz4_tokenize"].append((t2 - t1) * 1e6)
            out["lz4_total"].append((t2 - t0) * 1e6)
            wire["lz4"].append(sum(len(r.encode("utf-8")) for r in rows))

            T.cache_pollute()
            t0 = time.perf_counter()
            cur.execute("SELECT body FROM bench_tnt WHERE id = ANY(%s)", (picks,))
            blobs = [bytes(r[0]) for r in cur.fetchall()]
            t1 = time.perf_counter()
            decode_client(blobs, codec, enc, ans_model)
            t2 = time.perf_counter()
            out["tnt_fetch"].append((t1 - t0) * 1e6)
            out["tnt_decode"].append((t2 - t1) * 1e6)
            out["tnt_total"].append((t2 - t0) * 1e6)
            wire["tnt"].append(sum(len(b) for b in blobs))

    res = {k: C.summarize(v) for k, v in out.items()}
    res["wire_bytes"] = {k: int(np.median(v)) for k, v in wire.items()}
    res["fanout"] = fanout
    res["codec"] = codec
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5000)
    ap.add_argument("--domain", default="prose")
    ap.add_argument("--fanout", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--embedding-bytes", type=int, default=0,
                    help="0 isolates the text column; an unquantized embedding dilutes every ratio")
    ap.add_argument("--codecs", nargs="+", default=list(CLIENT_CODECS))
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_results.json"),
    )
    args = ap.parse_args()

    for c in args.codecs:
        if c not in CLIENT_CODECS:
            ap.error(f"no Python client decoder for {c!r}; choose from {CLIENT_CODECS}")

    texts = C.load_corpus(args.domain, args.docs)
    enc = tiktoken.get_encoding(f"{C.DEFAULT_TOKENIZER}_base")
    print(f"corpus: {len(texts)} docs from {args.domain}, tokenizer {C.DEFAULT_TOKENIZER}")

    runs = []
    with C.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
        C.train_tables(conn, args.domain)
        settings = {
            k: conn.execute(f"SHOW {k}").fetchone()[0]
            for k in ("shared_buffers", "default_toast_compression", "server_version")
        }

        for codec in args.codecs:
            table_id = C.ANS_TABLE_ID if codec == "ans" else 0
            print(f"\nloading token-native table with codec {codec}...")
            build_tables(conn, texts, codec, table_id, args.embedding_bytes)

            ans_model = None
            if codec == "ans":
                # Rebuild the same static model the extension used, so the client can decode.
                import constriction

                r50k = tiktoken.get_encoding("r50k_base")
                train_text = r50k.decode(T.load_ids(f"{args.domain}_train")[: 400 * 512].tolist())
                vocab = 200_019
                ids = np.array(enc.encode(train_text, disallowed_special=()), dtype=np.int64)
                counts = np.bincount(ids[ids < vocab], minlength=vocab).astype(np.int64) + 1
                probs = counts.astype(np.float64) / counts.sum()
                ans_model = constriction.stream.model.Categorical(probs, perfect=False)

            for fanout in args.fanout:
                print(f"  fanout {fanout}, {args.reps} reps...")
                runs.append(bench(conn, list(range(len(texts))), fanout, args.reps, codec, enc, ans_model))

    results = {
        "config": {
            "docs": len(texts),
            "domain": args.domain,
            "tokenizer": C.DEFAULT_TOKENIZER,
            "reps": args.reps,
            "embedding_bytes": args.embedding_bytes,
            **settings,
        },
        "runs": runs,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nPostgreSQL {settings['server_version']}, default_toast_compression="
          f"{settings['default_toast_compression']}")
    print("agent read latency, median us per query (lower is better)\n")
    hdr = (f"{'codec':<7} {'fan':>4} | {'text total':>11} {'= fetch':>8} {'+ tok':>8} | "
           f"{'tnt total':>10} {'= fetch':>8} {'+ dec':>7} | {'speedup':>8} {'wire':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        sp = r["text_total"]["median"] / r["tnt_total"]["median"]
        print(
            f"{r['codec']:<7} {r['fanout']:>4} | "
            f"{r['text_total']['median']:>11.1f} {r['text_fetch']['median']:>8.1f} "
            f"{r['text_tokenize']['median']:>8.1f} | "
            f"{r['tnt_total']['median']:>10.1f} {r['tnt_fetch']['median']:>8.1f} "
            f"{r['tnt_decode']['median']:>7.1f} | {sp:>7.1f}x "
            f"{r['wire_bytes']['text']:>5}->{r['wire_bytes']['tnt']:<5}"
        )

    print("\nlz4 text baseline (what Qdrant/Elasticsearch do), median us:")
    for r in runs:
        sp = r["lz4_total"]["median"] / r["tnt_total"]["median"]
        print(f"  {r['codec']:<7} fanout {r['fanout']:>4}: {r['lz4_total']['median']:>9.1f} "
              f"(tnt is {sp:.1f}x faster)")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
