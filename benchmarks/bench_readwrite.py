#!/usr/bin/env python3
"""Read and write latency: a UTF-8 `text` column vs a `pgtoken` column.

The workload is an agent, which holds token IDs and wants token IDs back. Both paths are
measured end to end from the client, because that is where the difference lives:

    write  text      IDs -> detokenize -> send text -> server compresses (pglz or lz4)
           pgtoken   IDs -> encode     -> send bytea

    read   text      SELECT -> server decompresses -> wire -> client tokenizes -> IDs
           pgtoken   SELECT -> wire -> client decodes -> IDs

Tokenizing is not optional on the text path: a model cannot consume characters. That is the
cost pgtoken removes, and on reads it recurs every time.

Two protocol rules, both of which quietly invalidate the result if ignored:

  * Tokenize is measured serving-cold. A back-to-back loop keeps tiktoken's multi-megabyte
    rank table resident and reports roughly half the real cost, so every iteration sweeps the
    cache first. Both paths face the same cache state, so neither is flattered.
  * Pin to a quiet core. Absolute microseconds on a loaded machine inflate several-fold,
    though the ratios hold:

        taskset -c 4 uv run python benchmarks/bench_readwrite.py

Usage:
    bash setup_pg.sh --start
    TOKEN_STORAGE_REPO=/path/to/token-storage \\
      taskset -c 4 uv run python benchmarks/bench_readwrite.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import pgcommon as C
import pgtoken_client as K
import tnbench as T

# (label, kind, codec) — `kind` selects the column type and therefore the code path.
VARIANTS = [
    ("text (pglz)", "text", None),
    ("text (lz4)", "text_lz4", None),
    ("pgtoken raw", "tok", "raw"),
    ("pgtoken freq", "tok", "freq"),
]


def ddl(name: str, kind: str) -> str:
    col = {
        "text": "body text",
        "text_lz4": "body text COMPRESSION lz4",
        "tok": "body bytea",
    }[kind]
    stmt = f"CREATE TABLE {name} (id integer PRIMARY KEY, {col})"
    # bytea holds an already-compressed payload, so tell PostgreSQL not to compress it again.
    return stmt + (f"; ALTER TABLE {name} ALTER COLUMN body SET STORAGE EXTERNAL" if kind == "tok" else "")


def prepare(conn, texts, id_lists, table):
    """Create one table per variant and return the payloads each will be given."""
    payloads = {}
    for label, kind, codec in VARIANTS:
        name = "b_" + label.split()[0] + ("_lz4" if "lz4" in label else "") + (f"_{codec}" if codec else "")
        name = name.replace("(", "").replace(")", "")
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        for stmt in ddl(name, kind).split(";"):
            conn.execute(stmt)
        if kind == "tok":
            tid = C.FREQ_TABLE_ID if codec == "freq" else 0
            payloads[label] = (name, [K.encode(ids, codec, tid, table) for ids in id_lists])
        else:
            payloads[label] = (name, texts)
    return payloads


def bench_write(conn, payloads, id_lists, texts, table, batch, reps):
    """Time an agent writing `batch` rows, from token IDs it already holds."""
    out = {}
    for label, kind, codec in VARIANTS:
        name, _ = payloads[label]
        enc = C.encoder()
        client_us, server_us, bytes_sent = [], [], []

        for r in range(reps):
            lo = (r * batch) % (len(id_lists) - batch)
            ids_batch = id_lists[lo : lo + batch]

            conn.execute(f"TRUNCATE {name}")
            T.cache_pollute()

            t0 = time.perf_counter()
            if kind == "tok":
                tid = C.FREQ_TABLE_ID if codec == "freq" else 0
                rows = [K.encode(ids, codec, tid, table) for ids in ids_batch]
            else:
                # The agent holds IDs, so the text column costs a detokenize before it can
                # send anything.
                rows = [enc.decode(list(ids)) for ids in ids_batch]
            t1 = time.perf_counter()
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {name} (id, body) VALUES (%s, %s)",
                    list(enumerate(rows)),
                )
            t2 = time.perf_counter()

            client_us.append((t1 - t0) * 1e6 / batch)
            server_us.append((t2 - t1) * 1e6 / batch)
            bytes_sent.append(
                sum(len(x) if isinstance(x, bytes) else len(x.encode()) for x in rows) / batch
            )

        out[label] = {
            "client_us_per_row": C.summarize(client_us)["median"],
            "insert_us_per_row": C.summarize(server_us)["median"],
            "total_us_per_row": C.summarize(client_us)["median"] + C.summarize(server_us)["median"],
            "bytes_per_row": float(np.median(bytes_sent)),
        }
    return out


def bench_read(conn, payloads, table, fanout, reps, n_rows):
    """Time an agent reading `fanout` rows and getting token IDs back."""
    out = {}
    rng = np.random.default_rng(4242)
    for label, kind, codec in VARIANTS:
        name, rows = payloads[label]
        # Load the table once for reading.
        conn.execute(f"TRUNCATE {name}")
        with conn.cursor() as cur:
            with cur.copy(f"COPY {name} (id, body) FROM STDIN") as cp:
                for i, p in enumerate(rows):
                    cp.write_row((i, p))
        conn.execute(f"VACUUM (ANALYZE) {name}")

        enc = C.encoder()
        fetch_us, codec_us, wire = [], [], []
        with conn.cursor(binary=True) as cur:
            for _ in range(reps):
                picks = rng.choice(n_rows, size=fanout, replace=False).tolist()
                T.cache_pollute()

                t0 = time.perf_counter()
                cur.execute(f"SELECT body FROM {name} WHERE id = ANY(%s)", (picks,))
                got = [r[0] for r in cur.fetchall()]
                t1 = time.perf_counter()
                if kind == "tok":
                    for b in got:
                        K.decode(bytes(b), table)
                else:
                    # A model needs IDs, so the text column pays a tokenize on every read.
                    for s in got:
                        enc.encode(s, disallowed_special=())
                t2 = time.perf_counter()

                fetch_us.append((t1 - t0) * 1e6)
                codec_us.append((t2 - t1) * 1e6)
                wire.append(
                    sum(len(x) if isinstance(x, (bytes, memoryview)) else len(x.encode()) for x in got)
                )

        out[label] = {
            "fetch_us": C.summarize(fetch_us)["median"],
            "codec_us": C.summarize(codec_us)["median"],
            "total_us": C.summarize(fetch_us)["median"] + C.summarize(codec_us)["median"],
            "p99_us": C.summarize(fetch_us)["p99"] + C.summarize(codec_us)["p99"],
            "wire_bytes": int(np.median(wire)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--domain", default="prose")
    ap.add_argument("--fanout", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 50])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "readwrite_results.json"),
    )
    args = ap.parse_args()

    texts = C.load_corpus(args.domain, args.docs)
    id_lists = [np.asarray(x, dtype=np.uint32) for x in C.tokenize_all(texts)]
    print(f"corpus: {len(texts)} docs from {args.domain}, {C.TOKENIZER} (tokenized client-side)")

    with C.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
        C.train_table(conn, args.domain)
        settings = {
            k: conn.execute(f"SHOW {k}").fetchone()[0]
            for k in ("server_version", "default_toast_compression", "shared_buffers")
        }
        table_path = os.path.join(
            os.path.expanduser("~/.local/share/pgtoken-pg/tables"), f"{C.FREQ_TABLE_ID}.tntt"
        )
        table = K.RankTable.load(table_path)

        payloads = prepare(conn, texts, id_lists, table)

        writes = {}
        for b in args.batch:
            print(f"  write, batch {b}...")
            writes[b] = bench_write(conn, payloads, id_lists, texts, table, b, args.reps)

        reads = {}
        for f in args.fanout:
            print(f"  read, fan-out {f}...")
            reads[f] = bench_read(conn, payloads, table, f, args.reps, len(texts))


    results = {
        "config": {
            "docs": len(texts),
            "domain": args.domain,
            "tokenizer": C.TOKENIZER,
            "reps": args.reps,
            **settings,
        },
        "write": writes,
        "read": reads,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nPostgreSQL {settings['server_version'].split()[0]}, "
          f"default_toast_compression={settings['default_toast_compression']}")

    for b, res in writes.items():
        print(f"\nWRITE, batch {b} — us per row (agent holds token IDs)")
        hdr = f"  {'variant':<14} {'total':>9} {'= client':>9} {'+ insert':>9} {'bytes':>8} {'vs text':>8}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        base = res["text (pglz)"]["total_us_per_row"]
        for label, _, _ in VARIANTS:
            r = res[label]
            print(f"  {label:<14} {r['total_us_per_row']:>9.1f} {r['client_us_per_row']:>9.1f} "
                  f"{r['insert_us_per_row']:>9.1f} {r['bytes_per_row']:>8.0f} "
                  f"{base / r['total_us_per_row']:>7.2f}x")

    for f, res in reads.items():
        print(f"\nREAD, fan-out {f} — us per query (agent wants token IDs)")
        hdr = f"  {'variant':<14} {'total':>9} {'= fetch':>9} {'+ decode':>9} {'wire B':>9} {'vs text':>8}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        base = res["text (pglz)"]["total_us"]
        for label, _, _ in VARIANTS:
            r = res[label]
            print(f"  {label:<14} {r['total_us']:>9.1f} {r['fetch_us']:>9.1f} "
                  f"{r['codec_us']:>9.1f} {r['wire_bytes']:>9} {base / r['total_us']:>7.2f}x")

    print("\nNote: the codec column is this Python client, which pays numpy overhead on\n"
          "512-element arrays. For the codec cost in isolation, run:\n"
          "  cd core && cargo run --release --example codec_bench")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
