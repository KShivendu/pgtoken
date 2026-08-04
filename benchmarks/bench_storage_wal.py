#!/usr/bin/env python3
"""Storage, WAL volume, and shared-buffer density: text vs token-native.

These are the effects a component-level benchmark cannot show, and the reason a real
Postgres extension is worth building rather than asserting:

  * Storage. Heap, TOAST and total relation size per table.
  * WAL. Payload goes to WAL on every insert, so a smaller payload means less fsync, faster
    replication and smaller archives. Measured as a pg_current_wal_lsn() delta around a bulk
    COPY, cross-checked against pg_stat_wal.
  * Buffer density. Documents per 8 kB page, and therefore per GB of shared_buffers. On a
    read-heavy RAG workload this raises the cache-hit ratio, which no per-value microbenchmark
    can capture.

Usage:
    bash 12_postgres/setup_pg.sh --start
    uv run python 12_postgres/harness/bench_storage_wal.py [--docs 20000] [--domain prose]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import psycopg

import pgcommon as C

TABLES = ("docs_text", "docs_text_lz4", "docs_tnt")


def relation_sizes(conn: psycopg.Connection) -> dict:
    out = {}
    for t in TABLES:
        row = conn.execute(
            """
            SELECT pg_relation_size(c.oid)                        AS heap_bytes,
                   COALESCE(pg_relation_size(c.reltoastrelid), 0) AS toast_bytes,
                   pg_indexes_size(c.oid)                         AS index_bytes,
                   pg_total_relation_size(c.oid)                  AS total_bytes,
                   c.relpages,
                   c.reltuples::bigint
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND n.nspname = current_schema()
            """,
            (t,),
        ).fetchone()
        out[t] = {
            "heap_bytes": row[0],
            "toast_bytes": row[1],
            "index_bytes": row[2],
            "total_bytes": row[3],
            "relpages": row[4],
            "reltuples": row[5],
        }
    return out


def payload_bytes(conn: psycopg.Connection) -> dict:
    """Logical payload size per table: what the column actually holds, before page overhead.

    `pg_column_size` reports the on-disk size including any compression Postgres applied, so
    this is where pglz's and lz4's real effect on the text column shows up.
    """
    out = {}
    for t, col in ((t, "body") for t in TABLES):
        row = conn.execute(
            f"""
            SELECT sum(pg_column_size({col}))::bigint,
                   avg(pg_column_size({col}))::numeric(12,1),
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY pg_column_size({col})),
                   sum(octet_length({col}))::bigint
            FROM {t}
            """
        ).fetchone()
        out[t] = {
            "stored_sum": int(row[0]),
            "stored_avg": float(row[1]),
            "stored_median": float(row[2]),
            # For text this is characters-as-bytes before compression; for bytea it equals
            # the stored size, since STORAGE EXTERNAL disables compression.
            "uncompressed_sum": int(row[3]),
        }
    return out


def toast_split(conn: psycopg.Connection) -> dict:
    """How many values went out of line.

    A value that exceeds TOAST_TUPLE_THRESHOLD (~2000 B) after any compression is pushed to
    the TOAST table, which costs an extra index and heap fetch on every read. Whether the
    text column crosses that line while the token-native one does not is exactly the flip
    the plan flagged as measure-do-not-assume.
    """
    out = {}
    for t in TABLES:
        n_toasted = conn.execute(
            f"SELECT count(*) FROM {t} WHERE pg_column_size(body) > 2000"
        ).fetchone()[0]
        total = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        out[t] = {"over_toast_threshold": n_toasted, "rows": total}
    return out


def buffer_density(conn: psycopg.Connection) -> dict:
    """Documents resident per 8 kB page, and the implied capacity per GB of shared_buffers."""
    block_size = int(conn.execute("SHOW block_size").fetchone()[0])
    out = {"block_size": block_size}
    for t in TABLES:
        conn.execute("SELECT pg_prewarm(%s)", (t,))
        row = conn.execute(
            """
            SELECT count(*) FROM pg_buffercache b
            JOIN pg_class c ON c.relfilenode = b.relfilenode
            WHERE c.relname = %s
            """,
            (t,),
        ).fetchone()
        pages = conn.execute(
            "SELECT relpages FROM pg_class WHERE relname = %s", (t,)
        ).fetchone()[0]
        rows = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        docs_per_page = rows / pages if pages else 0.0
        out[t] = {
            "buffers_resident": row[0],
            "heap_pages": pages,
            "rows": rows,
            "docs_per_page": round(docs_per_page, 2),
            "docs_per_gb_shared_buffers": int(docs_per_page * (1 << 30) / block_size),
        }
    return out


def measure_wal(
    conn: psycopg.Connection, texts: list[str], domain: str, embedding_bytes: int
) -> dict:
    """WAL bytes generated by a bulk COPY into each table.

    Measured per table with a fresh LSN delta. Each table is loaded in its own pass so the
    WAL attributed to it is only its own; a single interleaved load could not be split apart.
    """
    embedding = b"\x00" * embedding_bytes
    results = {}

    for t in TABLES:
        conn.execute(f"TRUNCATE {t}")
        conn.execute("CHECKPOINT")

        if t == "docs_tnt":
            # The agent write path: encode once, then COPY opaque blobs. Encoding happens
            # client-side here, which is the point of token-native storage -- the server
            # never runs a tokenizer.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pgtoken.encode(%s, %s, %s, %s)",
                    (texts[0], C.DEFAULT_TOKENIZER, C.DEFAULT_CODEC, C.FREQ_TABLE_ID),
                )
            encoded = []
            with conn.cursor() as cur:
                for chunk_start in range(0, len(texts), 500):
                    batch = texts[chunk_start : chunk_start + 500]
                    cur.execute(
                        "SELECT pgtoken.encode(t, %s, %s, %s) FROM unnest(%s::text[]) t",
                        (C.DEFAULT_TOKENIZER, C.DEFAULT_CODEC, C.FREQ_TABLE_ID, batch),
                    )
                    encoded.extend(r[0] for r in cur.fetchall())
            payloads = encoded
        else:
            payloads = texts

        lsn_before = conn.execute("SELECT pg_current_wal_lsn()").fetchone()[0]
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            with cur.copy(f"COPY {t} (id, embedding, body) FROM STDIN") as cp:
                for i, p in enumerate(payloads):
                    cp.write_row((i, embedding, p))
        elapsed = time.perf_counter() - t0
        lsn_after = conn.execute("SELECT pg_current_wal_lsn()").fetchone()[0]
        wal_bytes = conn.execute(
            "SELECT %s::pg_lsn - %s::pg_lsn", (lsn_after, lsn_before)
        ).fetchone()[0]

        results[t] = {
            "wal_bytes": int(wal_bytes),
            "wal_bytes_per_row": round(int(wal_bytes) / len(payloads), 1),
            "copy_seconds": round(elapsed, 3),
        }
        conn.execute(f"VACUUM (ANALYZE) {t}")

    return results


def run_one(conn, texts: list[str], domain: str, embedding_bytes: int) -> dict:
    C.apply_schema(conn)
    C.train_tables(conn, domain)
    wal = measure_wal(conn, texts, domain, embedding_bytes)
    sizes = relation_sizes(conn)
    payloads = payload_bytes(conn)
    toast = toast_split(conn)
    density = buffer_density(conn)

    base_payload = payloads["docs_text"]["stored_sum"]
    return {
        "embedding_bytes": embedding_bytes,
        "relation_sizes": sizes,
        "payload_bytes": payloads,
        "toast": toast,
        "buffer_density": density,
        "wal": wal,
        "ratios_vs_docs_text": {
            t: {
                "payload": round(base_payload / payloads[t]["stored_sum"], 3),
                "total_relation": round(
                    sizes["docs_text"]["total_bytes"] / sizes[t]["total_bytes"], 3
                )
                if sizes[t]["total_bytes"]
                else None,
                "wal": round(wal["docs_text"]["wal_bytes"] / wal[t]["wal_bytes"], 3)
                if wal[t]["wal_bytes"]
                else None,
            }
            for t in TABLES
        },
    }


def report(run: dict, docs: int) -> None:
    emb = run["embedding_bytes"]
    payloads, sizes, wal = run["payload_bytes"], run["relation_sizes"], run["wal"]
    toast, density = run["toast"], run["buffer_density"]

    label = (
        "text column only (no embedding)"
        if emb == 0
        else f"realistic row (+{emb} B embedding, i.e. a 1024-dim float32 vector)"
    )
    print(f"\n=== {label} ===")
    hdr = (
        f"{'table':<15} {'payload/row':>12} {'heap MB':>9} {'toast MB':>9} "
        f"{'total MB':>9} {'WAL B/row':>10} {'docs/page':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for t in TABLES:
        print(
            f"{t:<15} {payloads[t]['stored_avg']:>12.1f} "
            f"{sizes[t]['heap_bytes']/1e6:>9.1f} {sizes[t]['toast_bytes']/1e6:>9.1f} "
            f"{sizes[t]['total_bytes']/1e6:>9.1f} "
            f"{wal[t]['wal_bytes_per_row']:>10.1f} {density[t]['docs_per_page']:>10.2f}"
        )
    print("\n  vs docs_text (higher is better for token-native):")
    for t in TABLES:
        r = run["ratios_vs_docs_text"][t]
        tot = f"{r['total_relation']:>5.2f}x" if r["total_relation"] else "    -"
        print(
            f"    {t:<15} payload {r['payload']:>5.2f}x   total relation {tot}   "
            f"WAL {r['wal']:>5.2f}x"
        )
    print("\n  values over the ~2000 B TOAST threshold (each costs an extra fetch per read):")
    for t in TABLES:
        print(f"    {t:<15} {toast[t]['over_toast_threshold']:>7} / {toast[t]['rows']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=20_000)
    ap.add_argument("--domain", default="prose")
    ap.add_argument(
        "--embedding-bytes",
        type=int,
        nargs="+",
        default=[0, C.EMBEDDING_BYTES],
        help="row shapes to measure. 0 isolates the text column; 4096 adds an unquantized "
        "1024-dim float32 embedding, which is always TOASTed and dilutes every ratio.",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage_wal_results.json"),
    )
    args = ap.parse_args()

    texts = C.load_corpus(args.domain, args.docs)
    print(f"corpus: {len(texts)} docs from {args.domain}")

    runs = []
    with C.connect() as conn:
        settings = {
            k: conn.execute(f"SHOW {k}").fetchone()[0]
            for k in ("shared_buffers", "block_size", "default_toast_compression", "server_version")
        }
        for emb in args.embedding_bytes:
            print(f"loading and measuring WAL (embedding={emb} B)...")
            runs.append(run_one(conn, texts, args.domain, emb))

    results = {
        "config": {
            "docs": len(texts),
            "domain": args.domain,
            "tokenizer": C.DEFAULT_TOKENIZER,
            "codec": C.DEFAULT_CODEC,
            **settings,
        },
        "runs": runs,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"\nPostgreSQL {settings['server_version']}, shared_buffers={settings['shared_buffers']}, "
        f"default_toast_compression={settings['default_toast_compression']}"
    )
    print(
        f"{len(texts)} docs from {args.domain}, "
        f"token-native = {C.DEFAULT_TOKENIZER} +{C.DEFAULT_CODEC}"
    )
    for run in runs:
        report(run, len(texts))

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
