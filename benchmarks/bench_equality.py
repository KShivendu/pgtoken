#!/usr/bin/env python3
"""Equality, DISTINCT, GROUP BY and hash joins: text columns vs tokens columns.

A compressed text value has to be decompressed before PostgreSQL can compare or hash it.
A tokens value is compared and hashed as stored bytes, with no decode, which is only correct
because one vocabulary has exactly one encoding of a given id sequence. This measures what
that buys today, and so what a design with several encodings per vocabulary would give up.

Data: 20000 rows drawn with replacement from the 2931 distinct 512-token prose test chunks,
so each value repeats ~7 times. Stored four ways: text (pglz), text (lz4), tokens raw, tokens
freq. Runs in its own database (pgtoken_eq) with vocabulary ids 100 and 101, so it never
collides with the other benchmarks' vocabularies or ranking files.

Usage:
    uv run python benchmarks/bench_equality.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import psycopg
import pgtoken_rs as R
import tiktoken

import pgcommon as C
import tnbench as T

DB = "pgtoken_eq"
TABLES = {
    "text (pglz)": ("eq_text", "text"),
    "text (lz4)": ("eq_lz4", "text COMPRESSION lz4"),
    "tokens raw": ("eq_raw", "pgtoken.eq_raw"),
    "tokens freq": ("eq_freq", "pgtoken.eq_freq"),
}
QUERIES = {
    # count(DISTINCT body) is not here: an aggregate's DISTINCT always sorts, and tokens has no
    # ordering, so it errors. SELECT DISTINCT can hash instead, which is what this times.
    "SELECT DISTINCT body (hash)": ("SELECT count(*) FROM (SELECT DISTINCT body FROM {t}) s", False),
    "GROUP BY body (hash aggregate)": ("SELECT count(*) FROM (SELECT body FROM {t} GROUP BY body) s", False),
    "self join a.body = b.body (hash join)": ("SELECT count(*) FROM {t} a JOIN {t} b ON a.body = b.body", False),
    "WHERE body = <one value> (scan)": ("SELECT count(*) FROM {t} WHERE body = (SELECT body FROM {t} WHERE id = 7)", True),
}


def setup(rows: int, seed: int) -> None:
    with C.connect() as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB,)).fetchone():
            admin.execute(f"CREATE DATABASE {DB}")
    env = C.pg_env()
    conn = psycopg.connect(host=env["PGHOST"], port=int(env["PGPORT"]), user="postgres", dbname=DB, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
    for name, kind, vid in (("eq_raw", "raw", 100), ("eq_freq", "freq", 101)):
        if not conn.execute("SELECT 1 FROM pgtoken.vocabulary WHERE name = %s", (name,)).fetchone():
            conn.execute("SELECT pgtoken.create_vocabulary(%s, 50257, compression => %s, id => %s)", (name, kind, vid))
    if conn.execute("SELECT ranked FROM pgtoken.vocabulary_info('eq_freq')").fetchone()[0] is None:
        train = T.load_ids("prose_train")[:2_000_000]
        conn.execute("CREATE TEMP TABLE tr(ids int[])")
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO tr VALUES (%s)", [(train[i:i + 4096].tolist(),) for i in range(0, len(train), 4096)])
        conn.execute("SELECT pgtoken.train('eq_freq', 'SELECT ids FROM tr')")
    tbl = R.Table.from_bytes(C.ranking_bytes(conn, "eq_freq"))

    enc = tiktoken.get_encoding("r50k_base")  # the corpus is stored as r50k (= gpt2) ids
    test = T.load_ids("prose_test")
    chunks = [test[i * 512:(i + 1) * 512].astype(np.uint32) for i in range(len(test) // 512)]
    # A text column cannot hold a NUL byte, and one chunk has one (token 188, "\x00"). A tokens
    # column stores it fine, but every arm must hold the same data, so drop it everywhere.
    nul = [i for i, c in enumerate(chunks) if "\x00" in enc.decode(c.tolist())]
    chunks = [c for i, c in enumerate(chunks) if i not in nul]
    print(f"dropped {len(nul)} chunk(s) a text column cannot store (NUL byte): {nul}", flush=True)
    pick = np.random.default_rng(seed).integers(0, len(chunks), rows)
    texts = {i: enc.decode(chunks[i].tolist()) for i in set(pick.tolist())}
    print(f"{rows} rows from {len(chunks)} chunks, {len(set(pick.tolist()))} distinct values", flush=True)

    for label, (t, col) in TABLES.items():
        conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.execute(f"CREATE TABLE {t} (id int PRIMARY KEY, body {col})")
        with conn.cursor() as cur, cur.copy(f"COPY {t} (id, body) FROM STDIN (FORMAT BINARY)") as cp:
            is_text = col.startswith("text")
            cp.set_types(["int4", "text" if is_text else "bytea"])
            for rid, c in enumerate(pick.tolist()):
                if is_text:
                    v = texts[c]
                elif t == "eq_raw":
                    v = R.encode(chunks[c], "raw16", 100)
                else:
                    v = R.encode(chunks[c], "freq", 101, tbl)
                cp.write_row((rid, v))
        conn.execute(f"VACUUM ANALYZE {t}")
        mb = conn.execute(f"SELECT pg_total_relation_size('{t}') / 1e6").fetchone()[0]
        print(f"  loaded {label:<12} {t}: {mb:.1f} MB", flush=True)
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "equality_results.json"))
    args = ap.parse_args()
    setup(args.rows, args.seed)

    env = C.pg_env()
    conn = psycopg.connect(host=env["PGHOST"], port=int(env["PGPORT"]), user="postgres", dbname=DB, autocommit=True)
    conn.execute("SET search_path = public, pgtoken")  # a bare = on tokens needs pgtoken on the path
    conn.execute("SET max_parallel_workers_per_gather = 0")
    conn.execute("SET work_mem = '256MB'")
    for t, _ in TABLES.values():  # warm shared buffers so this times compare and hash, not disk
        conn.execute(f"SELECT count(*) FROM {t} WHERE body IS NOT NULL")

    res = {}
    for qname, (q, sort_ok) in QUERIES.items():
        conn.execute(f"SET enable_sort = {'on' if sort_ok else 'off'}")
        res[qname] = {}
        line = []
        for label, (t, _) in TABLES.items():
            ts, out = [], None
            for _ in range(args.reps):
                a = time.perf_counter()
                out = conn.execute(q.format(t=t)).fetchone()[0]
                ts.append((time.perf_counter() - a) * 1e3)
            plan = " ".join(r[0] for r in conn.execute("EXPLAIN " + q.format(t=t)).fetchall())
            node = next((n for n in ("HashAggregate", "Hash Join", "Sort", "Seq Scan") if n in plan), "?")
            res[qname][label] = {"ms_p50": float(np.median(ts)), "result": out, "plan_node": node}
            line.append(f"{label} {np.median(ts):7.1f} ms")
        results = {v["result"] for v in res[qname].values()}
        print(f"{qname:<40} " + " | ".join(line) + ("" if len(results) == 1 else f"   RESULTS DIFFER {results}"), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": res}, f, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
