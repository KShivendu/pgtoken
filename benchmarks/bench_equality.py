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

--source wildchat (PVLDB paper): real ChatGPT messages instead. Every message of the test half of
the WildChat sample cached by paper/compress_by_length.py (allenai/WildChat-1M, ODC-BY), each
stored once, no resampling, so duplicates are the natural ones ("hi", "continue", repeated
prompts). o200k ids (tiktoken), raw24 for the raw arm, freq trained on the train half. Also
reports the real stored size per arm (table, TOAST, average pg_column_size), which checks the
offline ratios of compress_by_length.py inside PostgreSQL.

Usage:
    uv run python benchmarks/bench_equality.py
    PGTOKEN_DSN=postgresql://postgres:postgres@127.0.0.1:5455/postgres \
        uv run python benchmarks/bench_equality.py --source wildchat --out paper/runs/<run>/equality.json
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


def db_conn() -> psycopg.Connection:
    dsn = os.environ.get("PGTOKEN_DSN")
    if dsn:
        return psycopg.connect(dsn, dbname=DB, autocommit=True)
    env = C.pg_env()
    return psycopg.connect(host=env["PGHOST"], port=int(env["PGPORT"]), user="postgres", dbname=DB, autocommit=True)


def load_wildchat(path: str) -> tuple[list[str], list[str]]:
    import gzip
    with gzip.open(os.path.expanduser(path), "rt", encoding="utf-8") as f:
        d = json.load(f)
    ok = lambda t: t and "\x00" not in t  # a text column cannot hold NUL
    return [t for t in d["train"] if ok(t)], [m["text"] for m in d["test"] if ok(m["text"])]


def setup_wildchat(path: str, max_rows: int) -> dict:
    global DB
    DB = "pgtoken_eq_wc"
    with C.connect() as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB,)).fetchone():
            admin.execute(f"CREATE DATABASE {DB}")
    conn = db_conn()
    conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
    enc = tiktoken.get_encoding("o200k_base")
    nv = enc.n_vocab
    raw = C.raw_codec_for(nv)
    train, test = load_wildchat(path)
    test = test[:max_rows] if max_rows else test
    print(f"wildchat: {len(test)} test messages ({len(set(test))} distinct), {len(train)} train messages", flush=True)
    for name, kind, vid in (("eq_wc_raw", "raw", 102), ("eq_wc_freq", "freq", 103)):
        if not conn.execute("SELECT 1 FROM pgtoken.vocabulary WHERE name = %s", (name,)).fetchone():
            conn.execute("SELECT pgtoken.create_vocabulary(%s, %s, compression => %s, id => %s)", (name, nv, kind, vid))
    if conn.execute("SELECT ranked FROM pgtoken.vocabulary_info('eq_wc_freq')").fetchone()[0] is None:
        conn.execute("CREATE TEMP TABLE tr(ids int[])")
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO tr VALUES (%s)",
                            [(enc.encode(t, disallowed_special=()),) for t in train])
        conn.execute("SELECT pgtoken.train('eq_wc_freq', 'SELECT ids FROM tr')")
    tbl = R.Table.from_bytes(C.ranking_bytes(conn, "eq_wc_freq"))
    ids = [np.asarray(enc.encode(t, disallowed_special=()), dtype=np.uint32) for t in test]
    tables = {"text (pglz)": ("wc_text", "text"), "text (lz4)": ("wc_lz4", "text COMPRESSION lz4"),
              "tokens raw24": ("wc_raw", "pgtoken.eq_wc_raw"), "tokens freq": ("wc_freq", "pgtoken.eq_wc_freq")}
    storage = {}
    for label, (t, col) in tables.items():
        conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.execute(f"CREATE TABLE {t} (id int PRIMARY KEY, body {col})")
        is_text = col.startswith("text")
        with conn.cursor() as cur, cur.copy(f"COPY {t} (id, body) FROM STDIN (FORMAT BINARY)") as cp:
            cp.set_types(["int4", "text" if is_text else "bytea"])
            for rid, (txt, x) in enumerate(zip(test, ids)):
                if is_text:
                    v = txt
                elif t == "wc_raw":
                    v = R.encode(x, raw, 102)
                else:
                    v = R.encode(x, "freq", 103, tbl)
                cp.write_row((rid, v))
        conn.execute(f"VACUUM ANALYZE {t}")
        row = conn.execute(
            f"SELECT pg_total_relation_size('{t}'), pg_relation_size('{t}'),"
            f" coalesce(pg_total_relation_size(nullif(reltoastrelid, 0)), 0),"
            f" pg_indexes_size('{t}'), (SELECT avg(pg_column_size(body)) FROM {t}),"
            f" (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY pg_column_size(body)) FROM {t})"
            f" FROM pg_class WHERE oid = '{t}'::regclass").fetchone()
        storage[label] = {"total_bytes": row[0], "heap_bytes": row[1], "toast_bytes": row[2],
                          "index_bytes": row[3], "avg_value_bytes": float(row[4]), "p50_value_bytes": float(row[5])}
        print(f"  loaded {label:<13} {t}: total {row[0] / 1e6:.1f} MB (heap {row[1] / 1e6:.1f}, TOAST "
              f"{row[2] / 1e6:.1f}, pkey {row[3] / 1e6:.1f}), value avg {row[4]:.0f} B, p50 {row[5]:.0f} B", flush=True)
    utf8 = [len(t.encode()) for t in test]
    storage["utf8"] = {"avg_value_bytes": float(np.mean(utf8)), "p50_value_bytes": float(np.median(utf8)),
                       "total_bytes": int(np.sum(utf8)), "tokens_total": int(sum(len(x) for x in ids))}
    for k in (0, len(test) // 2):  # two concrete rows, as stored
        print(f"  example row {k}: {len(ids[k])} tokens, {utf8[k]} B UTF-8: {test[k][:100]!r}", flush=True)
    conn.close()
    return {"tables": tables, "storage": storage, "rows": len(test), "distinct": len(set(test))}


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
    ap.add_argument("--source", choices=["prose", "wildchat"], default="prose")
    ap.add_argument("--wildchat-cache", default="~/.cache/pgtoken/compress_length/wildchat_40000.json.gz")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "equality_results.json"))
    args = ap.parse_args()
    tables, extra = TABLES, {}
    if args.source == "wildchat":
        extra = setup_wildchat(args.wildchat_cache, args.max_rows)
        tables = extra.pop("tables")
    else:
        setup(args.rows, args.seed)

    conn = db_conn()
    conn.execute("SET search_path = public, pgtoken")  # a bare = on tokens needs pgtoken on the path
    conn.execute("SET max_parallel_workers_per_gather = 0")
    conn.execute("SET work_mem = '256MB'")
    for t, _ in tables.values():  # warm shared buffers so this times compare and hash, not disk
        conn.execute(f"SELECT count(*) FROM {t} WHERE body IS NOT NULL")

    res = {}
    for qname, (q, sort_ok) in QUERIES.items():
        conn.execute(f"SET enable_sort = {'on' if sort_ok else 'off'}")
        res[qname] = {}
        line = []
        for label, (t, _) in tables.items():
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
        json.dump({"config": vars(args), "results": res, **extra}, f, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
