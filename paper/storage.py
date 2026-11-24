#!/usr/bin/env python3
"""Wave 2: storage by value length, every arm, three languages. Extends
benchmarks/bench_length_buckets.py (value size only) and bench_storage_wal.py (one shape) to the
paper's arms and to o200k.

For each domain (C4 English prose, code, Hindi; token-storage test splits) and each length bucket
(o200k tokens per value), the same values go into one table per arm, (id int PRIMARY KEY, body
<type>), loaded with binary COPY. Per (domain, bucket, arm):

    value B     avg pg_column_size(body): bytes PostgreSQL keeps per value, after any compression
    table B     heap + TOAST bytes per row (the primary key index is the same for every arm, left out)
    WAL B       WAL bytes per row written by the COPY (pg_current_wal_lsn before and after)
    rows/page   rows per 8 KB heap page
    toasted     share of values stored out of line in TOAST

Arms are rag_e2e's: text (pglz), text (lz4), text (lz4, forced: STORAGE MAIN + toast_tuple_target
128), int4[], bytea raw24, bytea freq (client codec), pgtoken raw, pgtoken freq.

Values: consecutive non-overlapping windows of each split's o200k token stream, decoded to text and
re-encoded (stored ids == tokenize(text), as in pgcommon.o200k_chunks).

Usage:
    python paper/storage.py [--rows 1000] [--buckets 16 64 128 256 512 1024 2048]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperlib as P  # noqa: E402

C = P.C
import tnbench as T  # noqa: E402

from rag_e2e import ARMS, COPY_HEADER, COPY_TRAILER, body_bin, ensure_vocabs, field  # noqa: E402

STORE_ARMS = [a for a in ARMS if a.coltype and a.client != "tokenize_batch"]


def windows(domain: str, n_tok: int, rows: int):
    o = C._tiktoken("o200k_base", "o200k").tt
    r50k = C._tiktoken("r50k_base", "r50k").tt
    stream = o.encode(r50k.decode(T.load_ids(f"{domain}_test").tolist()), disallowed_special=())
    texts, ids = [], []
    for s in range(0, len(stream) - n_tok + 1, n_tok):
        t = o.decode_bytes(stream[s:s + n_tok]).decode("utf-8", errors="ignore").replace("\x00", "")
        if not t:
            continue
        texts.append(t)
        ids.append(np.asarray(o.encode(t, disallowed_special=()), dtype=np.uint32))
        if len(texts) >= rows:
            break
    return texts, ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["prose", "code", "hindi"])
    ap.add_argument("--buckets", type=int, nargs="+", default=[16, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import pgtoken_rs as R

    out_dir = P.run_dir("storage", args.smoke)
    print(f"run dir: {out_dir}", flush=True)
    res = {"config": vars(args) | {"machine": P.machine()}, "cells": []}
    with C.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
        vocab = ensure_vocabs(conn)
        table = R.Table.from_bytes(vocab["ranking"])
        res["config"]["server_version"] = conn.execute("SHOW server_version").fetchone()[0]
        for dom in args.domains:
            for b in args.buckets:
                texts, ids = windows(dom, b, args.rows)
                n = len(texts)
                utf8 = float(np.mean([len(t.encode()) for t in texts]))
                print(f"\n{dom}, {b} tokens: {n} values, {utf8:.0f} B UTF-8 each", flush=True)
                print(f"  {'arm':<36} {'value B':>8} {'table B':>8} {'WAL B':>8} {'rows/page':>9} "
                      f"{'toasted':>8} {'vs text lz4':>11}", flush=True)
                base = None
                for a in STORE_ARMS:
                    name = f"st_{a.table}"
                    conn.execute(f"DROP TABLE IF EXISTS {name}")
                    conn.execute(f"CREATE TABLE {name} (id int PRIMARY KEY, body {a.coltype})")
                    if a.key == "text_lz4_forced":
                        conn.execute(f"ALTER TABLE {name} ALTER COLUMN body SET STORAGE MAIN, "
                                     "SET (toast_tuple_target = 128)")
                    conn.execute("CHECKPOINT")
                    lsn0 = conn.execute("SELECT pg_current_wal_insert_lsn()").fetchone()[0]
                    with conn.cursor() as cur:
                        with cur.copy(f"COPY {name} (id, body) FROM STDIN (FORMAT BINARY)") as cp:
                            cp.write(COPY_HEADER)
                            cp.write(b"".join(
                                struct.pack(">h", 2) + field(struct.pack(">i", i))
                                + field(body_bin(a, texts[i], ids[i], vocab, R, table))
                                for i in range(n)))
                            cp.write(COPY_TRAILER)
                    wal = conn.execute("SELECT pg_wal_lsn_diff(pg_current_wal_insert_lsn(), %s)",
                                       (lsn0,)).fetchone()[0]
                    conn.execute(f"VACUUM (ANALYZE) {name}")
                    val, heap, toast, pages, tname = conn.execute(
                        f"SELECT (SELECT avg(pg_column_size(body)) FROM {name}), "
                        "pg_relation_size(c.oid), coalesce(pg_relation_size(c.reltoastrelid), 0), "
                        "c.relpages, c.reltoastrelid::regclass::text FROM pg_class c "
                        "WHERE c.oid = %s::regclass", (name,)).fetchone()
                    toasted = conn.execute(f"SELECT count(DISTINCT chunk_id) FROM {tname}").fetchone()[0] \
                        if tname and tname != "-" else 0
                    cell = {"domain": dom, "bucket": b, "arm": a.key, "label": a.label, "rows": n,
                            "utf8_bytes": utf8, "value_bytes": float(val),
                            "table_bytes_per_row": (heap + toast) / n, "wal_bytes_per_row": float(wal) / n,
                            "rows_per_page": n / max(1, pages), "toasted_share": toasted / n}
                    res["cells"].append(cell)
                    if a.key == "text_lz4":
                        base = cell["table_bytes_per_row"]
                    print(f"  {a.label:<36} {cell['value_bytes']:>8.0f} {cell['table_bytes_per_row']:>8.0f} "
                          f"{cell['wal_bytes_per_row']:>8.0f} {cell['rows_per_page']:>9.1f} "
                          f"{100 * cell['toasted_share']:>7.0f}% "
                          f"{(base / cell['table_bytes_per_row']) if base else float('nan'):>10.2f}x",
                          flush=True)
                    conn.execute(f"DROP TABLE {name}")
                P.write_json(os.path.join(out_dir, "result.json"), res)
    print(f"\nwrote {out_dir}/result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
