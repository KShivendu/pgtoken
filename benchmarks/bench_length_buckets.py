#!/usr/bin/env python3
"""Stored size by value length: text (no compression, pglz, lz4) vs a tokens column.

bench_storage_wal.py measures one shape, 512-token chunks. Users also store short values
(chat turns, titles, 100-200 token snippets) and long ones (2000+ token documents), and
PostgreSQL treats those differently:

  * A text column is STORAGE EXTENDED. PostgreSQL only tries to compress a value once the
    whole row passes ~2 KB (TOAST_TUPLE_THRESHOLD). A 150-token value (~700 B) is never
    compressed, whatever the column's compression setting says.
  * pglz (the default) gives up unless it saves at least 25%. lz4 keeps any saving.
  * A tokens column is STORAGE EXTERNAL: the codec is the only compression, at every length.

So the honest "N x smaller than text" depends on length. This script stores the same values in
five single-column tables and reads pg_column_size(body), the bytes PostgreSQL actually keeps
for each value (inline or in TOAST, after any compression, including the varlena header):

  text_plain   text, STORAGE EXTERNAL, i.e. never compressed (the UTF-8 bytes + header)
  text_pglz    text, default storage, default_toast_compression (pglz)
  text_lz4     text, COMPRESSION lz4
  tok_gpt2     tokens.bench_hfgpt2_freq: gpt2 ids, freq ranking trained on the prose train
               split (the vocabulary the other benchmarks use)
  tok_o200k    tokens.o200k: the bundled preset, the server tokenizes the text

Chunks are non-overlapping windows of the token-storage test splits, cut at a fixed number of
r50k tokens and decoded to text, for prose, code and Hindi. The gpt2 vocabulary is trained on
prose only, so its code and Hindi rows show ranking drift; the o200k preset ships one ranking
counted over all three.

Usage:
    bash setup_pg.sh --start
    PGDATABASE=<scratch db> uv run python benchmarks/bench_length_buckets.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st

import numpy as np

import pgcommon as C
import tnbench as T

TABLES = ("text_plain", "text_pglz", "text_lz4", "tok_gpt2", "tok_o200k")
TEXT_BASELINES = ("text_plain", "text_pglz", "text_lz4")


def create_tables(conn, gpt2_vocab: str) -> None:
    conn.execute("DROP TABLE IF EXISTS " + ", ".join(TABLES))
    cols = "id integer PRIMARY KEY, domain text NOT NULL, bucket integer NOT NULL"
    conn.execute(f"CREATE TABLE text_plain ({cols}, body text NOT NULL)")
    conn.execute("ALTER TABLE text_plain ALTER COLUMN body SET STORAGE EXTERNAL")
    conn.execute(f"CREATE TABLE text_pglz ({cols}, body text NOT NULL)")
    conn.execute(f"CREATE TABLE text_lz4 ({cols}, body text COMPRESSION lz4 NOT NULL)")
    conn.execute(f"CREATE TABLE tok_gpt2 ({cols}, body tokens.{gpt2_vocab} NOT NULL)")
    conn.execute(f"CREATE TABLE tok_o200k ({cols}, body tokens.o200k NOT NULL)")


def ensure_o200k(conn) -> None:
    if conn.execute("SELECT 1 FROM pgtoken.vocabulary WHERE name = 'o200k'").fetchone() is None:
        conn.execute("SELECT pgtoken.create_vocabulary('o200k')")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, nargs="+", default=[16, 64, 128, 192, 512, 2048])
    ap.add_argument("--domains", nargs="+", default=["prose", "code", "hindi"])
    ap.add_argument("--rows", type=int, default=300, help="values per (domain, bucket)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "length_buckets_results.json"),
    )
    args = ap.parse_args()

    r50k = C._tiktoken("r50k_base", "r50k")
    enc = C.encoder()
    rng = np.random.default_rng(args.seed)

    with C.connect() as conn:
        C.apply_schema(conn)  # creates the extensions
        C.ensure_freq_vocab(conn, "prose")
        ensure_o200k(conn)
        gpt2_vocab = C._vocab_name("freq")
        create_tables(conn, gpt2_vocab)
        settings = {
            k: conn.execute(f"SHOW {k}").fetchone()[0]
            for k in ("server_version", "default_toast_compression", "block_size")
        }
        print(f"settings: {settings}", flush=True)

        rid = 0
        for domain in args.domains:
            stream = T.load_ids(f"{domain}_test")
            for b in args.buckets:
                chunks = T.make_chunks(stream, b, args.rows, rng)
                texts = [r50k.decode(c.tolist()).replace("\x00", "") for c in chunks]
                rows = []
                for t in texts:
                    rows.append((rid, domain, b, t, enc.encode(t)))
                    rid += 1
                with conn.cursor() as cur:
                    for tbl in TEXT_BASELINES:
                        cur.executemany(
                            f"INSERT INTO {tbl} (id, domain, bucket, body) VALUES (%s, %s, %s, %s)",
                            [r[:4] for r in rows],
                        )
                    cur.executemany(
                        "INSERT INTO tok_gpt2 (id, domain, bucket, body) VALUES (%s, %s, %s, %s::int[])",
                        [(r[0], r[1], r[2], r[4]) for r in rows],
                    )
                    cur.executemany(
                        "INSERT INTO tok_o200k (id, domain, bucket, body) VALUES (%s, %s, %s, %s)",
                        [r[:4] for r in rows],
                    )
                print(f"  loaded {domain} {b}-token bucket: {len(rows)} values", flush=True)

        # One row per value with every option's stored size side by side.
        data = conn.execute(
            """
            SELECT p.id, p.domain, p.bucket, p.body,
                   octet_length(p.body),
                   pgtoken.token_count(g.body), pgtoken.token_count(o.body),
                   pg_column_size(p.body), pg_column_size(z.body), pg_column_size(l.body),
                   pg_column_size(g.body), pg_column_size(o.body)
            FROM text_plain p
            JOIN text_pglz z USING (id) JOIN text_lz4 l USING (id)
            JOIN tok_gpt2 g USING (id) JOIN tok_o200k o USING (id)
            ORDER BY p.id
            """
        ).fetchall()

    recs = [
        {
            "id": r[0], "domain": r[1], "bucket": r[2], "text": r[3], "utf8": r[4],
            "n_gpt2": r[5], "n_o200k": r[6],
            "text_plain": r[7], "text_pglz": r[8], "text_lz4": r[9],
            "tok_gpt2": r[10], "tok_o200k": r[11],
        }
        for r in data
    ]

    summary = []
    hdr = (
        f"{'domain':<6} {'bucket':>6} {'o200k tok':>9} {'utf8':>6} | {'plain':>6} {'pglz':>6} "
        f"{'lz4':>6} | {'gpt2':>6} {'o200k':>6} | {'o200k vs pglz':>13} {'vs lz4':>7} "
        f"{'o200k >= best text':>18}"
    )
    print("\nmean stored bytes per value (pg_column_size)\n" + hdr + "\n" + "-" * len(hdr), flush=True)
    for domain in args.domains:
        for b in args.buckets:
            g = [x for x in recs if x["domain"] == domain and x["bucket"] == b]
            m = {k: st.mean(x[k] for x in g) for k in ("utf8", "n_o200k", "n_gpt2", *TABLES)}
            lose = [x for x in g if x["tok_o200k"] >= min(x[t] for t in TEXT_BASELINES)]
            row = {
                "domain": domain, "bucket": b, "rows": len(g), **{f"mean_{k}": v for k, v in m.items()},
                "o200k_vs_pglz": m["text_pglz"] / m["tok_o200k"],
                "o200k_vs_lz4": m["text_lz4"] / m["tok_o200k"],
                "gpt2_vs_pglz": m["text_pglz"] / m["tok_gpt2"],
                "gpt2_vs_lz4": m["text_lz4"] / m["tok_gpt2"],
                "o200k_not_smaller": len(lose),
                "pglz_compressed": sum(x["text_pglz"] < x["text_plain"] for x in g),
                "lz4_compressed": sum(x["text_lz4"] < x["text_plain"] for x in g),
            }
            summary.append(row)
            print(
                f"{domain:<6} {b:>6} {m['n_o200k']:>9.0f} {m['utf8']:>6.0f} | {m['text_plain']:>6.0f} "
                f"{m['text_pglz']:>6.0f} {m['text_lz4']:>6.0f} | {m['tok_gpt2']:>6.0f} "
                f"{m['tok_o200k']:>6.0f} | {row['o200k_vs_pglz']:>12.2f}x {row['o200k_vs_lz4']:>6.2f}x "
                f"{len(lose):>11}/{len(g)}",
                flush=True,
            )
            print(
                f"{'':<14} pglz compressed {row['pglz_compressed']}/{len(g)}, "
                f"lz4 compressed {row['lz4_compressed']}/{len(g)}; "
                f"gpt2 vs pglz {row['gpt2_vs_pglz']:.2f}x, vs lz4 {row['gpt2_vs_lz4']:.2f}x",
                flush=True,
            )

    def show(x, why):
        best = min(TEXT_BASELINES, key=lambda t: x[t])
        print(
            f"\n[{why}] id {x['id']}, {x['domain']}, {x['n_o200k']} o200k tokens "
            f"({x['n_gpt2']} gpt2), {x['utf8']} B UTF-8\n"
            f"  stored: plain {x['text_plain']}  pglz {x['text_pglz']}  lz4 {x['text_lz4']}  "
            f"gpt2 freq {x['tok_gpt2']}  o200k {x['tok_o200k']}  "
            f"(o200k vs best text {best}: {x[best] / x['tok_o200k']:.2f}x)\n"
            f"  text: {x['text'][:300]!r}{' ...' if len(x['text']) > 300 else ''}",
            flush=True,
        )

    print("\nexample rows", flush=True)
    examples = []
    for domain in args.domains:
        for b in args.buckets:
            g = [x for x in recs if x["domain"] == domain and x["bucket"] == b]
            g.sort(key=lambda x: min(x[t] for t in TEXT_BASELINES) / x["tok_o200k"])
            for x, why in ((g[len(g) // 2], "median"), (g[0], "worst for tokens")):
                show(x, f"{domain} {b}, {why}")
                examples.append({**x, "why": why})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"config": {**vars(args), "tokenizer": C.TOKENIZER, **settings},
             "summary": summary, "examples": examples},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
