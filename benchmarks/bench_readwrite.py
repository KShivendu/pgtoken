#!/usr/bin/env python3
"""Read and write latency: a UTF-8 `text` column vs a real `pgtoken.<vocabulary>` column.

The workload is an agent, which holds token IDs and wants token IDs back. Every path is
measured end to end from the client, because that is where the difference lives:

    write  text      IDs -> detokenize -> send text -> server compresses (pglz or lz4)
           pgtoken   IDs -> encode     -> send stored bytes (binary) -> server checks header
           server    IDs -> send int[]  -> server encodes (the `int[] -> tokens` cast)

    read   text      SELECT -> server decompresses -> wire -> client tokenizes -> IDs
           pgtoken   SELECT -> wire (stored bytes, binary) -> client decodes -> IDs
           server    SELECT body::int[] -> server decodes -> wire -> IDs

Arms (one table each, so writes never collide):

    text (pglz)                  plain text column, the default
    text (lz4)                   text COMPRESSION lz4, the strongest text baseline
    pgtoken raw, rust codec      fixed-width ids, client codec from py/ (pgtoken-core)
    pgtoken freq, rust codec     frequency-ranked varints, same Rust client codec
    pgtoken freq, python codec   the pure-Python reference codec; shows what the client costs
    pgtoken freq, server codec   no client codec at all: int[] in, int[] out

Protocol rules, each of which quietly invalidates the result if ignored:

  * Tokenize and decode are measured serving-cold: the CPU cache is swept before every timed
    call, for every arm, so a back-to-back loop cannot keep a tokenizer table or the rank
    table resident and flatter one side.
  * Arms run round-robin within each repetition, and every arm reads the same random rows in
    a repetition, so machine drift lands on all arms alike and reads can be compared pairwise.
  * Writes run with `synchronous_commit = off`. With it on, every single-row INSERT waits
    ~10 ms for a WAL flush, which swamps every difference this benchmark exists to show.
    Pass --sync-commit to measure that regime instead.
  * Pin the client to a quiet core. Absolute microseconds inflate on a loaded machine:

        taskset -c 4 uv run python benchmarks/bench_readwrite.py

Usage:
    bash setup_pg.sh --start
    taskset -c 4 uv run python benchmarks/bench_readwrite.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import pgtoken_rs as R
from psycopg.adapt import Dumper
from psycopg.pq import Format

import pgcommon as C
import pgtoken_client as K
import tnbench as T


@dataclass(frozen=True)
class Arm:
    key: str  # table suffix
    label: str
    column: str  # "text", "text_lz4", "raw", "freq"
    client: str  # "text", "rust", "python", "server"


ARMS = [
    Arm("text_pglz", "text (pglz)", "text", "text"),
    Arm("text_lz4", "text (lz4)", "text_lz4", "text"),
    Arm("raw_rs", "pgtoken raw, rust codec", "raw", "rust"),
    Arm("freq_rs", "pgtoken freq, rust codec", "freq", "rust"),
    Arm("freq_py", "pgtoken freq, python codec", "freq", "python"),
    Arm("freq_srv", "pgtoken freq, server codec", "freq", "server"),
]
BASELINE = "text_lz4"


class Tok(bytes):
    """Stored pgtoken value bytes, sent in binary as `pgtoken.tokens` (no bytea cast)."""


class Ctx:
    """Everything an arm needs to encode, decode and name its column type."""

    def __init__(self, conn, enc, raw_vid, freq_vid, ranking: bytes):
        self.enc = enc
        self.raw_codec = C.raw_codec_for(enc.n_vocab)
        self.raw_vid = raw_vid
        self.freq_vid = freq_vid
        self.tbl_py = K.RankTable.from_bytes(ranking)
        self.tbl_rs = R.Table.from_bytes(ranking)
        self.domain = {
            "raw": f"pgtoken.{C._vocab_name('raw')}",
            "freq": f"pgtoken.{C._vocab_name('freq')}",
        }

        class TokDumper(Dumper):
            format = Format.BINARY
            oid = conn.execute("SELECT 'pgtoken.tokens'::regtype::oid").fetchone()[0]

            def dump(self, obj):
                return obj

        conn.adapters.register_dumper(Tok, TokDumper)

    def coltype(self, column: str) -> str:
        return {
            "text": "text",
            "text_lz4": "text COMPRESSION lz4",
            "raw": self.domain["raw"],
            "freq": self.domain["freq"],
        }[column]

    # ── client side of a write ──
    def to_wire(self, arm: Arm, ids: np.ndarray):
        if arm.client == "text":
            return self.enc.decode(ids.tolist())
        if arm.client == "server":
            return ids.tolist()
        if arm.column == "raw":
            return Tok(R.encode(ids, self.raw_codec, self.raw_vid))
        if arm.client == "rust":
            return Tok(R.encode(ids, "freq", self.freq_vid, self.tbl_rs))
        return Tok(K.encode(ids, "freq", self.freq_vid, self.tbl_py))

    # ── client side of a read ──
    def from_wire(self, arm: Arm, value):
        if arm.client == "text":
            return self.enc.encode(value)
        if arm.client == "server":
            return value
        if arm.column == "raw":
            return R.decode(value)
        if arm.client == "rust":
            return R.decode(value, self.tbl_rs)
        return K.decode(bytes(value), self.tbl_py)


def wire_len(x) -> int:
    if isinstance(x, (bytes, memoryview)):
        return len(x)
    if isinstance(x, str):
        return len(x.encode())
    return 4 * len(x)  # int[]: four bytes an id on the wire, plus a small header


def create_table(conn, ctx: Ctx, name: str, arm: Arm) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"CREATE TABLE {name} (id integer PRIMARY KEY, body {ctx.coltype(arm.column)})")


def copy_rows(conn, name: str, arm: Arm, rows) -> None:
    """Bulk-load `(id, wire value)` rows the way a client of this arm would."""
    with conn.cursor() as cur:
        if arm.client == "server":
            # Text COPY: the server parses '{1,2,3}' with the tokens input function and
            # encodes. Binary COPY would hand int4[] bytes to tokens_recv, which is wrong.
            with cur.copy(f"COPY {name} (id, body) FROM STDIN") as cp:
                for row in rows:
                    cp.write_row(row)
        else:
            with cur.copy(f"COPY {name} (id, body) FROM STDIN (FORMAT BINARY)") as cp:
                # bytea's binary dumper sends the bytes untouched; the server reads them with
                # the column's own receive function (tokens_recv for a tokens column).
                cp.set_types(["int4", "text" if arm.client == "text" else "bytea"])
                for row in rows:
                    cp.write_row(row)


def insert_sql(name: str, arm: Arm) -> str:
    val = "%s::int4[]" if arm.client == "server" else "%s"
    return f"INSERT INTO {name} (id, body) VALUES (%s, {val})"


def pct(a, p):
    return float(np.percentile(np.asarray(a, dtype=np.float64), p))


def summary(total, parts: dict) -> dict:
    out = {
        "n": len(total),
        "p50": pct(total, 50),
        "p90": pct(total, 90),
        "p99": pct(total, 99),
        "mean": float(np.mean(total)),
    }
    for k, v in parts.items():
        out[f"{k}_p50"] = pct(v, 50)
    return out


# ── writes ──────────────────────────────────────────────────────────────────────────────


def bench_write(conn, ctx: Ctx, id_lists, batch: int, reps: int, warmup: int) -> dict:
    """Time an agent writing `batch` rows from token IDs it already holds."""
    names = {a.key: f"rw_w_{a.key}" for a in ARMS}
    for a in ARMS:
        create_table(conn, ctx, names[a.key], a)
    t = {a.key: {"total": [], "client": [], "send": [], "bytes": []} for a in ARMS}

    next_id = 0
    for r in range(warmup + reps):
        lo = (r * batch) % (len(id_lists) - batch)
        ids_batch = id_lists[lo : lo + batch]
        for a in ARMS:
            T.cache_pollute()
            t0 = time.perf_counter()
            vals = [ctx.to_wire(a, ids) for ids in ids_batch]
            t1 = time.perf_counter()
            if batch == 1:
                conn.execute(insert_sql(names[a.key], a), (next_id, vals[0]))
            else:
                copy_rows(conn, names[a.key], a, zip(range(next_id, next_id + batch), vals))
            t2 = time.perf_counter()
            if r >= warmup:
                t[a.key]["total"].append((t2 - t0) * 1e6 / batch)
                t[a.key]["client"].append((t1 - t0) * 1e6 / batch)
                t[a.key]["send"].append((t2 - t1) * 1e6 / batch)
                t[a.key]["bytes"].append(sum(wire_len(v) for v in vals) / batch)
        next_id += batch
        if (r + 1) % 50 == 0:
            line = "  ".join(
                f"{a.key} {pct(t[a.key]['total'], 50):.0f}" for a in ARMS if t[a.key]["total"]
            )
            print(f"    write batch {batch}: rep {r + 1}/{warmup + reps}  p50 us/row: {line}", flush=True)

    out = {}
    for a in ARMS:
        s = summary(t[a.key]["total"], {"client": t[a.key]["client"], "send": t[a.key]["send"]})
        s["wire_bytes_per_row"] = float(np.median(t[a.key]["bytes"]))
        s["raw_total_us"] = t[a.key]["total"]
        out[a.key] = s
    out["_check"] = check_written(conn, ctx, names, id_lists, batch, warmup + reps)
    return out


def check_written(conn, ctx: Ctx, names, id_lists, batch, n_reps) -> dict:
    """Read every written row back and compare with the IDs the agent started from.

    For pgtoken this must be exact. For text it measures the round trip
    detokenize -> store -> tokenize, which is not guaranteed to give the same IDs back.
    """
    expect = {}
    nid = 0
    for r in range(n_reps):
        lo = (r * batch) % (len(id_lists) - batch)
        for ids in id_lists[lo : lo + batch]:
            expect[nid] = ids
            nid += 1
    res = {}
    for a in ARMS:
        if a.client == "text":
            got = {i: ctx.enc.encode(s) for i, s in conn.execute(f"SELECT id, body FROM {names[a.key]}")}
        else:
            got = dict(conn.execute(f"SELECT id, body::int[] FROM {names[a.key]}").fetchall())
        bad = [i for i, ids in expect.items() if list(got.get(i, [])) != ids.tolist()]
        ex = None
        if bad:
            i = bad[0]
            want, have = expect[i].tolist(), list(got.get(i, []))
            j = next((k for k in range(min(len(want), len(have))) if want[k] != have[k]), min(len(want), len(have)))
            ex = {
                "row": i,
                "first_diff_at": j,
                "ids_written": want[max(0, j - 2) : j + 4],
                "ids_read_back": have[max(0, j - 2) : j + 4],
                "text_around": ctx.enc.decode(want[max(0, j - 4) : j + 6]),
            }
        res[a.key] = {"rows": len(expect), "mismatched": len(bad), "example": ex}
    return res


# ── reads ───────────────────────────────────────────────────────────────────────────────


def load_read_tables(conn, ctx: Ctx, texts, id_lists) -> dict:
    """One loaded table per column kind; arms that share a column read the same table."""
    tables = {}
    for a in ARMS:
        if a.column in tables:
            continue
        name = f"rw_r_{a.column}"
        create_table(conn, ctx, name, a)
        loader = a if a.client != "server" else Arm("", "", a.column, "rust")
        if loader.client == "text":
            rows = list(enumerate(texts))
        else:
            rows = [(i, ctx.to_wire(loader, ids)) for i, ids in enumerate(id_lists)]
        copy_rows(conn, name, loader, rows)
        conn.execute(f"VACUUM (ANALYZE) {name}")
        conn.execute(f"SELECT count(*) FROM {name} WHERE body IS NOT NULL")  # warm buffers
        tables[a.column] = name
    return tables


def bench_read(conn, ctx: Ctx, tables, id_lists, fanout: int, reps: int, warmup: int, seed: int) -> dict:
    """Time an agent reading `fanout` rows and getting token IDs back."""
    rng = np.random.default_rng(seed + fanout)
    t = {a.key: {"total": [], "fetch": [], "codec": [], "bytes": []} for a in ARMS}
    mismatches = {a.key: 0 for a in ARMS}
    with conn.cursor(binary=True) as cur:
        for r in range(warmup + reps):
            picks = rng.choice(len(id_lists), size=fanout, replace=False).tolist()
            for a in ARMS:
                expr = "body::int[]" if a.client == "server" else "body"
                sql = f"SELECT id, {expr} FROM {tables[a.column]} WHERE id = ANY(%s)"
                T.cache_pollute()
                t0 = time.perf_counter()
                cur.execute(sql, (picks,))
                got = cur.fetchall()
                t1 = time.perf_counter()
                decoded = [ctx.from_wire(a, v) for _, v in got]
                t2 = time.perf_counter()
                for (i, _), ids in zip(got, decoded):
                    if not np.array_equal(np.asarray(ids, dtype=np.uint32), id_lists[i]):
                        mismatches[a.key] += 1
                if r >= warmup:
                    t[a.key]["total"].append((t2 - t0) * 1e6)
                    t[a.key]["fetch"].append((t1 - t0) * 1e6)
                    t[a.key]["codec"].append((t2 - t1) * 1e6)
                    t[a.key]["bytes"].append(sum(wire_len(v) for _, v in got))
            if (r + 1) % 50 == 0:
                line = "  ".join(
                    f"{a.key} {pct(t[a.key]['total'], 50):.0f}" for a in ARMS if t[a.key]["total"]
                )
                print(f"    read {fanout} rows: rep {r + 1}/{warmup + reps}  p50 us: {line}", flush=True)

    out = {}
    for a in ARMS:
        s = summary(t[a.key]["total"], {"fetch": t[a.key]["fetch"], "codec": t[a.key]["codec"]})
        s["wire_bytes"] = int(np.median(t[a.key]["bytes"]))
        s["mismatched_rows"] = mismatches[a.key]
        s["raw_total_us"] = t[a.key]["total"]
        out[a.key] = s
    return out


def paired(res: dict, a: str, b: str) -> dict:
    """Per-repetition head to head: same rows, same cache state, so a fair pairwise test."""
    x, y = np.asarray(res[a]["raw_total_us"]), np.asarray(res[b]["raw_total_us"])
    wins, losses = int((x < y).sum()), int((x > y).sum())
    n = wins + losses
    # Two-sided sign test, normal approximation (n is in the hundreds).
    z = (wins - n / 2) / np.sqrt(n / 4) if n else 0.0
    ratio = y / x
    boot = np.random.default_rng(0).choice(ratio, size=(2000, len(ratio)))
    lo, hi = np.percentile(np.median(boot, axis=1), [2.5, 97.5])
    return {"wins": wins, "losses": losses, "z": float(z), "median_speedup": float(np.median(ratio)),
            "speedup_ci95": [float(lo), float(hi)]}


# ── per-document examples ───────────────────────────────────────────────────────────────


def examples(conn, ctx: Ctx, tables, texts, id_lists) -> list[dict]:
    """Shortest, median and longest documents: stored size per column and cold codec cost."""
    order = np.argsort([len(t.encode()) for t in texts])
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    out = []
    for i in picks:
        sizes = {
            col: conn.execute(f"SELECT pg_column_size(body) FROM {name} WHERE id = %s", (i,)).fetchone()[0]
            for col, name in tables.items()
        }
        blob_f = R.encode(id_lists[i], "freq", ctx.freq_vid, ctx.tbl_rs)

        def cold(fn, n=50):
            ts = []
            for _ in range(n):
                T.cache_pollute()
                t0 = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t0) * 1e6)
            return float(np.median(ts))

        out.append({
            "row": i,
            "text_start": texts[i][:70],
            "tokens": int(len(id_lists[i])),
            "utf8_bytes": len(texts[i].encode()),
            "stored_bytes": sizes,
            "tokenize_us": cold(lambda: ctx.enc.encode(texts[i])),
            "rust_freq_decode_us": cold(lambda: R.decode(blob_f, ctx.tbl_rs)),
            "python_freq_decode_us": cold(lambda: K.decode(blob_f, ctx.tbl_py)),
        })
    return out


# ── main ────────────────────────────────────────────────────────────────────────────────


def machine() -> dict:
    cpu = "?"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip())
    return {"cpu": cpu, "host": platform.node(), "git": sha + ("-dirty" if dirty else ""),
            "affinity": sorted(os.sched_getaffinity(0)), "loadavg": os.getloadavg()}


def print_table(title: str, res: dict, parts: tuple[str, str], bytes_key: str) -> None:
    print(f"\n{title}")
    hdr = (f"  {'arm':<28} {'p50':>9} {'p90':>9} {'p99':>9}   {parts[0] + ' p50':>11} "
           f"{parts[1] + ' p50':>11} {'bytes':>8} {'vs lz4':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    base = res[BASELINE]["p50"]
    for a in ARMS:
        r = res[a.key]
        print(f"  {a.label:<28} {r['p50']:>9.1f} {r['p90']:>9.1f} {r['p99']:>9.1f}   "
              f"{r[parts[0] + '_p50']:>11.1f} {r[parts[1] + '_p50']:>11.1f} "
              f"{r[bytes_key]:>8.0f} {base / r['p50']:>6.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--domain", default="prose")
    ap.add_argument("--fanout", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 100])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sync-commit", action="store_true",
                    help="keep synchronous_commit on, so each write waits for a WAL flush")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "readwrite_results.json"),
    )
    args = ap.parse_args()

    texts = C.load_corpus(args.domain, args.docs)
    enc = C.encoder()
    id_lists = [np.asarray(x, dtype=np.uint32) for x in C.tokenize_all(texts)]
    n_tok = np.array([len(x) for x in id_lists])
    print(f"corpus: {len(texts)} docs from {args.domain}, {enc.name} (vocab {enc.n_vocab}), "
          f"tokens/doc p50 {int(np.median(n_tok))}, min {n_tok.min()}, max {n_tok.max()}", flush=True)

    with C.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
        if not args.sync_commit:
            conn.execute("SET synchronous_commit = off")
        raw_vid = C.ensure_raw_vocab(conn)
        freq_vid, _ = C.ensure_freq_vocab(conn, args.domain)
        ctx = Ctx(conn, enc, raw_vid, freq_vid, C.ranking_bytes(conn, C._vocab_name("freq")))
        settings = {
            k: conn.execute(f"SHOW {k}").fetchone()[0]
            for k in ("server_version", "default_toast_compression", "shared_buffers",
                      "synchronous_commit")
        }
        print(f"settings: {settings}", flush=True)

        writes = {}
        for b in args.batch:
            print(f"  write, batch {b}...", flush=True)
            writes[b] = bench_write(conn, ctx, id_lists, b, args.reps, args.warmup)
            print_table(f"WRITE, batch {b}: us per row", writes[b], ("client", "send"),
                        "wire_bytes_per_row")

        tables = load_read_tables(conn, ctx, texts, id_lists)
        reads = {}
        for f in args.fanout:
            print(f"  read, {f} rows per query...", flush=True)
            reads[f] = bench_read(conn, ctx, tables, id_lists, f, args.reps, args.warmup, args.seed)
            print_table(f"READ, {f} rows per query: us per query", reads[f], ("fetch", "codec"),
                        "wire_bytes")
        ex = examples(conn, ctx, tables, texts, id_lists)

    head2head = {
        "read": {f: {a.key: paired(res, a.key, BASELINE) for a in ARMS if a.key != BASELINE}
                 for f, res in reads.items()},
        "write": {b: {a.key: paired(res, a.key, BASELINE) for a in ARMS if a.key != BASELINE}
                  for b, res in writes.items()},
    }

    results = {
        "config": {
            "docs": len(texts),
            "domain": args.domain,
            "tokenizer": enc.name,
            "n_vocab": enc.n_vocab,
            "raw_codec": ctx.raw_codec,
            "reps": args.reps,
            "warmup": args.warmup,
            "arms": [a.__dict__ for a in ARMS],
            "machine": machine(),
            **settings,
        },
        "write": writes,
        "read": reads,
        "head_to_head_vs_text_lz4": head2head,
        "examples": ex,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)

    print(f"\nPostgreSQL {settings['server_version'].split()[0]}, synchronous_commit="
          f"{settings['synchronous_commit']}, client on CPUs {sorted(os.sched_getaffinity(0))}")
    for kind, h in head2head.items():
        for size, arms in h.items():
            for k, v in arms.items():
                print(f"  {kind} {size}: {k} vs text_lz4: faster in {v['wins']}/{v['wins'] + v['losses']} "
                      f"reps, median speedup {v['median_speedup']:.2f}x "
                      f"[{v['speedup_ci95'][0]:.2f}, {v['speedup_ci95'][1]:.2f}]")
    for b, w in writes.items():
        for k, c in w["_check"].items():
            print(f"  write check, batch {b}, {k}: {c['mismatched']}/{c['rows']} rows read back "
                  f"with different IDs" + (f"; e.g. {c['example']}" if c["example"] else ""))
    for f, res in reads.items():
        bad = {k: v["mismatched_rows"] for k, v in res.items() if v["mismatched_rows"]}
        print(f"  read check, {f} rows: " + (f"MISMATCHES {bad}" if bad else "every arm returned the stored IDs"))
    for e in ex:
        print(f"  example row {e['row']}: {e['tokens']} tokens, {e['utf8_bytes']} B utf8, "
              f"stored {e['stored_bytes']}, tokenize {e['tokenize_us']:.1f} us, "
              f"rust freq decode {e['rust_freq_decode_us']:.1f} us, "
              f"python {e['python_freq_decode_us']:.1f} us: {e['text_start']!r}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
