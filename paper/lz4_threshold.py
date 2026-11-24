#!/usr/bin/env python3
"""When should a freq value be stored as freq_lz4? Size vs client decode time, offline.

pgtoken 0.1.0 writes `freq_lz4` whenever lz4 makes the freq payload smaller at all. That costs
decode time on every read (an lz4 block decompress before the rank lookups). This measures, on
real rows, what other rules would give. The pgtoken format is not changed: every candidate value
is a valid 0.1.0 value (plain freq, or freq_lz4), so this only asks which one a writer should pick.

    freq only                 never lz4
    lz4 when smaller          pgtoken 0.1.0's rule
    lz4 if it saves >= X%     X = 5, 10, 20 (of the freq value's size)

Sizes: the plain freq value from the pure-Python reference codec (benchmarks/pgtoken_client.py,
which always writes plain freq) and the freq_lz4 value from pgtoken_rs (the extension's own Rust
codec, which writes freq_lz4 when smaller). Decode: pgtoken_rs.decode of 100 random rows per
call, cache swept before each call (serving-cold, like rag_e2e), rules round-robin, same rows.

Usage:
    python paper/lz4_threshold.py --data /data/dbpedia /data/c4 --sample 100000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperlib as P  # noqa: E402

C = P.C
import pgtoken_client as K  # noqa: E402
import tnbench as T  # noqa: E402

from rag_e2e import Data  # noqa: E402

RULES = [("freq only", None), ("lz4 when smaller", 0.0), ("lz4 if saves >= 5%", 0.05),
         ("lz4 if saves >= 10%", 0.10), ("lz4 if saves >= 20%", 0.20)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=100000)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import pgtoken_rs as R

    out_dir = P.run_dir("lz4-threshold", args.smoke)
    print(f"run dir: {out_dir}", flush=True)
    with C.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
        row = conn.execute("SELECT id FROM pgtoken.vocabulary WHERE name = 'o200k'").fetchone()
        vid = row[0] if row else conn.execute(
            "SELECT pgtoken.create_vocabulary('o200k')").fetchone()[0]
        ranking = bytes(conn.execute(
            "SELECT ranking FROM pgtoken.vocabulary_codec('o200k')").fetchone()[0])
    tbl_rs, tbl_py = R.Table.from_bytes(ranking), K.RankTable.from_bytes(ranking)

    res = {"config": vars(args) | {"machine": P.machine()}, "datasets": {}}
    for path in args.data:
        meta_rows = len(np.load(os.path.join(path, "ids_off.npy"))) - 1
        data = Data(path, meta_rows, 0, 0, quiet=True)
        rng = np.random.default_rng(1)
        rows = np.sort(rng.choice(meta_rows, size=min(args.sample, meta_rows), replace=False))
        plain, lz, ntok = [], [], []
        t0 = time.time()
        for j, i in enumerate(rows):
            ids = data.ids(int(i))
            p = K.encode(ids, "freq", vid, tbl_py)
            r = R.encode(ids, "freq", vid, tbl_rs)
            assert np.array_equal(R.decode(p, tbl_rs), ids)
            plain.append(p)
            lz.append(r if (r[0] & 0x0F) == 4 else None)
            ntok.append(len(ids))
            if j % 20000 == 0:
                print(f"  {os.path.basename(path)}: encoded {j}/{len(rows)} ({time.time() - t0:.0f} s)",
                      flush=True)
        psize = np.array([len(x) for x in plain])
        lsize = np.array([len(x) if x is not None else 10**9 for x in lz])
        saves = 1 - lsize / psize  # negative / -inf when lz4 is not smaller
        chosen = {}
        for name, thr in RULES:
            use = np.zeros(len(rows), bool) if thr is None else (saves > thr) if thr == 0 else (saves >= thr)
            chosen[name] = [lz[j] if use[j] else plain[j] for j in range(len(rows))]
        # decode timing: same random batches for every rule, round-robin, cache swept
        dec = {name: [] for name, _ in RULES}
        brng = np.random.default_rng(2)
        for rep in range(args.reps):
            pick = brng.choice(len(rows), size=args.batch, replace=False)
            order = RULES[rep % len(RULES):] + RULES[:rep % len(RULES)]
            for name, _ in order:
                vals = [chosen[name][j] for j in pick]
                T.cache_pollute()
                t1 = time.perf_counter_ns()
                for v in vals:
                    R.decode(v, tbl_rs)
                dec[name].append((time.perf_counter_ns() - t1) / 1e3)
        lens = np.array(ntok)
        ds = {"rows_sampled": len(rows), "tokens": {p: float(np.percentile(lens, p)) for p in (25, 50, 90)},
              "rules": {}}
        print(f"\n{os.path.basename(path)}: {len(rows)} rows, o200k tokens p25/p50/p90 "
              f"{ds['tokens'][25]:.0f}/{ds['tokens'][50]:.0f}/{ds['tokens'][90]:.0f}; decode of "
              f"{args.batch} rows, cache swept, {args.reps} reps", flush=True)
        print(f"  {'rule':<22} {'rows lz4':>8} {'B/row':>8} {'B/token':>8} {'vs freq':>8} "
              f"{'decode p50 us':>14} {'p90':>7}", flush=True)
        base = None
        for name, _ in RULES:
            sz = np.array([len(v) for v in chosen[name]])
            n_lz = sum(1 for v in chosen[name] if (v[0] & 0x0F) == 4)
            r_ = {"rows_lz4": n_lz / len(rows), "bytes_per_row": float(sz.mean()),
                  "bytes_per_token": float(sz.sum() / lens.sum()),
                  "decode_us_per_batch": P.dist(dec[name])}
            base = base or r_["bytes_per_row"]
            ds["rules"][name] = r_
            print(f"  {name:<22} {100 * r_['rows_lz4']:>7.1f}% {r_['bytes_per_row']:>8.1f} "
                  f"{r_['bytes_per_token']:>8.3f} {r_['bytes_per_row'] / base:>7.3f}x "
                  f"{r_['decode_us_per_batch']['p50']:>14.1f} {r_['decode_us_per_batch']['p90']:>7.1f}",
                  flush=True)
        # what lz4 saves, distribution over rows where it is smaller
        s_ = saves[np.isfinite(saves) & (lsize < 10**9)]
        ds["lz4_saving_when_smaller"] = {p: float(np.percentile(s_, p)) for p in (10, 25, 50, 75, 90)} if s_.size else {}
        print(f"  lz4 saving where smaller (share of the freq value): "
              + ", ".join(f"p{p} {100 * v:.1f}%" for p, v in ds["lz4_saving_when_smaller"].items()),
              flush=True)
        res["datasets"][os.path.basename(path)] = ds
        P.write_json(os.path.join(out_dir, "result.json"), res)
    print(f"wrote {out_dir}/result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
