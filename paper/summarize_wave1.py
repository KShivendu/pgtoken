#!/usr/bin/env python3
"""Print the Wave 1 headline tables from rag_e2e.py run directories.

For each run: per arm, e2e p50/p90/p99 (ms), stored bytes per row, decode share, client and
server CPU per query, at each (k, clients, sweep) config, plus the pgtoken freq verdicts against
text (lz4) and bytea raw24 with bootstrap CIs, and the Little's-law ratios.

Usage:
    python paper/summarize_wave1.py paper/runs/<run> [<run> ...] [--k 100] [--clients 1 16]
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--k", type=int, nargs="+", default=[100])
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 16])
    args = ap.parse_args()
    for run in args.runs:
        r = json.load(open(os.path.join(run, "results.json")))
        cfg0 = r["config"]
        print(f"\n### {os.path.basename(run)}: {cfg0['data']['source']}, {cfg0['data']['rows']} rows, "
              f"tokens p50 {cfg0['data']['tokens']['50']:.0f}, shared_buffers "
              f"{cfg0['settings']['shared_buffers']}")
        for res in r["results"]:
            c = res["cfg"]
            if c["k"] not in args.k or c["clients"] not in args.clients:
                continue
            print(f"\nk={c['k']}, {c['clients']} client(s), {'COLD (bodies on disk)' if c['cold'] else 'warm'}, "
                  f"CPU-cache sweep {'on' if c['sweep'] else 'off'}")
            print(f"| arm | p50 | p90 | p99 | stored B/row | decode share | client CPU ms | server CPU ms | qps | Little |")
            print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
            for key, s in res["summary"].items():
                e = s["e2e_us"]
                sb = s["body_bytes_per_row"]
                srv = s.get("server_cpu_us_p50")
                print(f"| {s['label']} | {e['p50'] / 1e3:.2f} | {e['p90'] / 1e3:.2f} | {e['p99'] / 1e3:.2f} | "
                      f"{sb if sb is None else round(sb)} | {100 * s['decode_share_p50']:.0f}% | "
                      f"{s['client_cpu_us_p50'] / 1e3:.2f} | "
                      f"{'-' if srv is None else f'{srv / 1e3:.2f}'} | {s['throughput_qps']:.0f} | "
                      f"{s['littles_ratio']:.2f} |")
            f = res["summary"].get("pgtoken_freq", {})
            for b in ("vs_text_lz4", "vs_bytea_raw24"):
                v = f.get(b)
                if v:
                    print(f"pgtoken freq {b}: {v['median_speedup']:.2f}x "
                          f"[{v['speedup_ci95'][0]:.2f}, {v['speedup_ci95'][1]:.2f}], faster in "
                          f"{v['faster_in']}/{v['n']} queries")
            br = res["summary"].get("bytea_raw24", {}).get("vs_bytea_raw24")
            pr = res["summary"].get("pgtoken_raw", {}).get("vs_bytea_raw24")
            if pr:
                print(f"pgtoken raw vs bytea raw24: {pr['median_speedup']:.2f}x "
                      f"[{pr['speedup_ci95'][0]:.2f}, {pr['speedup_ci95'][1]:.2f}]")
            if res.get("disk"):
                d = res["disk"]
                print(f"disk: {d['read_iops']:.0f} read IOPS, {d['reads_per_query']:.1f} reads/query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
