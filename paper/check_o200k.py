#!/usr/bin/env python3
"""Which o200k tokenizer should the text arms use? Check ID-for-ID equality, then time it.

The text arms of every paper benchmark pay a tokenize on read. To make the text baseline as
strong as possible, they must use the fastest o200k tokenizer that gives exactly tiktoken's
o200k_base ids. Candidates:

    tiktoken o200k_base                     the reference (OpenAI's own)
    HF tokenizers, Xenova/gpt-4o            tokenizer.json converted from o200k_base
    HF tokenizers, openai/gpt-oss-20b       o200k_harmony: o200k_base plus extra special tokens

Equality: every candidate encodes the same real chunks (C4 English, code, Hindi; 512 o200k
tokens each, from pgcommon.o200k_chunks) and must return tiktoken's ids exactly. The first
differing position is printed for any mismatch.

Speed, in ns per token, single thread, one 512-token chunk per call, output as a numpy uint32
array (what the RAG client hands to a model):
    warm         back-to-back calls, tables stay in cache (flattering)
    cold         64 MB cache sweep before every call (serving-cold, as bench_readwrite does)
    batch-100    100 chunks in one call with the library's own threads (all cores)

Usage:
    uv run python paper/check_o200k.py [--chunks 600] [--smoke]
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
import tnbench as T  # noqa: E402


def candidates() -> dict:
    out = {}
    tt = C._tiktoken("o200k_base", "tiktoken-o200k").tt
    out["tiktoken o200k_base"] = {
        "one": lambda s: tt.encode_to_numpy(s, disallowed_special=()).astype(np.uint32, copy=False),
        "batch": lambda xs: tt.encode_ordinary_batch(xs, num_threads=os.cpu_count()),
        "source": f"tiktoken {__import__('tiktoken').__version__}",
    }
    import tokenizers

    for repo, rev in C.O200K_HF.items():
        path = C.hf_tokenizer_json(repo)
        hf = C._hf_file(path, repo, n_vocab=200019).hf

        def one(s, hf=hf):
            return np.asarray(hf.encode(s, add_special_tokens=False).ids, dtype=np.uint32)

        def batch(xs, hf=hf):
            return [e.ids for e in hf.encode_batch(xs, add_special_tokens=False)]

        out[f"HF {repo}"] = {"one": one, "batch": batch,
                             "source": f"tokenizers {tokenizers.__version__}, {repo}@{rev[:10]}",
                             "path": path}
    return out


def first_diff(a, b) -> int:
    n = min(len(a), len(b))
    d = np.nonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0]
    return int(d[0]) if d.size else n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=600, help="English chunks (code, Hindi: half each)")
    ap.add_argument("--timing-chunks", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_dir = P.run_dir("o200k-check", args.smoke)
    print(f"run dir: {out_dir}", flush=True)
    sets = {}
    for dom, n in (("prose", args.chunks), ("code", args.chunks // 2), ("hindi", args.chunks // 2)):
        texts, ids = C.o200k_chunks(dom)
        sets[dom] = (texts[:n], ids[:n])
        print(f"  {dom}: {min(n, len(texts))} chunks of {int(np.median([len(x) for x in ids]))} "
              f"o200k tokens (of {len(texts)} available)", flush=True)

    cands = candidates()
    ref = "tiktoken o200k_base"
    res = {"config": {"chunks": {d: len(v[0]) for d, v in sets.items()},
                      "machine": P.machine()}, "candidates": {}}

    # ── equality ──
    for name, c in cands.items():
        eq = {}
        for dom, (texts, ids) in sets.items():
            bad = []
            for i, (t, want) in enumerate(zip(texts, ids)):
                got = c["one"](t)
                if not np.array_equal(got, want):
                    j = first_diff(got, want)
                    bad.append({"chunk": i, "first_diff_at": j, "got": got[j:j + 4].tolist(),
                                "want": want[j:j + 4].tolist(), "text": C._tiktoken(
                                    "o200k_base", "x").tt.decode(want[max(0, j - 3):j + 4].tolist())})
            eq[dom] = {"chunks": len(texts), "mismatched": len(bad), "examples": bad[:3]}
            print(f"  {name:<28} {dom:<6} {len(texts) - len(bad)}/{len(texts)} chunks ID-for-ID "
                  f"equal to {ref}" + (f"; first: {bad[0]}" if bad else ""), flush=True)
        res["candidates"][name] = {"source": c["source"], "equality": eq}

    # ── speed (English only, the paper's main corpus) ──
    texts, ids = sets["prose"]
    texts, ids = texts[: args.timing_chunks], ids[: args.timing_chunks]
    n_tok = np.array([len(x) for x in ids], dtype=np.float64)
    order = list(cands)
    warm = {k: [] for k in order}
    cold = {k: [] for k in order}
    for i, t in enumerate(texts):
        for k in order[i % len(order):] + order[: i % len(order)]:  # rotate start
            f = cands[k]["one"]
            T.cache_pollute()
            t0 = time.perf_counter_ns()
            f(t)
            cold[k].append((time.perf_counter_ns() - t0) / n_tok[i])
    for k in order:
        f = cands[k]["one"]
        for _ in range(2):
            for i, t in enumerate(texts):
                t0 = time.perf_counter_ns()
                f(t)
                if _:
                    warm[k].append((time.perf_counter_ns() - t0) / n_tok[i])
    batch = {k: [] for k in order}
    for rep in range(10):
        for k in order:
            lo = (rep * 100) % max(1, len(texts) - 100)
            xs = texts[lo:lo + 100]
            T.cache_pollute()
            t0 = time.perf_counter_ns()
            cands[k]["batch"](xs)
            batch[k].append((time.perf_counter_ns() - t0) / n_tok[lo:lo + 100].sum())

    print(f"\nns per token, {len(texts)} English chunks of 512 o200k tokens, "
          f"{os.cpu_count()} cores ({P.machine()['cpu']})", flush=True)
    print(f"  {'tokenizer':<28} {'equal':>11} {'cold p50':>9} {'cold p90':>9} {'warm p50':>9} "
          f"{'batch-100 p50':>14}   us/chunk cold p50", flush=True)
    for k in order:
        eq = res["candidates"][k]["equality"]
        ok = sum(v["chunks"] - v["mismatched"] for v in eq.values())
        tot = sum(v["chunks"] for v in eq.values())
        s = {"cold": P.dist(cold[k]), "warm": P.dist(warm[k]), "batch100": P.dist(batch[k])}
        res["candidates"][k]["ns_per_token"] = s
        print(f"  {k:<28} {ok:>5}/{tot:<5} {s['cold']['p50']:>9.1f} {s['cold']['p90']:>9.1f} "
              f"{s['warm']['p50']:>9.1f} {s['batch100']['p50']:>14.1f}   "
              f"{s['cold']['p50'] * 512 / 1000:.1f}", flush=True)

    exact = [k for k in order if all(v["mismatched"] == 0
                                     for v in res["candidates"][k]["equality"].values())]
    best = min(exact, key=lambda k: res["candidates"][k]["ns_per_token"]["cold"]["p50"])
    res["fastest_exact"] = best
    # Both HF files run the same library code (gpt-oss only adds special tokens), so any HF
    # winner maps to hf-o200k, which loads the Xenova/gpt-4o file.
    res["pgtoken_tokenizer_env"] = "tiktoken-o200k" if best.startswith("tiktoken") else "hf-o200k"
    print(f"\nfastest tokenizer with identical ids (cold p50): {best} "
          f"-> PGTOKEN_TOKENIZER={res['pgtoken_tokenizer_env']}", flush=True)
    P.write_json(os.path.join(out_dir, "result.json"), res)
    print(f"wrote {out_dir}/result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
