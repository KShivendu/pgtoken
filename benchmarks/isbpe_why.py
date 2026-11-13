#!/usr/bin/env python3
"""Why does ISBPE's ~10% token saving on code shrink to ~6% of stored bytes?

Splits the stored size of the 16 KB code values from isbpe_storage.py into parts:
  - tokens by LEB128 width (rank < 128 costs 1 byte, < 16384 costs 2, else 3)
  - plain freq bytes (no lz4)
  - what lz4 then saves
And shows one line tokenized both ways, with each token's stored width.

Usage: same as isbpe_storage.py, with this file's name.
"""

from __future__ import annotations

import numpy as np

import isbpe_storage as S
import pgtoken_rs._native as pg

RESEARCH = S.RESEARCH


def widths(ids, table_ranks):
    r = np.array([table_ranks.get(int(i), 10**9) for i in ids])
    return np.where(r < 128, 1, np.where(r < 16384, 2, 3))


def main() -> int:
    dom = "code"
    train = S.utf8_prefix(open(f"{RESEARCH}/data/code_tr.txt", "rb").read(), S.TRAIN_BYTES)
    test = open(f"{RESEARCH}/data/code_te.txt", "rb").read()
    toks, _ = S.tokenizers_for(dom)
    toks.pop("o200k")
    import isbpe_rs
    from tokenizers import Tokenizer
    isb = isbpe_rs.ISBPE.load(f"{RESEARCH}/models/exp138/isbpe_code_dual_65536.json")
    bpe = Tokenizer.from_file(f"{RESEARCH}/models/exp138/bpe_code_same_65536.json")
    show = {"bpe": lambda i: bpe.id_to_token(i).replace("Ġ", " ").replace("Ċ", "\\n"),
            "isbpe": lambda i: str(isb[i]).replace("\n", "\\n")}

    ranks, tables = {}, {}
    for name, fn in toks.items():
        ids = S.encode_big(fn, train)
        vals, counts = np.unique(ids, return_counts=True)
        order = np.lexsort((vals, -counts))
        ranks[name] = {int(t): r for r, t in enumerate(vals[order])}
        tables[name] = S.rank_table(ids)
        print(f"{name} ranking ready", flush=True)

    vals = S.chunks(test, 16384, 60)
    print(f"\n60 code values of 16 KB, totals per value:")
    print(f"{'':6} {'tokens':>7} {'1-byte':>7} {'2-byte':>7} {'3-byte':>7} {'freq B':>7} {'lz4 B':>7}")
    per = {}
    for name, fn in toks.items():
        t = w1 = w2 = w3 = fb = lb = 0
        for v in vals:
            ids = np.asarray(fn(v), dtype=np.uint32)
            w = widths(ids, ranks[name])
            t += len(ids); w1 += (w == 1).sum(); w2 += (w == 2).sum(); w3 += (w == 3).sum()
            fb += w.sum()
            lb += len(pg.encode(ids, "freq", 1, tables[name]))
        n = len(vals)
        per[name] = (t / n, fb / n, lb / n)
        print(f"{name:6} {t/n:7.0f} {w1/n:7.0f} {w2/n:7.0f} {w3/n:7.0f} {fb/n:7.0f} {lb/n:7.0f}",
              flush=True)

    # Token strings that BPE uses and ISBPE removes: count by string, over all values.
    from collections import Counter
    cb, ci = Counter(), Counter()
    for v in vals:
        cb.update(show["bpe"](i) for i in toks["bpe"](v))
        ci.update(show["isbpe"](i) for i in toks["isbpe"](v))
    print("\nmost-used tokens, per 16 KB value (string, uses, stored width)")
    for name, c in (("bpe", cb), ("isbpe", ci)):
        print(f"  {name}:")
        for s, k in c.most_common(12):
            print(f"    {s!r:16} {k/len(vals):6.0f}", flush=True)

    line = b"        if key_in_self == self._KEYS.ALIAS:\n            return self.items\n"
    print(f"\nexample {line.decode()!r}")
    for name, fn in toks.items():
        ids = fn(line)
        w = widths(ids, ranks[name])
        print(f"  {name:6} {len(ids)} tokens, {w.sum()} bytes: " +
              " | ".join(f"{show[name](i)!r}:{x}" for i, x in zip(ids, w)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
