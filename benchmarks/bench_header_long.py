import sys, glob, numpy as np
# Bytes per token on long repo files: LEB128 ranks vs streamvbyte. Usage: uv run python benchmarks/bench_header_long.py
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_header_short.py")).read().split("ex = [")[0])
files = ["README.md"] + glob.glob("benchmarks/*.py")[:5] + glob.glob("core/src/*.rs")[:3]
for f in files:
    t = open(f).read(); ids = enc.encode(t, disallowed_special=()); rs = np.array([rank(i) for i in ids])
    n=len(rs); s=svb(list(rs)); l=sum(leb(int(r)) for r in rs)
    print(f"{f:40s} n={n:6d} svb={s/n:.3f} B/tok leb={l/n:.3f} B/tok  <128:{(rs<128).mean():.2f} <256:{(rs<256).mean():.2f} <16384:{(rs<16384).mean():.2f} <65536:{(rs<65536).mean():.2f}")
