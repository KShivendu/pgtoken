#!/usr/bin/env python3
"""Check pgtoken_rs against the pure-Python reference codec, with no PostgreSQL needed.

benchmarks/test_client.py checks both clients against a live extension. This is the fast,
server-free half that CI runs: for every codec, the Rust and Python clients must produce
byte-identical values and decode each other's output back to the same ids. The one exception is
a value Rust writes as `freq_lz4` (lz4 made it smaller): Python writes plain `freq` for it, and
each must still decode the other's.

Usage:
    uv run python py/tests/test_roundtrip.py
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np
import pgtoken_rs as R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks"))
import pgtoken_client as K  # noqa: E402


def tntt(token_of_rank: list[int]) -> bytes:
    """A rank table file, laid out as core/src/tables.rs writes it."""
    return b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", len(token_of_rank)) + struct.pack(
        f"<{len(token_of_rank)}I", *token_of_rank
    )


def main() -> int:
    rng = np.random.default_rng(7)
    # Ranked tokens plus ids the ranking never saw, which fall back to rank k + id.
    ranking = [464, 13, 11, 262, 290, 286, 0, 50256]
    blob = tntt(ranking)
    tbl_rs = R.Table.from_bytes(blob)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rank.tntt")
    with open(path, "wb") as f:
        f.write(blob)
    try:
        tbl_py = K.RankTable.load(path)
    finally:
        os.remove(path)
    assert tbl_rs.k == len(ranking)

    cases = {
        "raw8": [[], [0], [255, 1, 2], rng.integers(0, 256, 512)],
        "raw16": [[], [65535, 0], rng.integers(0, 65536, 512)],
        "raw24": [[200018, 7], rng.integers(0, 1 << 24, 512)],
        "freq": [[], [464, 13, 99999], ranking * 3, rng.integers(0, 1 << 24, 512),
                 rng.choice(ranking, 512), [464, 13, 99999, 11] * 100],
    }
    n = n_lz4 = 0
    for codec, lists in cases.items():
        for ids in lists:
            arr = np.asarray(ids, dtype=np.uint32)
            rs_tbl, py_tbl = (tbl_rs, tbl_py) if codec == "freq" else (None, None)
            rs = R.encode(arr, codec, 3, rs_tbl)
            py = K.encode(arr, codec, 3, py_tbl)
            if rs[0] == 0x14:
                n_lz4 += 1
                assert py[0] == 0x12 and len(rs) < len(py), f"n={len(arr)}: freq_lz4 must be smaller"
            else:
                assert rs == py, f"{codec} n={len(arr)}: rust and python encode differ"
            assert np.array_equal(R.decode(py, rs_tbl), arr), f"{codec}: rust decode failed"
            assert np.array_equal(K.decode(rs, py_tbl), arr), f"{codec}: python decode failed"
            n += 1

    for bad in (lambda: R.encode([1], "zstd", 3), lambda: R.encode([1], "freq", 3),
                lambda: R.encode([256], "raw8", 3), lambda: R.decode(b"junk")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    assert n_lz4 >= 1, "no case exercised freq_lz4"
    print(f"ok: rust and python clients agree on {n} values ({n_lz4} stored by rust as freq_lz4), "
          f"and reject 4 bad inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
