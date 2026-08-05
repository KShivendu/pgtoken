#!/usr/bin/env python3
"""Check the Python client against the extension, both directions.

The benchmark's numbers are only meaningful if the client codec really is interoperable with
the one in the database, so this asserts it rather than assuming it:

  * bytes the extension wrote decode correctly in Python,
  * bytes Python wrote decode correctly in the extension,
  * and the two produce byte-identical output for the same input.

Usage:
    TOKEN_STORAGE_REPO=/path/to/token-storage uv run python benchmarks/test_client.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import pgcommon as C
import pgtoken_client as K

CASES = [
    [],
    [0],
    [1, 2, 3],
    [24912, 2375],
    [7] * 9,                                  # every quad remainder
    [0, 255, 256, 65535],                     # varint width boundaries
    [65536, 200018, 1000000],                 # beyond raw16
    [199999, 3, 1, 7654321],                  # ids the coding table never saw
    list(range(300)),
]


def main() -> int:
    failures: list[str] = []

    with C.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
        C.train_table(conn, "prose")
        table_path = os.path.join(
            os.path.expanduser("~/.local/share/pgtoken-pg/tables"), f"{C.FREQ_TABLE_ID}.tntt"
        )
        table = K.RankTable.load(table_path) if os.path.exists(table_path) else None
        if table is None:
            print(f"note: no coding table at {table_path}; skipping freq", file=sys.stderr)

        codecs = ["raw16", "raw24"] + (["freq"] if table else [])

        for ids in CASES:
            lit = "{" + ",".join(map(str, ids)) + "}"
            arr = np.asarray(ids, dtype=np.uint32)
            for codec in codecs:
                if codec == "raw16" and len(arr) and arr.max() > 0xFFFF:
                    continue
                tid = C.FREQ_TABLE_ID if codec == "freq" else 0
                tag = f"{codec} n={len(ids)}"

                pg_blob = bytes(
                    conn.execute(
                        "SELECT pgtoken.encode(%s::int[], %s, %s::int)", (lit, codec, tid)
                    ).fetchone()[0]
                )
                py_blob = K.encode(arr, codec, tid, table)

                # 1. identical output for identical input
                if pg_blob != py_blob:
                    failures.append(f"{tag}: encode differs (pg {len(pg_blob)}B, py {len(py_blob)}B)")

                # 2. Python decodes what the extension wrote
                got = K.decode(pg_blob, table)
                if not np.array_equal(got, arr):
                    failures.append(f"{tag}: python could not decode the extension's bytes")

                # 3. the extension decodes what Python wrote
                back = conn.execute(
                    "SELECT pgtoken.decode(%s::bytea)", (py_blob,)
                ).fetchone()[0]
                if list(back or []) != list(ids):
                    failures.append(f"{tag}: extension could not decode python's bytes")

                # 4. header-only token count agrees
                if K.token_count(pg_blob) != len(ids):
                    failures.append(f"{tag}: token_count disagrees")

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok: python client and extension agree on {len(CASES)} cases x {len(codecs)} codecs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
