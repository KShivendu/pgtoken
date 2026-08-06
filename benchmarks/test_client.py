#!/usr/bin/env python3
"""Check the Python client against the extension, both directions.

The benchmark's numbers are only meaningful if the client codec really is interoperable with
the one in the database, so this asserts it rather than assuming it:

  * bytes the extension wrote decode correctly in Python,
  * bytes Python wrote decode correctly in the extension,
  * and the two produce byte-identical output for the same input.

A codec is no longer nameable directly (see `pgtoken.tokens`): every value names a vocabulary,
and the vocabulary's `vocab_size` is what picks the width under test. So each parity case runs
against a small, disposable vocabulary sized for the codec it exercises -- 256 for raw8, 60000
for raw16, 200019 for raw24 -- plus one `freq` vocabulary trained on a tiny synthetic corpus, so
this file needs nothing beyond a live server, psycopg and numpy.

Usage:
    uv run python benchmarks/test_client.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import psycopg

import pgtoken_client as K

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    [],
    [0],
    [1, 2, 3],
    [24912, 2375],
    [7] * 9,  # every quad remainder
    [0, 255, 256, 65535],  # varint width boundaries
    [65536, 200018, 1000000],  # beyond raw16
    [199999, 3, 1, 7654321],  # ids the coding table never saw
    list(range(300)),
]

# (codec, vocabulary name, vocab_size). vocab_size is what selects the width under test now,
# since a codec is no longer nameable directly -- every value names its vocabulary instead.
RAW_VOCABS = [
    ("raw8", "parity_raw8", 256),
    ("raw16", "parity_raw16", 60000),
    ("raw24", "parity_raw24", 200019),
]

FREQ_VOCAB = "parity_freq"
# Large enough that no CASES entry is out of range, so freq is exercised against the same cases
# as the raw codecs rather than a narrower set of its own.
FREQ_VOCAB_SIZE = 16_777_216
# A small, self-contained corpus: just enough to make some CASES ids "seen" (ranked) while
# others -- 199999, 1000000, 7654321, ... -- stay unseen and fall back to rank `k + id`, which
# is the whole property `freq` exists to prove.
FREQ_TRAIN_IDS = [7, 3, 1, 24912, 2375, 0, 255, 256] * 40


def connect(**kwargs) -> psycopg.Connection:
    """Connect to the local no-root PostgreSQL set up by `setup_pg.sh`."""
    script = os.path.join(os.path.dirname(HERE), "setup_pg.sh")
    out = subprocess.run(["bash", script, "--env"], capture_output=True, text=True, check=True).stdout
    env = {}
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        key, _, val = line[len("export ") :].partition("=")
        env[key] = val.strip().strip("'\"")
    return psycopg.connect(
        host=env["PGHOST"],
        port=int(env["PGPORT"]),
        user=env.get("PGUSER", "postgres"),
        dbname=os.environ.get("PGDATABASE", "postgres"),
        autocommit=True,
        **kwargs,
    )


def ensure_vocabulary(conn, name: str, vocab_size: int, compression: str = "raw") -> int:
    """Create `name` if it does not already exist, and return its id.

    A vocabulary's name is permanently reserved once created, even past a drop, so re-running
    this script against the same database has to notice an existing row rather than retry the
    create and hit "already exists".
    """
    row = conn.execute("SELECT id FROM pgtoken.vocabulary WHERE name = %s", (name,)).fetchone()
    if row is not None:
        return row[0]
    return conn.execute(
        "SELECT pgtoken.create_vocabulary(%s, %s, compression => %s)",
        (name, vocab_size, compression),
    ).fetchone()[0]


def ensure_freq_table(conn, name: str, vocab_size: int) -> tuple[int, K.RankTable | None]:
    """Create and train `name` if needed, and return its id and ranking (if one loaded).

    `pgtoken.train` is write-once, so this only trains when `vocabulary_info` reports no
    ranking yet -- re-running the script must not attempt to retrain.
    """
    vocab_id = ensure_vocabulary(conn, name, vocab_size, compression="freq")
    ranked = conn.execute("SELECT ranked FROM pgtoken.vocabulary_info(%s)", (name,)).fetchone()[0]
    if ranked is None:
        lit = "{" + ",".join(map(str, FREQ_TRAIN_IDS)) + "}"
        conn.execute("SELECT pgtoken.train(%s, %s)", (name, f"SELECT '{lit}'::int[]"))

    table_dir = conn.execute("SELECT current_setting('pgtoken.table_dir')").fetchone()[0]
    path = os.path.join(table_dir, f"{vocab_id}.tntt")
    if not os.path.exists(path):
        return vocab_id, None
    return vocab_id, K.RankTable.load(path)


def main() -> int:
    failures: list[str] = []

    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")

        vocabs: list[tuple[str, str, int, int]] = []  # (codec, name, vocab_size, vocabulary_id)
        for codec, name, vocab_size in RAW_VOCABS:
            vocab_id = ensure_vocabulary(conn, name, vocab_size)
            vocabs.append((codec, name, vocab_size, vocab_id))

        table: K.RankTable | None = None
        try:
            freq_id, table = ensure_freq_table(conn, FREQ_VOCAB, FREQ_VOCAB_SIZE)
        except psycopg.Error as e:
            print(f"note: could not train {FREQ_VOCAB!r} ({e}); skipping freq", file=sys.stderr)
        else:
            if table is None:
                print(f"note: no coding table for {FREQ_VOCAB!r}; skipping freq", file=sys.stderr)
            else:
                vocabs.append(("freq", FREQ_VOCAB, FREQ_VOCAB_SIZE, freq_id))

        n_cases = 0
        for ids in CASES:
            arr = np.asarray(ids, dtype=np.uint32)
            lit = "{" + ",".join(map(str, ids)) + "}"

            for codec, vocab, vocab_size, vocab_id in vocabs:
                if len(arr) and arr.max() >= vocab_size:
                    continue  # out of this vocabulary's declared range
                n_cases += 1
                tag = f"{codec} n={len(ids)}"
                rank_table = table if codec == "freq" else None

                pg_blob = bytes(
                    conn.execute(
                        f"SELECT %s::pgtoken.tokens('{vocab}')::bytea", (lit,)
                    ).fetchone()[0]
                )
                py_blob = K.encode(arr, codec, vocab_id, rank_table)

                # 1. identical output for identical input
                if pg_blob != py_blob:
                    failures.append(f"{tag}: encode differs (pg {len(pg_blob)}B, py {len(py_blob)}B)")

                # 2. Python decodes what the extension wrote
                got = K.decode(pg_blob, rank_table)
                if not np.array_equal(got, arr):
                    failures.append(f"{tag}: python could not decode the extension's bytes")

                # 3. the extension decodes what Python wrote
                back = conn.execute("SELECT %s::pgtoken.tokens::int[]", (py_blob,)).fetchone()[0]
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
    print(f"ok: python client and extension agree on {n_cases} (case, codec) combinations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
