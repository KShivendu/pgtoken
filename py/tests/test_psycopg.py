#!/usr/bin/env python3
"""Check `pgtoken_rs.psycopg` against a live server: INSERT, COPY and SELECT, raw and freq.

Runs in its own database, `pgtoken_pytest`, dropped and re-created each run, so it never touches
other data. Connects with the usual PG* environment variables when PGHOST is set, and otherwise
to the local server from `setup_pg.sh`.

Usage:
    uv run python py/tests/test_psycopg.py
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import psycopg
from pgtoken_rs.psycopg import register

DB = "pgtoken_pytest"
HERE = os.path.dirname(os.path.abspath(__file__))


def server_env() -> dict:
    if os.environ.get("PGHOST"):
        return {}
    script = os.path.join(HERE, "..", "..", "setup_pg.sh")
    out = subprocess.run(["bash", script, "--env"], capture_output=True, text=True, check=True)
    env = {}
    for line in out.stdout.splitlines():
        if line.startswith("export PGHOST=") or line.startswith("export PGPORT="):
            key, _, val = line[len("export ") :].partition("=")
            env[key.lower().removeprefix("pg")] = val.strip("'\"")
    return {"host": env["host"], "port": int(env["port"]), "user": "postgres"}


def connect(dbname: str) -> psycopg.Connection:
    return psycopg.connect(dbname=dbname, autocommit=True, **server_env())


def expect_error(kind, fn, what: str) -> None:
    try:
        fn()
    except kind as e:
        print(f"ok: {what} -> {type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
    else:
        raise AssertionError(f"{what}: expected {kind.__name__}")


def main() -> int:
    with connect(os.environ.get("PGDATABASE", "postgres")) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {DB}")
        admin.execute(f"CREATE DATABASE {DB}")

    rng = np.random.default_rng(7)
    docs = [rng.integers(0, 50_000, size=n, dtype=np.int64) for n in (0, 1, 7, 300, 5000)]

    with connect(DB) as conn:
        conn.execute("CREATE EXTENSION pgtoken")
        conn.execute("SELECT pgtoken.create_vocabulary('py_raw', 60000)")
        conn.execute("SELECT pgtoken.create_vocabulary('py_freq', 200019, compression => 'freq')")
        conn.execute("SELECT pgtoken.create_vocabulary('py_untrained', 300, compression => 'freq')")
        conn.execute(
            "SELECT pgtoken.train('py_freq', $$SELECT ARRAY[1,1,1,2,3,40000]::int[]$$)"
        )

        for name in ("py_raw", "py_freq"):
            vocab = register(conn, name)
            table = f"docs_{name}"
            conn.execute(f"CREATE TABLE {table} (id int PRIMARY KEY, body {vocab.type_name})")

            # INSERT: a list and a numpy array, both encoded on the client.
            for i, ids in enumerate(docs):
                param = ids.tolist() if i % 2 else ids
                conn.execute(
                    f"INSERT INTO {table} VALUES (%s, %s)", [i, vocab.encode(param)]
                )
            # COPY: plain id arrays, once the column type is set.
            with conn.cursor() as cur:
                with cur.copy(f"COPY {table} FROM STDIN (FORMAT BINARY)") as cp:
                    cp.set_types(["int4", vocab.type_name])
                    for i, ids in enumerate(docs):
                        cp.write_row((100 + i, ids))

            n = 0
            for binary in (True, False):
                rows = conn.execute(
                    f"SELECT id, body, body::int[] FROM {table} ORDER BY id", binary=binary
                ).fetchall()
                assert len(rows) == 2 * len(docs), len(rows)
                for rid, got, server in rows:
                    want = docs[rid % 100]
                    assert isinstance(got, np.ndarray) and got.dtype == np.uint32
                    assert np.array_equal(got, want), (name, rid, binary)
                    assert list(server) == want.tolist(), (name, rid, "server decode")
                    n += 1
            # COPY out, read back with the same loaders.
            with conn.cursor() as cur:
                with cur.copy(f"COPY {table} TO STDOUT (FORMAT BINARY)") as cp:
                    cp.set_types(["int4", vocab.type_name])
                    out = list(cp.rows())
            assert all(np.array_equal(b, docs[i % 100]) for i, b in out)
            print(
                f"ok: {name} ({vocab.codec}): {n} SELECT rows (binary and text) and "
                f"{len(out)} COPY rows match",
                flush=True,
            )

        # The bundled o200k preset: the ranking comes from the extension binary, not a train.
        conn.execute("SELECT pgtoken.create_vocabulary('o200k')")
        o200k = register(conn, "o200k")
        conn.execute("CREATE TABLE docs_o200k (body tokens.o200k)")
        hello = [13225, 11, 2375, 0]  # "Hello, world!" in o200k_base
        conn.execute("INSERT INTO docs_o200k VALUES (%s)", [o200k.encode(hello)])
        got, text = conn.execute(
            "SELECT body, pgtoken.text(body) FROM docs_o200k", binary=True
        ).fetchone()
        assert o200k.codec == "freq" and got.tolist() == hello and text == "Hello, world!"
        print(f"ok: o200k preset ({o200k.codec}) encodes on the client and reads {text!r}", flush=True)

        raw, freq = register(conn, "py_raw"), register(conn, "py_freq")
        expect_error(ValueError, lambda: raw.encode([1, 60000]), "id past vocab_size")
        expect_error(ValueError, lambda: raw.encode([-1]), "negative id")
        expect_error(TypeError, lambda: raw.encode([1.5]), "float id")
        expect_error(ValueError, lambda: raw.encode([[1, 2]]), "2-D ids")
        expect_error(
            psycopg.Error,
            lambda: conn.execute("INSERT INTO docs_py_raw VALUES (999, %s)", [freq.encode([1])]),
            "freq value into a raw column",
        )
        expect_error(psycopg.Error, lambda: register(conn, "nope"), "unknown vocabulary")
        expect_error(ValueError, lambda: register(conn, "py_untrained"), "untrained freq")

    with connect(DB) as other:
        register(other, "py_raw")  # only raw: a freq value cannot be decoded here
        expect_error(
            ValueError,
            lambda: other.execute("SELECT body FROM docs_py_freq LIMIT 1", binary=True).fetchone(),
            "freq read without registering it",
        )
        got = other.execute("SELECT body FROM docs_py_raw WHERE id = 2", binary=True).fetchone()[0]
        assert np.array_equal(got, docs[2])
        print("ok: a second connection reads raw after registering only raw", flush=True)

    with connect(os.environ.get("PGDATABASE", "postgres")) as admin:
        admin.execute(f"DROP DATABASE {DB}")
    print("ok: all psycopg checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
