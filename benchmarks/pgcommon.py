"""Shared plumbing for the Postgres harness: connection, corpus loading, table training.

Kept separate so bench_latency.py and bench_storage_wal.py load an identical corpus into
identical tables. If the two benchmarks disagreed about the data, neither number would mean
anything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import psycopg
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixture")

# The benchmarks need the held-out corpora and the reference codec implementations from the
# token-storage repo, which is where the paper's numbers come from. That repo is not vendored
# here: its corpora run to ~110 MB, and it is the authority on the reference behaviour, so
# copying either would invite drift. The extension itself has no such dependency.
TOKEN_STORAGE = os.environ.get(
    "TOKEN_STORAGE_REPO", os.path.expanduser("~/projects/token-storage")
)
if not os.path.isfile(os.path.join(TOKEN_STORAGE, "tnbench.py")):
    raise SystemExit(
        f"cannot find tnbench.py under {TOKEN_STORAGE}\n"
        "These benchmarks need the token-storage repo for its corpora and reference codecs:\n"
        "  git clone https://github.com/KShivendu/token-storage.git\n"
        "  TOKEN_STORAGE_REPO=/path/to/token-storage uv run python benchmarks/bench_latency.py"
    )

sys.path.insert(0, TOKEN_STORAGE)

import tnbench as T  # noqa: E402

# The default tokenizer and codec for the token-native table. o200k is the tokenizer
# GPT-4o-class models use, and +freq is the paper's recommended default: most of ANS's ratio
# at the fastest decode.
DEFAULT_TOKENIZER = "o200k"
DEFAULT_CODEC = "freq"

# A 1024-dim float32 embedding is 4096 bytes. Held constant across all three tables.
EMBEDDING_BYTES = 4096

# Coding table ids, matching what train_tables() writes.
FREQ_TABLE_ID = 10
ANS_TABLE_ID = 11


def pg_env() -> dict:
    """Environment for the local no-root PostgreSQL, from setup_pg.sh --env."""
    script = os.path.join(os.path.dirname(HERE), "setup_pg.sh")
    out = subprocess.run(
        ["bash", script, "--env"], capture_output=True, text=True, check=True
    ).stdout
    env = {}
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        key, _, val = line[len("export ") :].partition("=")
        env[key] = val.strip().strip("'\"")
    return env


def connect(**kwargs) -> psycopg.Connection:
    env = pg_env()
    return psycopg.connect(
        host=env["PGHOST"],
        port=int(env["PGPORT"]),
        user=env.get("PGUSER", "postgres"),
        dbname=os.environ.get("PGDATABASE", "postgres"),
        autocommit=True,
        **kwargs,
    )


def load_corpus(domain: str = "prose", n_docs: int | None = None) -> list[str]:
    """Load chunk texts, preferring the exported fixture and falling back to the corpus.

    Using the fixture keeps the Postgres numbers on exactly the documents the Rust
    cross-language tests validated against.
    """
    path = os.path.join(FIXTURE, "chunks", f"{domain}.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for line in f if line.strip()]
    else:
        r50k = tiktoken.get_encoding("r50k_base")
        rng = np.random.default_rng(9012)
        chunks = T.make_chunks(T.load_ids(f"{domain}_test"), 512, 40, rng)
        texts = [r50k.decode(c.tolist()) for c in chunks]

    if n_docs is None:
        return texts
    # Repeat the sampled chunks to reach the requested row count. Repetition inflates
    # LZ-family and dictionary methods, which is why every codec here is order-0 or
    # per-value: none of them can see across rows, so cycling the corpus does not advantage
    # the token-native side. It would matter for a zstd --train baseline.
    return [texts[i % len(texts)] for i in range(n_docs)]


def apply_schema(conn: psycopg.Connection) -> None:
    with open(os.path.join(HERE, "schema.sql"), encoding="utf-8") as f:
        conn.execute(f.read())


def train_tables(conn: psycopg.Connection, domain: str = "prose") -> None:
    """Train the +freq and +ANS coding tables from the domain's train split.

    Trained through the extension's own SQL entry point, on the train split only, so the
    tables never see the test chunks the benchmark measures.
    """
    # pgtoken.table_info raises if the table file is absent, which is the cheapest existence
    # check available; the tables are files on disk, not catalog rows.
    try:
        conn.execute("SELECT 1 FROM pgtoken.table_info(%s)", (FREQ_TABLE_ID,)).fetchone()
        conn.execute("SELECT 1 FROM pgtoken.table_info(%s)", (ANS_TABLE_ID,)).fetchone()
        return  # both already trained
    except psycopg.Error:
        pass

    r50k = tiktoken.get_encoding("r50k_base")
    train_ids = T.load_ids(f"{domain}_train")
    # A slice is enough to fit a stable unigram table and keeps setup to a few seconds.
    train_text = r50k.decode(train_ids[: 400 * 512].tolist())

    conn.execute("DROP TABLE IF EXISTS tnt_train_corpus")
    conn.execute("CREATE TABLE tnt_train_corpus(body text)")
    with conn.cursor() as cur:
        # Split into rows so the training query looks like a normal corpus scan.
        step = 4096
        rows = [(train_text[i : i + step],) for i in range(0, len(train_text), step)]
        cur.executemany("INSERT INTO tnt_train_corpus(body) VALUES (%s)", rows)

    for tid, fn in ((FREQ_TABLE_ID, "train_freq_table"), (ANS_TABLE_ID, "train_ans_table")):
        try:
            conn.execute(
                f"SELECT pgtoken.{fn}(%s, %s, 'SELECT body FROM tnt_train_corpus')",
                (tid, DEFAULT_TOKENIZER),
            )
        except psycopg.errors.RaiseException as e:
            if "already exists" not in str(e):
                raise


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
    }
