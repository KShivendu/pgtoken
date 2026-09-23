"""Shared plumbing for the Postgres harness: connection, corpus, tokenizer, vocabularies.

Kept separate so bench_readwrite.py and bench_storage_wal.py load an identical corpus into
identical tables and encode with the identical client codec. If the two benchmarks disagreed
about the data or the vocabulary, neither number would mean anything.

The extension has no tokenizer: a real deployment tokenizes client-side, so the benchmark does
too. The default is HuggingFace `tokenizers` v1, the fast splitter released in 2026. That
choice is deliberate: the read path's cost is "a text column re-tokenizes on every read", and a
modern tokenizer does that in microseconds, not the hundreds of microseconds a cold tiktoken
rank table shows. Benchmarking against the fast tokenizer is the honest test. Override with
PGTOKEN_TOKENIZER=tiktoken-r50k or tiktoken-o200k to compare.
"""

from __future__ import annotations

import functools
import glob
import json
import os
import subprocess
import sys

import numpy as np
import psycopg

import pgtoken_client as K

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
        "  TOKEN_STORAGE_REPO=/path/to/token-storage uv run python benchmarks/bench_readwrite.py"
    )

sys.path.insert(0, TOKEN_STORAGE)

import tnbench as T  # noqa: E402

# Which tokenizer produces the IDs. One tokenizer drives everything: the stored IDs, the write
# path's detokenize, and the read path's re-tokenize, so the text-vs-token comparison is like
# for like.
TOKENIZER = os.environ.get("PGTOKEN_TOKENIZER", "hf-gpt2")
DEFAULT_CODEC = "freq"

# A 1024-dim float32 embedding is 4096 bytes. Held constant across all three tables.
EMBEDDING_BYTES = 4096


# ── Tokenizer ─────────────────────────────────────────────────────────────────────────────


class Tokenizer:
    """Uniform wrapper so the benchmark never branches on which library it got.

    `encode(text) -> list[int]` and `decode(ids) -> text`. `n_vocab` is what a pgtoken
    vocabulary is sized to, which is what picks the storage width.
    """

    def __init__(self, name: str, encode_fn, decode_fn, n_vocab: int):
        self.name = name
        self._encode = encode_fn
        self._decode = decode_fn
        self.n_vocab = n_vocab

    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def decode(self, ids) -> str:
        return self._decode(list(ids))


def _hf(model: str, repo: str) -> Tokenizer:
    from tokenizers import Tokenizer as HF

    hits = glob.glob(
        os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo}/snapshots/*/tokenizer.json")
    )
    if hits:
        path = hits[0]
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "tokenizer.json")

    tok = HF.from_file(path)
    # tokenizers v1 dropped get_vocab_size from the Tokenizer object, so read the size straight
    # from the file: the largest id in the vocab and added-tokens, plus one. For gpt2 this is
    # 50257, which the extension turns into a raw16 width.
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    ids = list(spec.get("model", {}).get("vocab", {}).values())
    ids += [t["id"] for t in spec.get("added_tokens", [])]
    n_vocab = max(ids) + 1

    return Tokenizer(
        f"hf-{model}",
        lambda t: tok.encode(t, add_special_tokens=False).ids,
        lambda ids: tok.decode(ids, skip_special_tokens=False),
        n_vocab,
    )


def _tiktoken(enc_name: str, label: str) -> Tokenizer:
    import tiktoken

    enc = tiktoken.get_encoding(enc_name)
    return Tokenizer(
        label,
        lambda t: enc.encode(t, disallowed_special=()),
        lambda ids: enc.decode(ids),
        enc.n_vocab,
    )


@functools.lru_cache(maxsize=1)
def encoder() -> Tokenizer:
    """The client-side tokenizer. Nothing about it is known to the database."""
    if TOKENIZER == "hf-gpt2":
        return _hf("gpt2", "openai-community--gpt2")
    if TOKENIZER == "tiktoken-r50k":
        return _tiktoken("r50k_base", "tiktoken-r50k")
    if TOKENIZER == "tiktoken-o200k":
        return _tiktoken("o200k_base", "tiktoken-o200k")
    raise SystemExit(
        f"unknown PGTOKEN_TOKENIZER={TOKENIZER!r}; use hf-gpt2, tiktoken-r50k or tiktoken-o200k"
    )


def tokenize_all(texts: list[str]) -> list[list[int]]:
    enc = encoder()
    return [enc.encode(t) for t in texts]


def raw_codec_for(vocab_size: int) -> str:
    """The raw width the extension picks for a vocabulary of this size, mirrored client-side."""
    if vocab_size <= 256:
        return "raw8"
    if vocab_size <= 65536:
        return "raw16"
    return "raw24"


# ── Connection and corpus ─────────────────────────────────────────────────────────────────


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
    cross-language tests validated against. The text is tokenizer-independent either way.
    """
    path = os.path.join(FIXTURE, "chunks", f"{domain}.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for line in f if line.strip()]
    else:
        r50k = _tiktoken("r50k_base", "r50k")
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


# ── Vocabularies (named; matches the current extension API) ─────────────────────────────────


def _vocab_name(kind: str) -> str:
    """A per-tokenizer name so switching tokenizers does not collide with a reserved size.

    A vocabulary's name and size are reserved forever, so `bench_hfgpt2_freq` and
    `bench_tiktokeno200k_freq` are distinct and each keeps its own trained ranking.
    """
    slug = encoder().name.replace("-", "")
    return f"bench_{slug}_{kind}"


def table_dir(conn: psycopg.Connection) -> str:
    return conn.execute("SELECT current_setting('pgtoken.table_dir')").fetchone()[0]


def ensure_vocabulary(
    conn: psycopg.Connection, name: str, vocab_size: int, compression: str = "raw"
) -> int:
    """Create `name` if absent, else return the reserved id (create would raise on a rerun)."""
    row = conn.execute("SELECT id FROM pgtoken.vocabulary WHERE name = %s", (name,)).fetchone()
    if row is not None:
        return row[0]
    return conn.execute(
        "SELECT pgtoken.create_vocabulary(%s, %s, compression => %s)",
        (name, vocab_size, compression),
    ).fetchone()[0]


def ensure_raw_vocab(conn: psycopg.Connection) -> int:
    """A raw vocabulary sized to the tokenizer; the width follows from n_vocab."""
    return ensure_vocabulary(conn, _vocab_name("raw"), encoder().n_vocab, "raw")


def ensure_freq_vocab(
    conn: psycopg.Connection, domain: str = "prose"
) -> tuple[int, K.RankTable]:
    """Create and train the freq vocabulary from the domain's train split.

    Trained on the train split only, so the ranking never sees the test chunks the benchmark
    measures. `pgtoken.train` is write-once, so a rerun retrains nothing. Returns the
    vocabulary id and the loaded ranking (the client codec needs both).
    """
    name = _vocab_name("freq")
    vocab_id = ensure_vocabulary(conn, name, encoder().n_vocab, "freq")

    ranked = conn.execute(
        "SELECT ranked FROM pgtoken.vocabulary_info(%s)", (name,)
    ).fetchone()[0]
    if ranked is None:
        # Recover text from the stored r50k ids, then re-tokenize with the target tokenizer so
        # the ranking is over the ids the benchmark actually stores. A slice fits a stable
        # frequency table and keeps setup to a few seconds.
        r50k = _tiktoken("r50k_base", "r50k")
        train_text = r50k.decode(T.load_ids(f"{domain}_train")[: 400 * 512].tolist())
        enc = encoder()
        step = 4096
        rows = [
            (enc.encode(train_text[i : i + step]),)
            for i in range(0, len(train_text), step)
        ]
        conn.execute("DROP TABLE IF EXISTS pgtoken_train_corpus")
        conn.execute("CREATE TABLE pgtoken_train_corpus(ids int[])")
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO pgtoken_train_corpus(ids) VALUES (%s)", rows)
        conn.execute(
            "SELECT pgtoken.train(%s, 'SELECT ids FROM pgtoken_train_corpus')", (name,)
        )

    path = os.path.join(table_dir(conn), f"{vocab_id}.tntt")
    return vocab_id, K.RankTable.load(path)


# ── Stats ───────────────────────────────────────────────────────────────────────────────────


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
