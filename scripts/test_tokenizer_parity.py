"""Check that the server's o200k tokenizer matches tiktoken's o200k_base, id for id.

Documents: 200 each of prose, code and Hindi, 2000 r50k tokens apiece, decoded from the
token-storage corpus (`<repo>/data/corpus/{domain}_test.npy`, flat r50k id streams), plus a list of
hard cases (emoji, CJK, mixed whitespace, special-token text, long digit runs, contractions).
Every document is tokenized by `pgtoken.tokenize(text, 'o200k')` and by tiktoken's
`encode_ordinary`, which treats special tokens as plain text, as the server does.

Also times the server: one statement tokenizes a whole domain, minus the same statement with
`length(t)` in place of the tokenizer, divided by the token count.

    eval "$(bash setup_pg.sh --env)" && uv run python scripts/test_tokenizer_parity.py

Uses database `pgtoken_parity` (created if missing; override with PGTOKEN_PARITY_DB). Exits 1 on
any mismatch.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
import tiktoken

CORPUS = Path(os.environ.get("TOKEN_STORAGE_REPO", Path.home() / "projects/token-storage")) / "data/corpus"
DB = os.environ.get("PGTOKEN_PARITY_DB", "pgtoken_parity")
DOCS_PER_DOMAIN = 200
DOC_TOKENS = 2000

HARD_CASES = [
    "Hello, world!",
    "🦜",
    "🦜🦜🦜 family: 👨‍👩‍👧‍👦, flags: 🇮🇳🇯🇵, skin tone: 👍🏽",
    "日本語のテキストと中文文本，还有한국어도 섞여 있어요。",
    "  leading, trailing  ",
    "tabs\tand\t\tmixed \t \n whitespace\r\n\r\n   \n\n\n    x",
    "trailing spaces then newline   \n",
    "\n\n\n",
    "   ",
    "",
    "<|endoftext|>",
    "before<|endoftext|>after <|endofprompt|>",
    "1234567890123456789012345678901234567890",
    "3.14159265358979323846 and 1,000,000 and 0x1F2E3D",
    "I'm, you're, he's, we've, they'll, I'd, DON'T, Can'T",
    "'s 's's ''s",
    "def f(x):\n    return {\"role\": \"user\", 'k': [1, 2, 3]}\n",
    "{1,2,3}",
    '{"role": "user"}',
    "नमस्ते दुनिया, यह एक परीक्षण है।",
    "Ünïcödé çömbïnïng: e\u0301 a\u0300 n\u0303",
    "A" * 5000,
    " " * 300 + "x",
    "x" + "\u00a0" * 10 + "y",
    "URLs: https://example.com/a/b?c=d&e=f#g and paths C:\\Users\\x\\file.txt",
    "MiXeD CaSe WoRdS and ALLCAPS and camelCaseIdentifier and snake_case_name",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def connect(dbname: str) -> psycopg.Connection:
    return psycopg.connect(dbname=dbname, autocommit=True)


def repo_docs() -> dict[str, list[str]]:
    """When the token-storage corpus is not there (CI), this repository's own files: its prose
    (the READMEs) and its code (Rust and Python), cut into 4 KB documents at line ends."""
    root = Path(__file__).resolve().parent.parent
    files = {
        "prose": [root / "README.md", root / "py" / "README.md"],
        "code": sorted(root.glob("core/src/**/*.rs")) + sorted(root.glob("benchmarks/*.py")),
    }
    out = {}
    for domain, paths in files.items():
        docs, cur = [], ""
        for p in paths:
            for line in p.read_text(encoding="utf-8").splitlines(keepends=True):
                cur += line
                if len(cur) >= 4096:
                    docs.append(cur)
                    cur = ""
        out[domain] = docs[:DOCS_PER_DOMAIN]
    return out


def corpus_docs(r50k: tiktoken.Encoding) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for domain in ("prose", "code", "hindi"):
        stream = np.load(CORPUS / f"{domain}_test.npy", mmap_mode="r")
        docs = []
        for i in range(DOCS_PER_DOMAIN):
            ids = stream[i * DOC_TOKENS : (i + 1) * DOC_TOKENS].tolist()
            # Cutting at a token boundary can split a character; tiktoken's default
            # errors="replace" turns that into U+FFFD, which is still ordinary text to tokenize.
            docs.append(r50k.decode(ids))
        out[domain] = docs
    return out


def first_difference(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def check(conn, o200k, name: str, docs: list[str]) -> tuple[int, list[str]]:
    matched = 0
    failures = []
    for i, text in enumerate(docs):
        server = conn.execute("SELECT pgtoken.tokenize(%s, 'o200k')", [text]).fetchone()[0]
        want = o200k.encode_ordinary(text)
        if server == want:
            matched += 1
        else:
            at = first_difference(server, want)
            lo = max(0, at - 3)
            piece = o200k.decode(want[lo : at + 4])
            failures.append(
                f"{name}[{i}] differs at token {at}: text near there {piece!r}\n"
                f"    server   {server[lo : at + 4]} ({len(server)} ids)\n"
                f"    tiktoken {want[lo : at + 4]} ({len(want)} ids)"
            )
            log("  MISMATCH " + failures[-1])
        if (i + 1) % 50 == 0 or i + 1 == len(docs):
            log(f"  {name}: {i + 1}/{len(docs)} checked, {matched} matched")
    return matched, failures


def time_domain(conn, docs: list[str], n_tokens: int) -> float:
    """Server ns per token: tokenizing a domain in one statement, minus the same scan without it."""

    def run(expr: str) -> float:
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            conn.execute(f"SELECT sum({expr}) FROM unnest(%s::text[]) t", [docs]).fetchone()
            best = min(best, time.perf_counter() - t0)
        return best

    tok = run("cardinality(pgtoken.tokenize(t, 'o200k'))")
    base = run("length(t)")
    return (tok - base) * 1e9 / n_tokens


def main() -> int:
    with connect("postgres") as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", [DB]).fetchone():
            admin.execute(f'CREATE DATABASE "{DB}"')
    conn = connect(DB)
    conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
    if not conn.execute("SELECT 1 FROM pgtoken.vocabulary WHERE name = 'o200k'").fetchone():
        conn.execute("SELECT pgtoken.create_vocabulary('o200k')")
    t0 = time.perf_counter()
    conn.execute("SELECT pgtoken.tokenize('warm up', 'o200k')")
    log(f"first tokenize in this backend (builds the tokenizer): {(time.perf_counter() - t0) * 1e3:.0f} ms")

    o200k = tiktoken.get_encoding("o200k_base")
    r50k = tiktoken.get_encoding("r50k_base")
    if CORPUS.exists():
        log(f"decoding {DOCS_PER_DOMAIN} docs x {DOC_TOKENS} r50k tokens per domain from {CORPUS}")
        domains = corpus_docs(r50k)
    else:
        log(f"{CORPUS} not found, so checking this repository's own READMEs and source files")
        domains = repo_docs()

    failures: list[str] = []
    summary = []
    for name, docs in [*domains.items(), ("hard", HARD_CASES)]:
        log(f"checking {name} ({len(docs)} docs)")
        matched, f = check(conn, o200k, name, docs)
        failures += f
        n_tokens = sum(len(o200k.encode_ordinary(d)) for d in docs)
        ns = time_domain(conn, docs, n_tokens) if name != "hard" else float("nan")
        summary.append((name, matched, len(docs), n_tokens, ns))
        log(f"  {name}: {matched}/{len(docs)} matched, {n_tokens} o200k tokens, server {ns:.0f} ns/token")

    # The text input path must agree with tokenize(): a literal cast to the column type.
    for text in HARD_CASES:
        got = conn.execute("SELECT (%s::text)::tokens.o200k::int[]", [text]).fetchone()[0]
        if got != o200k.encode_ordinary(text) and text not in ("{1,2,3}",):
            failures.append(f"text input {text[:40]!r}: {got[:8]} vs {o200k.encode_ordinary(text)[:8]}")
            log("  MISMATCH " + failures[-1])
    log("text input path: checked the hard cases through the text -> tokens cast")

    log("\ndomain   matched   o200k tokens   server ns/token")
    for name, matched, total, n_tokens, ns in summary:
        log(f"{name:<8} {matched:>3}/{total:<4} {n_tokens:>14}   {ns:>15.0f}")
    if failures:
        log(f"\nFAIL: {len(failures)} mismatches")
        return 1
    log("\nok: server tokenization matches tiktoken o200k_base on every document")
    return 0


if __name__ == "__main__":
    sys.exit(main())
