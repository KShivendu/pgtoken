#!/usr/bin/env python3
"""Export a tiktoken encoding's token -> bytes table into a pgtoken vocabulary.

    uv run python scripts/load_mapping.py --encoding o200k_base --vocabulary o200k

This is the client half of `pgtoken.text()`. The extension has no tokenizer: it takes a
`token_id -> bytes` table and concatenates, so producing that table is your job. Swap tiktoken for
HuggingFace or SentencePiece by replacing `export_pairs` -- everything below it is generic.

Bytes, not text, on purpose. A single character routinely spans two tokens, so neither token is
valid UTF-8 alone; pgtoken concatenates the bytes and interprets UTF-8 once, at the end.
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
import tiktoken


def export_pairs(encoding: str) -> tuple[list[tuple[int, bytes]], int, list[int]]:
    """Return (pairs, n_vocab, unassigned_ids) for a tiktoken encoding.

    `decode_single_token_bytes` raises on ids the encoding never assigned -- o200k_base has 19 of
    them (199998 and 200000-200017), which is why the obvious one-line comprehension crashes. The
    two real special tokens are fine: they decode to their literal text, b'<|endoftext|>' and
    b'<|endofprompt|>'.
    """
    enc = tiktoken.get_encoding(encoding)
    pairs: list[tuple[int, bytes]] = []
    unassigned: list[int] = []
    for i in range(enc.n_vocab):
        try:
            pairs.append((i, enc.decode_single_token_bytes(i)))
        except KeyError:
            unassigned.append(i)
    return pairs, enc.n_vocab, unassigned


def check_vocab_size(cur: psycopg.Cursor, vocabulary: str, n_vocab: int) -> None:
    """Fail before writing anything if the declared ID space does not match the tokenizer's.

    A mismatch is worth catching here because the mapping is write-once: get it wrong and the
    vocabulary is sealed to a mapping that disagrees with its own declared size.
    """
    try:
        cur.execute(
            "SELECT vocab_size, mapped FROM pgtoken.vocabulary_info(%s)", (vocabulary,)
        )
        row = cur.fetchone()
    except psycopg.Error as e:
        # vocabulary_info raises for an unknown name, and would also raise if pgtoken.table_dir
        # were unset. Both are the operator's to fix, so report them rather than a traceback.
        sys.exit(str(e).strip())
    if row is None:
        sys.exit(f"vocabulary {vocabulary!r} does not exist")
    vocab_size, mapped = row
    if mapped is not None:
        sys.exit(
            f"vocabulary {vocabulary!r} already has a mapping ({mapped} ids). "
            "Mappings are write-once; create a new vocabulary to change one."
        )
    if vocab_size != n_vocab:
        sys.exit(
            f"vocabulary {vocabulary!r} declares vocab_size {vocab_size}, but the encoding has "
            f"{n_vocab} ids. Create it with vocab_size => {n_vocab}."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoding", default="o200k_base", help="tiktoken encoding name")
    ap.add_argument("--vocabulary", required=True, help="pgtoken vocabulary to load into")
    ap.add_argument(
        "--dsn",
        default=os.environ.get("PGTOKEN_DSN", ""),
        help="libpq connection string (or set PGTOKEN_DSN)",
    )
    args = ap.parse_args()

    pairs, n_vocab, unassigned = export_pairs(args.encoding)
    print(f"{args.encoding}: {len(pairs)} of {n_vocab} ids have bytes", end="")
    print(f", {len(unassigned)} unassigned" if unassigned else "")

    with psycopg.connect(args.dsn, autocommit=False) as conn, conn.cursor() as cur:
        check_vocab_size(cur, args.vocabulary, n_vocab)

        # A temp table keeps the staging rows out of the user's schema and drops them on commit.
        # Binary COPY because this is 200k rows of arbitrary bytes; the text protocol would spend
        # its time escaping them.
        cur.execute("CREATE TEMP TABLE mapping_staging (id int, bytes bytea) ON COMMIT DROP")
        with cur.copy(
            "COPY mapping_staging (id, bytes) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["int4", "bytea"])
            for pair in pairs:
                copy.write_row(pair)

        cur.execute(
            "SELECT pgtoken.load_mapping(%s, 'SELECT id, bytes FROM mapping_staging')",
            (args.vocabulary,),
        )
        print(cur.fetchone()[0])
        conn.commit()

    # Unassigned ids are left out rather than mapped to nothing: pgtoken treats a missing entry as
    # a fault, so if one ever turns up in stored data you get an error naming it instead of a
    # silent gap in the text.
    if unassigned:
        print(
            f"left out {len(unassigned)} unassigned ids "
            f"({unassigned[0]}..{unassigned[-1]}); pgtoken.text will error if one is ever stored"
        )


if __name__ == "__main__":
    main()
