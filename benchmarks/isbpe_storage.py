#!/usr/bin/env python3
"""How many bytes does pgtoken store per value when the ids come from ISBPE instead of BPE?

ISBPE (Implied-Space BPE, ~/projects/research/tokenizer) stores no space inside a token and
marks only where a space is missing, so it emits fewer tokens. This checks whether that
survives pgtoken's `freq` codec (frequency rank, LEB128, lz4 on top when smaller).

Tokenizers, all with 65,536-ish ids:
  o200k   OpenAI's, via tiktoken (200k ids)
  bpe     BPE trained on the same text with the same pretokenizer as ISBPE (exp138 "same")
  isbpe   ISBPE dual-flag, exp138 model
  glue    ISBPE dual-flag with 61,536 regular ids plus 4,000 glue ids (an id for a common pair
          of adjacent tokens, like `#.#` + `get`), 65,536 in all. Built like the paper's
          exp138 earned_glue: the exp138 61,536 model, glue counted on the same 20 MB it was
          trained on, rule "marks".

Each tokenizer gets its own `freq` ranking, trained on the same first TRAIN_BYTES of the
domain's training split. Test values are cut from the held-out split at fixed byte sizes, on
UTF-8 boundaries, so every tokenizer stores exactly the same text. Every value is decoded back
and checked against its ids before any size counts.

Usage:
    uv run --with ~/projects/research/tokenizer/isbpe-rs/target/wheels/isbpe_rs-0.2.0-cp39-abi3-manylinux_2_34_x86_64.whl \
        --with tokenizers==0.22.1 python benchmarks/isbpe_storage.py
"""

from __future__ import annotations

import os
import struct
import time
import zlib

import numpy as np
import tiktoken
from tokenizers import Tokenizer

import isbpe_rs
import pgtoken_rs._native as pg

RESEARCH = os.path.expanduser("~/projects/research/tokenizer")
DOMAINS = {"code": "code", "english": "c4"}
GLUE_CORPUS = {"code": "code_20mb.txt", "english": "c4_20mb.txt"}  # exp138 training text
TRAIN_BYTES = 8_000_000
SIZES = ((256, 400), (1024, 300), (4096, 200), (16384, 60))  # (value bytes, values)


def chunks(text: bytes, size: int, count: int) -> list[bytes]:
    out, i = [], 0
    while len(out) < count and i + size <= len(text):
        j = i + size
        while j < len(text) and (text[j] & 0xC0) == 0x80:  # back up to a UTF-8 char start
            j -= 1
        out.append(text[i:j])
        i = j
    return out


def utf8_prefix(text: bytes, n: int) -> bytes:
    while n < len(text) and (text[n] & 0xC0) == 0x80:
        n -= 1
    return text[:n]


def rank_table(ids: np.ndarray) -> pg.Table:
    """Same ranking as RankTable::train: count desc, then id asc."""
    vals, counts = np.unique(ids, return_counts=True)
    order = np.lexsort((vals, -counts))
    ranked = vals[order].astype("<u4")
    head = b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", len(ranked))
    return pg.Table.from_bytes(head + ranked.tobytes())


def tokenizers_for(dom: str):
    o200k = tiktoken.get_encoding("o200k_base")
    bpe = Tokenizer.from_file(f"{RESEARCH}/models/exp138/bpe_{dom}_same_65536.json")
    isb = isbpe_rs.ISBPE.load(f"{RESEARCH}/models/exp138/isbpe_{dom}_dual_65536.json")
    glue = isbpe_rs.ISBPE.load(f"{RESEARCH}/models/exp138/isbpe_{dom}_dual_61536.json")
    glue.add_glue(open(f"{RESEARCH}/data/{GLUE_CORPUS[dom]}", "rb").read(), 4000, "marks")
    assert len(glue) == 65536, len(glue)

    def dec(b):
        return b.decode("utf-8", errors="surrogatepass")

    return {
        "o200k": lambda b: o200k.encode(dec(b), disallowed_special=()),
        "bpe": lambda b: bpe.encode(dec(b), add_special_tokens=False).ids,
        "isbpe": lambda b: isb.encode(dec(b)),
        "glue": lambda b: glue.encode(dec(b)),
    }, {"isbpe": lambda ids: isb.decode(ids), "bpe": lambda ids: bpe.decode(ids)}


def encode_big(fn, text: bytes, step: int = 1 << 20) -> np.ndarray:
    parts, i = [], 0
    while i < len(text):
        piece = utf8_prefix(text[i:], step)
        parts.append(np.asarray(fn(piece), dtype=np.uint32))
        i += len(piece)
    return np.concatenate(parts)


def main() -> int:
    for dom, prefix in DOMAINS.items():
        train = utf8_prefix(open(f"{RESEARCH}/data/{prefix}_tr.txt", "rb").read(), TRAIN_BYTES)
        test = open(f"{RESEARCH}/data/{prefix}_te.txt", "rb").read()
        toks, _ = tokenizers_for(dom)
        tables = {}
        for name, fn in toks.items():
            t0 = time.time()
            ids = encode_big(fn, train)
            tables[name] = rank_table(ids)
            mb_s = len(train) / 1e6 / (time.time() - t0)
            print(f"[{dom}] {name:6s} ranking from {len(ids):,} train tokens, "
                  f"{tables[name].k:,} ranked ids, tokenize {mb_s:.1f} MB/s", flush=True)

        print(f"\n[{dom}] bytes per value (avg). stored = pgtoken freq, lz4 when smaller", flush=True)
        print(f"{'value':>7} {'n':>4} {'zlib text':>10} | " + " | ".join(
            f"{n:>5} tok  {n:>5} B  lz4" for n in toks), flush=True)
        for size, count in SIZES:
            vals = chunks(test, size, count)
            res = {n: [] for n in toks}  # (tokens, stored bytes, is_lz4)
            for v in vals:
                for name, fn in toks.items():
                    ids = np.asarray(fn(v), dtype=np.uint32)
                    blob = pg.encode(ids, "freq", 1, tables[name])
                    back = np.frombuffer(pg.decode(blob, tables[name]), dtype="<u4")
                    assert np.array_equal(back, ids), (dom, name, v[:60])
                    res[name].append((len(ids), len(blob), blob[0] == 0x14))
            ztext = np.mean([len(zlib.compress(v, 6)) for v in vals])
            cells = []
            for name in toks:
                a = np.array(res[name])
                cells.append(f"{a[:, 0].mean():9.1f}  {a[:, 1].mean():7.1f}  {int(a[:, 2].sum()):3d}")
            print(f"{len(vals[0]):>7} {len(vals):>4} {ztext:10.1f} | " + " | ".join(cells), flush=True)

            # Concrete values: where ISBPE beats BPE most and where it loses most, in stored bytes.
            diff = [(res["isbpe"][k][1] - res["bpe"][k][1], k) for k in range(len(vals))]
            diff.sort()
            for tag, (d, k) in (("isbpe best", diff[0]), ("isbpe worst", diff[-1])):
                snippet = vals[k][:90].decode("utf-8", errors="replace").replace("\n", "\\n")
                print(f"    {tag:11s} {d:+5d} B  bpe {res['bpe'][k][1]} B / isbpe "
                      f"{res['isbpe'][k][1]} B / o200k {res['o200k'][k][1]} B  "
                      f"text {len(vals[k])} B: {snippet!r}", flush=True)
            gd = sorted((res["glue"][k][1] - res["isbpe"][k][1], k) for k in range(len(vals)))
            for tag, (d, k) in (("glue best", gd[0]), ("glue worst", gd[-1])):
                snippet = vals[k][:90].decode("utf-8", errors="replace").replace("\n", "\\n")
                print(f"    {tag:11s} {d:+5d} B vs isbpe  isbpe {res['isbpe'][k][1]} B / glue "
                      f"{res['glue'][k][1]} B  text {len(vals[k])} B: {snippet!r}", flush=True)
            print(f"    glue vs isbpe: smaller on {sum(d < 0 for d, _ in gd)}, larger on "
                  f"{sum(d > 0 for d, _ in gd)}, of {len(vals)}", flush=True)
            wins = sum(1 for d, _ in diff if d < 0)
            losses = sum(1 for d, _ in diff if d > 0)
            print(f"    isbpe vs bpe: smaller on {wins}, larger on {losses}, of {len(vals)}",
                  flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
