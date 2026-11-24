#!/usr/bin/env python3
"""Compression ratio vs value length, per dataset and method, every value compressed on its own.

The paper's claim: agents and chat apps write lots of SHORT text that a model reads again. Stored
as the model's own token IDs and compressed per value with pgtoken's `freq` codec, it is about 2x
smaller, where general compressors and PostgreSQL compress nothing. A database compresses each
value alone, so every method here sees one value at a time. Shared side tables (zstd
dictionaries, the FSST symbol table, pgtoken's rank table) are trained once per dataset on a
held-out train split and counted once per column (amortised), not per value.

Datasets (train and test are disjoint; nothing is evaluated on its own train split):
  wildchat  allenai/WildChat-1M (ODC-BY, not gated): every message is one value (user and
            assistant turns separately), plus every whole conversation as one value. Split by
            conversation hash.
  dbpedia   KShivendu/dbpedia-entities-openai-1M, title + "\\n" + text (embedding column not read).
  c4        allenai/c4 en validation shard 0, one document per value. Split by URL hash.
  code      token-storage code_train.npy / code_test.npy (codeparrot Python, r50k ids) decoded to
            text. No file boundaries survive in those arrays, so code has cut buckets only.
Lengths: (a) natural values bucketed by o200k token count, (b) cuts of exactly 16 ... 2048 o200k
tokens, cut on token boundaries at value starts and decoded back to text.

Methods (bytes include every per-value header and frame header):
  utf8            the text, no compression
  pg text         what a PostgreSQL text column stores: utf8 under ~2 kB (no compression is
                  tried below TOAST_TUPLE_THRESHOLD, 2032 B); above it lz4 when smaller
  lz4             LZ4_compress_default block + 4-byte raw size (PostgreSQL keeps the raw size in
                  the compressed varlena header, so this matches what it would store)
  zstd-3          zstd level 3 frame, content size on, no checksum, no dict id, and the 4-byte
                  magic number not counted (= the magicless frame). The standard frame adds 4 B
  zstd-3 + dict   same, with a 112 KB dictionary trained on the train split's text
  fsst            FSST (github.com/cwida/fsst), symbol table trained on the train split
  schmidt 16-bit  Schmidt et al. (PVLDB 19(11) 2026, github.com/umbra-db/token-vldb2026): their
                  cl100k vocabulary patched to 65,536 tokens of 1..7 bytes, greedy longest match,
                  2 bytes per token. Our re-implementation of their TokenizerCompressor
  raw24 <tok>     pgtoken raw24 value: 5-6 B header + 3 B per token
  zstd-3 + dict on ids <tok>  zstd-3 with a 112 KB dictionary over the raw24 token-ID bytes
  freq <tok>      pgtoken freq without lz4: header + one LEB128 rank per token
  freq+lz4 <tok>  what pgtoken writes when asked for freq: freq_lz4 when lz4 makes it smaller
  freq C4 table <tok>  pgtoken's choice with the rank table trained on C4 (the "default table")
Tokenizers: o200k (tiktoken o200k_base IDs, HF fast tokenizer for the timed re-tokenize) and Qwen3
(Qwen/Qwen3-Embedding-0.6B tokenizer.json).

Decode-to-IDs time, single thread, ns per token: text methods = decompress + tokenize with the HF
fast tokenizer; ID methods = decode to a numpy uint32 array. Best of 3, on a random share of the
values (--time-frac). Timings are secondary to sizes on a shared box.

Usage:
    uv run --with pyarrow --with lz4 python paper/compress_by_length.py --smoke
    uv run --with pyarrow --with lz4 python paper/compress_by_length.py --datasets wildchat
    uv run --with pyarrow --with lz4 python paper/compress_by_length.py --resume <run dir> \\
        --datasets dbpedia,c4,code --tokenizers o200k,qwen3
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperlib as P  # noqa: E402

C = P.C
import lz4.block  # noqa: E402
import pgtoken_rs as R  # noqa: E402
import tiktoken  # noqa: E402
import zstandard as zstd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser(os.environ.get("PGTOKEN_CL_CACHE", "~/.cache/pgtoken/compress_length"))
FSST_SRC = os.path.expanduser("~/.cache/pgtoken/fsst")
FSST_COMMIT = "e638d4cf8c26129d73c242a4127b42b975de5b63"
PG_TOAST_THRESHOLD = 2032
CUT_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
NAT_EDGES = [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 1 << 62]
NAT_LABELS = ["1-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1023", "1024-2047",
              "2048+"]
QWEN3_REPO = "Qwen/Qwen3-Embedding-0.6B"
TEXT_METHODS = ["utf8", "pg_text", "lz4", "zstd3", "zstd3_dict", "fsst", "schmidt16"]
ID_METHODS = ["raw24", "zstd3_dict_ids", "freq", "freq_lz4", "freq_c4"]
LABEL = {"utf8": "utf8 (no compression)", "pg_text": "PostgreSQL text", "lz4": "lz4",
         "zstd3": "zstd-3", "zstd3_dict": "zstd-3 + dict (text)", "fsst": "FSST",
         "schmidt16": "Schmidt 16-bit (cl100k patched)", "raw24": "raw24", "zstd3_dict_ids":
         "zstd-3 + dict (ID bytes)", "freq": "freq", "freq_lz4": "freq+lz4 (pgtoken)",
         "freq_c4": "freq+lz4, C4 table"}


def log(*a):
    print(*a, flush=True)


def crc(s: str) -> int:
    return zlib.crc32(s.encode("utf-8", "surrogatepass"))


# ── data ─────────────────────────────────────────────────────────────────────────────────────


def _cached(name: str, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".json.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    d = build()
    with gzip.open(path + ".part", "wt", encoding="utf-8") as f:
        json.dump(d, f)
    os.replace(path + ".part", path)
    return d


def load_wildchat(n_convs: int) -> dict:
    """Messages and whole conversations of the first n_convs conversations of shard 0, 1, ..."""

    def build():
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = sorted(fs.glob("datasets/allenai/WildChat-1M/data/train-*.parquet"))
        train, test, convs = [], [], []
        seen = 0
        for fp in files:
            pf = pq.ParquetFile(fs.open(fp))
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg, columns=["conversation_hash", "conversation", "language"])
                for row in t.to_pylist():
                    msgs = [(m["role"], m["content"]) for m in row["conversation"] if m["content"]]
                    if not msgs:
                        continue
                    split = "train" if crc(row["conversation_hash"]) % 2 == 0 else "test"
                    if split == "train":
                        train.extend(c for _, c in msgs)
                    else:
                        cid = row["conversation_hash"]
                        for i, (role, c) in enumerate(msgs):
                            test.append({"text": c, "role": role, "lang": row["language"],
                                         "src": f"{cid}#{i}"})
                        convs.append({"text": "\n\n".join(f"{r}: {c}" for r, c in msgs),
                                      "role": "conversation", "lang": row["language"],
                                      "src": cid})
                    seen += 1
                    if seen >= n_convs:
                        return {"train": train, "test": test, "test_convs": convs}
                if rg % 10 == 0:
                    log(f"  wildchat: {seen} conversations read")
        return {"train": train, "test": test, "test_convs": convs}

    return _cached(f"wildchat_{n_convs}", build)


def load_dbpedia(n_rows: int) -> dict:
    def build():
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = sorted(fs.glob("datasets/KShivendu/dbpedia-entities-openai-1M/data/train-*.parquet"))
        train, test = [], []
        for fp in files:
            pf = pq.ParquetFile(fs.open(fp))
            for b in pf.iter_batches(batch_size=2000, columns=["_id", "title", "text"]):
                for r in b.to_pylist():
                    v = (r["title"] or "") + "\n" + (r["text"] or "")
                    if crc(r["_id"]) % 2 == 0:
                        train.append(v)
                    else:
                        test.append({"text": v, "role": "abstract", "src": r["_id"]})
                if len(train) + len(test) >= n_rows:
                    return {"train": train, "test": test}
        return {"train": train, "test": test}

    return _cached(f"dbpedia_{n_rows}", build)


def load_c4(n_docs: int) -> dict:
    def build():
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        train, test = [], []
        with fs.open("datasets/allenai/c4/en/c4-validation.00000-of-00008.json.gz", "rb") as raw:
            with gzip.open(raw, "rt", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if crc(d["url"]) % 2 == 0:
                        train.append(d["text"])
                    else:
                        test.append({"text": d["text"], "role": "doc", "src": d["url"]})
                    if len(train) + len(test) >= n_docs:
                        break
        return {"train": train, "test": test}

    return _cached(f"c4_{n_docs}", build)


def load_code(n_train_tokens: int, n_test_tokens: int) -> dict:
    """codeparrot Python from token-storage, r50k ids decoded to text. One long test text: there
    are no file boundaries, so code is evaluated on cuts only. Train values are 512-token windows
    (as in bench_zstd_dict), cut from the train text."""
    import tnbench as T

    r50k = tiktoken.get_encoding("r50k_base")
    tr = T.load_ids("code_train")[:n_train_tokens].tolist()
    te = T.load_ids("code_test")[:n_test_tokens].tolist()
    train_text = r50k.decode(tr)
    test_text = r50k.decode(te)
    o200k = tiktoken.get_encoding("o200k_base")
    ids = o200k.encode_ordinary(train_text)
    train = []
    for s in range(0, len(ids) - 512, 512):
        t = _valid_text(o200k, ids[s:s + 512])
        if t:
            train.append(t)
    return {"train": train, "test": [], "stream": test_text}


def _valid_text(enc, ids) -> str | None:
    """Decode a token window, dropping up to 3 tokens at either end until it is valid UTF-8."""
    for head in range(4):
        for tail in range(4):
            part = ids[head:len(ids) - tail]
            if not part:
                return None
            try:
                return enc.decode_bytes(part).decode("utf-8")
            except UnicodeDecodeError:
                continue
    return None


def make_cuts(texts: list[str], n_per: int, rng, boundary: bool, lengths=CUT_LENGTHS) -> list[dict]:
    """Windows of exactly L o200k tokens. With boundary=True each window starts at a value start
    and runs on across the next values ("\\n\\n" between them); windows never overlap. Then up to
    n_per windows per L are picked at random."""
    enc = tiktoken.get_encoding("o200k_base")
    sep = enc.encode_ordinary("\n\n")
    stream, starts = [], []
    for ids in enc.encode_ordinary_batch(texts, num_threads=1):
        starts.append(len(stream))
        stream.extend(ids)
        stream.extend(sep)
    out = []
    for L in lengths:
        cands = []
        if boundary:
            pos = 0
            for s in starts:
                if s < pos:
                    continue
                if s + L > len(stream):
                    break
                cands.append(s)
                pos = s + L
        else:
            cands = list(range(0, len(stream) - L, L))
        pick = sorted(rng.choice(len(cands), size=min(n_per, len(cands)), replace=False))
        for j in pick:
            s = cands[j]
            t = _valid_text(enc, stream[s:s + L])
            if t:
                out.append({"text": t, "role": f"cut{L}", "cut": L, "src": f"pos{s}"})
    return out


# ── tokenizers ───────────────────────────────────────────────────────────────────────────────


class Tok:
    """IDs from the reference tokenizer, and a HF fast tokenizer for the timed re-tokenize."""

    def __init__(self, name: str):
        from tokenizers import Tokenizer as HF

        self.name = name
        if name == "o200k":
            self.tt = tiktoken.get_encoding("o200k_base")
            self.hf = HF.from_file(C.hf_tokenizer_json())
            self.vocab = 200019
        elif name == "qwen3":
            from huggingface_hub import hf_hub_download

            self.tt = None
            self.hf = HF.from_file(hf_hub_download(QWEN3_REPO, "tokenizer.json"))
            with open(hf_hub_download(QWEN3_REPO, "tokenizer.json"), encoding="utf-8") as f:
                spec = json.load(f)
            ids = list(spec["model"]["vocab"].values()) + [t["id"] for t in spec["added_tokens"]]
            self.vocab = max(ids) + 1
        else:
            raise SystemExit(f"unknown tokenizer {name}")

    def ids_batch(self, texts: list[str]) -> list[np.ndarray]:
        if self.tt is not None:
            # encode_ordinary: special-token text such as '<|endoftext|>' stays ordinary text,
            # as pgcommon does with disallowed_special=().
            return [np.asarray(x, dtype=np.uint32)
                    for x in self.tt.encode_ordinary_batch(texts, num_threads=1)]
        out = []
        for i in range(0, len(texts), 256):
            for e in self.hf.encode_batch(texts[i:i + 256], add_special_tokens=False):
                out.append(np.asarray(e.ids, dtype=np.uint32))
        return out

    def hf_encode(self, text: str):
        return self.hf.encode(text, add_special_tokens=False).ids


# ── side tables ──────────────────────────────────────────────────────────────────────────────


def rank_table(ids_list: list[np.ndarray], vocab: int):
    allids = np.concatenate([a for a in ids_list if len(a)]).astype(np.int64)
    counts = np.bincount(allids, minlength=vocab)[:vocab]
    seen = np.nonzero(counts)[0]
    order = seen[np.argsort(-counts[seen], kind="stable")].astype(np.uint32)
    blob = b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", len(order)) + order.astype("<u4").tobytes()
    rank_of = np.arange(vocab, dtype=np.int64) + len(order)
    rank_of[order] = np.arange(len(order))
    return R.Table.from_bytes(blob), rank_of, len(blob)


def leb_len(x: np.ndarray) -> np.ndarray:
    return 1 + (x >= 1 << 7) + (x >= 1 << 14) + (x >= 1 << 21) + (x >= 1 << 28)


def header_len(n: int) -> int:
    return 4 + int(leb_len(np.array([n]))[0])


def pack24(ids: np.ndarray) -> bytes:
    return ids.astype("<u4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def unpack24(b: bytes) -> np.ndarray:
    a = np.frombuffer(b, np.uint8).reshape(-1, 3).astype(np.uint32)
    return a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)


def zstd_pair(level: int, dict_data=None):
    """Standard frames, content size on, no checksum, no dict id. Sizes are reported as
    len(frame) - 4: the 4-byte magic number is not counted (a column knows its codec), which is
    exactly the magicless frame. Level-based compressor, so zstd adapts its parameters to each
    small input (a fixed-parameter magicless compressor came out 1 B larger per value)."""
    zc = zstd.ZstdCompressor(level=level, dict_data=dict_data, write_content_size=True,
                             write_checksum=False, write_dict_id=False)
    zd = zstd.ZstdDecompressor(dict_data=dict_data)
    return zc, zd


ZSTD_MAGIC = 4


def train_zstd_dict(samples: list[bytes], size: int, cap_bytes: int):
    """Train on samples in a fixed shuffled order until cap_bytes, then the dictionary."""
    rng = np.random.default_rng(11)
    order = rng.permutation(len(samples))
    picked, total = [], 0
    for i in order:
        if total >= cap_bytes:
            break
        picked.append(samples[i])
        total += len(samples[i])
    t0 = time.perf_counter()
    d = zstd.train_dictionary(size, picked)
    return d, time.perf_counter() - t0, total


# ── FSST / Schmidt helper binary ─────────────────────────────────────────────────────────────


def fsst_cli() -> str:
    out_dir = os.path.join(FSST_SRC, "build-cli")
    exe = os.path.join(out_dir, "fsst_cli")
    src = os.path.join(HERE, "fsst_cli.cpp")
    if os.path.exists(exe) and os.path.getmtime(exe) > os.path.getmtime(src):
        return exe
    if not os.path.isdir(FSST_SRC):
        subprocess.run(["git", "clone", "-q", "https://github.com/cwida/fsst.git", FSST_SRC], check=True)
    subprocess.run(["git", "-C", FSST_SRC, "checkout", "-q", FSST_COMMIT], check=True)
    os.makedirs(out_dir, exist_ok=True)
    cc = ["g++", "-std=c++17", "-I", FSST_SRC]
    subprocess.run(cc + ["-O3", "-DNDEBUG", "-c", f"{FSST_SRC}/libfsst.cpp", "-o", f"{out_dir}/libfsst.o"], check=True)
    subprocess.run(cc + ["-O1", "-c", f"{FSST_SRC}/fsst_avx512.cpp", "-o", f"{out_dir}/fsst_avx512.o"], check=True)
    subprocess.run(cc + ["-O3", "-DNDEBUG", src, f"{out_dir}/libfsst.o", f"{out_dir}/fsst_avx512.o",
                         "-o", exe, "-lpthread"], check=True)
    return exe


def write_values(path: str, texts) -> None:
    with open(path, "wb") as f:
        for t in texts:
            b = t.encode("utf-8", "surrogatepass")
            f.write(struct.pack("<I", len(b)))
            f.write(b)


def read_pairs(path: str, skip: int = 0) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(skip)
        return np.frombuffer(f.read(), dtype="<u4").reshape(-1, 2)


def schmidt_vocab() -> tuple[str, dict]:
    """Their extract_tokens.py --patch on cl100k, ported (same steps, same random seed)."""
    import random
    import string

    path = os.path.join(CACHE, "schmidt_cl100k_patched.bin")
    meta_p = path + ".json"
    if os.path.exists(path) and os.path.exists(meta_p):
        with open(meta_p) as f:
            return path, json.load(f)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = sorted(enc._mergeable_ranks.items(), key=lambda kv: kv[1])
    unique = {}
    for tok, rank in tokens:
        unique.setdefault(tok, rank)
    for c in range(256):
        unique.setdefault(bytes([c]), -1)
    for tok, _ in tokens:
        if b"\x00" in tok and len(tok) > 1:
            unique.pop(tok, None)
        if len(tok) > 4 and tok.startswith(b"<|") and tok.endswith(b"|>"):
            unique.pop(tok, None)
    sizes_old = defaultdict(list)
    for tok, rank in unique.items():
        sizes_old[len(tok)].append((tok, rank))
    sizes, count = {}, 0
    for size in range(1, 5):
        if count + len(sizes_old[size]) > 2 ** 16:
            break
        sizes[size] = list(sizes_old[size])
        count += len(sizes_old[size])
    token_set = {t for s in sizes.values() for t, _ in s}

    def add(tok, rank=-1):
        nonlocal count
        if tok not in token_set and count < 2 ** 16:
            sizes.setdefault(len(tok), []).append((tok, rank))
            token_set.add(tok)
            count += 1

    up, lo, dg = string.ascii_uppercase, string.ascii_lowercase, string.digits
    for a in up:
        for b in up:
            add(bytes([ord(a), ord(b)]))
    for a in lo:
        for b in lo:
            add(bytes([ord(a), ord(b)]))
    for a in dg:
        for b in dg:
            add(bytes([ord(a), ord(b)]))
    for a in ",.":
        for b in dg:
            add(bytes([ord(a), ord(b)]))
    b64 = up + lo + dg + "+-/ _\n"
    for a in b64:
        for b in b64:
            add(bytes([ord(a), ord(b)]))
    for a in dg:
        for b in dg:
            for c in dg:
                add(bytes([ord(a), ord(b), ord(c)]))
    for size in range(5, 16):
        if count + len(sizes_old[size]) > 2 ** 16:
            break
        sizes[size] = list(sizes_old[size])
        count += len(sizes_old[size])
        for t, _ in sizes[size]:
            token_set.add(t)
        if size == 6:
            if count + 52 <= 2 ** 16:
                for c in up + lo:
                    add(bytes([ord(c)]) * 6)
                break
    for s in range(max(sizes.keys()) + 1, 17):
        lst = list(sizes_old.get(s, []))
        random.seed(0)
        random.shuffle(lst)
        for t, r in lst:
            if all(chr(b) in string.ascii_letters for b in t):
                add(t, r)
        for t, r in lst:
            if all(chr(b) in string.ascii_letters + string.digits + " " for b in t):
                add(t, r)
        for t, r in lst:
            add(t, r)
    vocab = [t for s in sorted(sizes) for t, _ in sorted(sizes[s])]
    meta = {"tokens": len(vocab), "max_len": max(map(len, vocab)),
            "by_len": {str(k): len(v) for k, v in sorted(sizes.items())}}
    assert meta["tokens"] <= 65536 and meta["max_len"] <= 7, meta
    with open(path, "wb") as f:
        for t in vocab:
            f.write(bytes([len(t)]) + t)
    with open(meta_p, "w") as f:
        json.dump(meta, f)
    return path, meta


# ── one pass: one dataset, one tokenizer ─────────────────────────────────────────────────────


def best_ns(fn, reps=3) -> int:
    best = 1 << 62
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        best = min(best, time.perf_counter_ns() - t0)
    return best


def bucket_of(row) -> str:
    if row.get("cut"):
        return str(row["cut"])
    n = row["tok_o200k"]
    for lo, hi, lab in zip(NAT_EDGES, NAT_EDGES[1:], NAT_LABELS):
        if lo <= n < hi:
            return lab
    return NAT_LABELS[-1]


def run_pass(ds: str, data: dict, values: list[dict], tok: Tok, c4_table, args, out_dir: str) -> dict:
    """Train every side table on data['train'], then size and time every test value."""
    rng = np.random.default_rng(args.seed)
    meta = {"dataset": ds, "tokenizer": tok.name, "n_train_values": len(data["train"]),
            "n_test_values": len(values)}
    tmp = os.path.join(CACHE, f"tmp-{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    text_pass = tok.name == args.text_pass_tokenizer

    # train side tables
    t0 = time.perf_counter()
    train_ids = tok.ids_batch(data["train"])
    meta["train_tokens"] = int(sum(len(a) for a in train_ids))
    meta["tokenize_train_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    table, rank_of, table_bytes = rank_table(train_ids, tok.vocab)
    meta["freq_table"] = {"train_s": time.perf_counter() - t0, "bytes": table_bytes, "k": table.k}
    zdict_ids, s, nbytes = train_zstd_dict([pack24(a) for a in train_ids if len(a)],
                                           args.dict_kb * 1024, args.dict_sample_mb << 20)
    meta["zstd_dict_ids"] = {"train_s": s, "bytes": len(zdict_ids.as_bytes()), "sample_bytes": nbytes}
    del train_ids
    zc_ids, zd_ids = zstd_pair(3, zdict_ids)
    log(f"[{ds}/{tok.name}] trained: freq table k={table.k} ({table_bytes / 1024:.0f} KB, "
        f"{meta['freq_table']['train_s']:.2f} s), id dict {meta['zstd_dict_ids']['bytes'] // 1024} KB "
        f"({s:.1f} s on {nbytes / 1e6:.1f} MB)")

    if text_pass:
        zdict_text, s, nbytes = train_zstd_dict([t.encode("utf-8", "surrogatepass") for t in data["train"]],
                                                args.dict_kb * 1024, args.dict_sample_mb << 20)
        meta["zstd_dict_text"] = {"train_s": s, "bytes": len(zdict_text.as_bytes()), "sample_bytes": nbytes}
        zc3, zd3 = zstd_pair(3)
        zc3d, zd3d = zstd_pair(3, zdict_text)
        exe = fsst_cli()
        write_values(f"{tmp}/train.bin", data["train"])
        write_values(f"{tmp}/test.bin", (v["text"] for v in values))
        subprocess.run([exe, "fsst", f"{tmp}/train.bin", f"{tmp}/test.bin", f"{tmp}/fsst.out"], check=True)
        with open(f"{tmp}/fsst.out", "rb") as f:
            fsst_table, fsst_train_ns = struct.unpack("<IQ", f.read(12))
        fsst_rows = read_pairs(f"{tmp}/fsst.out", 12)
        meta["fsst"] = {"train_s": fsst_train_ns / 1e9, "table_bytes": fsst_table,
                        "table_counted": "once per column (amortised), not per value"}
        vpath, vmeta = schmidt_vocab()
        subprocess.run([exe, "schmidt", vpath, f"{tmp}/test.bin", f"{tmp}/schmidt.out"], check=True)
        schmidt_rows = read_pairs(f"{tmp}/schmidt.out")
        meta["schmidt"] = vmeta
        log(f"[{ds}] text tables: zstd dict {meta['zstd_dict_text']['bytes'] // 1024} KB "
            f"({meta['zstd_dict_text']['train_s']:.1f} s), FSST table {fsst_table} B "
            f"({fsst_train_ns / 1e6:.1f} ms), Schmidt vocab {vmeta['tokens']} tokens")
    shutil.rmtree(tmp, ignore_errors=True)

    # evaluate
    ids_all = tok.ids_batch([v["text"] for v in values])
    path = os.path.join(out_dir, f"values_{ds}_{tok.name}.jsonl")
    fout = open(path, "w", encoding="utf-8")
    running = defaultdict(lambda: defaultdict(list))
    t_start = time.perf_counter()
    for i, (v, ids) in enumerate(zip(values, ids_all)):
        b = v["text"].encode("utf-8", "surrogatepass")
        n = len(ids)
        row = {"dataset": ds, "kind": "cut" if v.get("cut") else "natural", "role": v["role"],
               "i": i, "src": v.get("src"), "lang": v.get("lang"), "utf8": len(b),
               f"tok_{tok.name}": n}
        if v.get("cut"):
            row["cut"] = v["cut"]
        if "tok_o200k" in v:
            row["tok_o200k"] = v["tok_o200k"]
        sizes, times = {}, {}
        timed = rng.random() < args.time_frac
        # ID methods
        raw = R.encode(ids, "raw24", 1)
        sizes["raw24"] = len(raw)
        ranks = rank_of[ids]
        sizes["freq"] = header_len(n) + int(leb_len(ranks).sum())
        fq = R.encode(ids, "freq", 1, table)
        sizes["freq_lz4"] = len(fq)
        if fq[0] & 0x0F == 2:
            assert len(fq) == sizes["freq"], (len(fq), sizes["freq"])
        assert np.array_equal(np.frombuffer(R.decode(fq, table), "<u4"), ids)
        if c4_table is not None:
            sizes["freq_c4"] = len(R.encode(ids, "freq", 1, c4_table))
        p24 = pack24(ids)
        zi = zc_ids.compress(p24)
        sizes["zstd3_dict_ids"] = len(zi) - ZSTD_MAGIC
        if timed and n:
            times["raw24"] = best_ns(lambda: np.frombuffer(R.decode(raw), "<u4"))
            times["freq_lz4"] = best_ns(lambda: np.frombuffer(R.decode(fq, table), "<u4"))
            times["zstd3_dict_ids"] = best_ns(lambda: unpack24(zd_ids.decompress(zi)))
            tok_ns = best_ns(lambda: tok.hf_encode(v["text"]))
            times["tokenize"] = tok_ns
        if text_pass:
            sizes["utf8"] = len(b)
            l4 = lz4.block.compress(b, mode="default", store_size=True)
            sizes["lz4"] = len(l4)
            pg_lz4 = len(b) > PG_TOAST_THRESHOLD and len(l4) < len(b)
            sizes["pg_text"] = len(l4) if pg_lz4 else len(b)
            z3 = zc3.compress(b)
            sizes["zstd3"] = len(z3) - ZSTD_MAGIC
            z3d = zc3d.compress(b)
            sizes["zstd3_dict"] = len(z3d) - ZSTD_MAGIC
            assert zd3d.decompress(z3d) == b
            sizes["fsst"] = int(fsst_rows[i, 0])
            sizes["schmidt16"] = 2 * int(schmidt_rows[i, 0])
            if timed and n:
                times["utf8"] = best_ns(lambda: b.decode("utf-8", "surrogatepass")) + tok_ns
                times["lz4"] = best_ns(lambda: lz4.block.decompress(l4).decode("utf-8", "surrogatepass")) + tok_ns
                times["pg_text"] = times["lz4"] if pg_lz4 else times["utf8"]
                times["zstd3"] = best_ns(lambda: zd3.decompress(z3).decode("utf-8", "surrogatepass")) + tok_ns
                times["zstd3_dict"] = best_ns(lambda: zd3d.decompress(z3d).decode("utf-8", "surrogatepass")) + tok_ns
                times["fsst"] = int(fsst_rows[i, 1]) + tok_ns
        elif timed and n:
            # text sizes do not depend on the tokenizer; only the re-tokenize time does
            times["tokenize"] = tok_ns
        row["bytes"] = sizes
        if times:
            row["ns"] = times
        if v.get("preview") is None:
            row["preview"] = v["text"][:200]
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        key = row["role"] if row["kind"] == "natural" else "cut"
        for m in ("freq_lz4", "zstd3_dict", "fsst", "zstd3_dict_ids"):
            if m in sizes and sizes[m]:
                running[key][m].append(len(b) / sizes[m])
        if (i + 1) % args.progress_every == 0 or i + 1 == len(values):
            el = time.perf_counter() - t_start
            parts = []
            for k, d in running.items():
                parts.append(k + " " + " ".join(f"{m}={np.median(x):.2f}" for m, x in d.items()))
            log(f"[{ds}/{tok.name}] {i + 1}/{len(values)} values, {el:.0f} s | median ratio: "
                + " | ".join(parts))
    fout.close()
    meta["eval_s"] = time.perf_counter() - t_start
    return meta


# ── summary ──────────────────────────────────────────────────────────────────────────────────


def load_rows(out_dir: str) -> list[dict]:
    """Join the per-tokenizer rows of each value into one row with all methods."""
    by_key = {}
    for fn in sorted(os.listdir(out_dir)):
        if not (fn.startswith("values_") and fn.endswith(".jsonl")):
            continue
        tk = fn[:-6].rsplit("_", 1)[1]
        with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                k = (r["dataset"], r["i"])
                cur = by_key.setdefault(k, {kk: vv for kk, vv in r.items() if kk not in ("bytes", "ns")}
                                        | {"bytes": {}, "ns": {}})
                cur[f"tok_{tk}"] = r[f"tok_{tk}"]
                for m, x in r["bytes"].items():
                    cur["bytes"][m if m in TEXT_METHODS else f"{m}@{tk}"] = x
                for m, x in r.get("ns", {}).items():
                    if m == "tokenize":
                        cur["ns"][f"tokenize@{tk}"] = x
                    else:
                        cur["ns"][m if m in TEXT_METHODS else f"{m}@{tk}"] = x
    rows = list(by_key.values())
    for r in rows:
        r["bucket"] = bucket_of(r)
    return rows


def method_order(rows) -> list[str]:
    seen = set()
    for r in rows:
        seen.update(r["bytes"])
    out = [m for m in TEXT_METHODS if m in seen]
    for tk in ("o200k", "qwen3"):
        out += [f"{m}@{tk}" for m in ID_METHODS if f"{m}@{tk}" in seen]
    return out


def label(m: str) -> str:
    base, _, tk = m.partition("@")
    return LABEL[base] + (f" [{tk}]" if tk else "")


def dist(x) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {}
    return {"n": int(len(x)), "p25": float(np.percentile(x, 25)), "p50": float(np.median(x)),
            "p75": float(np.percentile(x, 75)), "mean": float(x.mean())}


def groups(rows):
    """(dataset, group name) -> rows. Natural groups by role, plus all messages for WildChat."""
    g = defaultdict(list)
    for r in rows:
        ds = r["dataset"]
        if r["kind"] == "cut":
            g[(ds, "cut")].append(r)
        else:
            g[(ds, r["role"])].append(r)
            if ds == "wildchat" and r["role"] in ("user", "assistant"):
                g[(ds, "messages")].append(r)
    return g


def sign_test_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    z = (abs(w - l) - 1) / math.sqrt(n)
    return float(math.erfc(max(z, 0) / math.sqrt(2)))


def wlt(rows, a: str, b: str) -> dict:
    """a vs b on stored bytes per value: win = a smaller."""
    d = [(r["bytes"][b] - r["bytes"][a], r) for r in rows if a in r["bytes"] and b in r["bytes"]]
    wins = [x for x, _ in d if x > 0]
    losses = [x for x, _ in d if x < 0]
    n = len(d)
    return {"n": n, "wins": len(wins), "losses": len(losses), "ties": n - len(wins) - len(losses),
            "win_bytes_each": float(np.mean(wins)) if wins else 0.0,
            "loss_bytes_each": float(np.mean(losses)) if losses else 0.0,
            "net_bytes_per_value": float(np.mean([x for x, _ in d])) if d else 0.0,
            "sign_p": sign_test_p(len(wins), len(losses))}


def summarize(out_dir: str, config: dict) -> None:
    rows = load_rows(out_dir)
    if not rows:
        return
    methods = method_order(rows)
    res = {"config": config, "groups": {}, "gate": None}
    md = ["# Compression ratio vs value length", "",
          "ratio = UTF-8 bytes / stored bytes (per-value headers and frame headers included; "
          "dictionaries, symbol tables and rank tables counted once per column, not per value). "
          "Cells are the median ratio over the values in the bucket; n is the number of values.", ""]
    for (ds, grp), rs in sorted(groups(rows).items()):
        bks = [b for b in (NAT_LABELS if grp != "cut" else [str(x) for x in CUT_LENGTHS])
               if any(r["bucket"] == b for r in rs)]
        gres = {}
        for b in bks + ["all"]:
            sel = [r for r in rs if b == "all" or r["bucket"] == b]
            cell = {"n": len(sel), "tok_o200k_p50": float(np.median([r["tok_o200k"] for r in sel]))}
            for m in methods:
                v = [r for r in sel if m in r["bytes"] and r["bytes"][m]]
                if not v:
                    continue
                cell[m] = {"ratio": dist([r["utf8"] / r["bytes"][m] for r in v]),
                           "bytes_per_o200k_token": dist([r["bytes"][m] / max(r["tok_o200k"], 1) for r in v])}
                tm = [r["ns"][m] / max(r.get(f"tok_{m.partition('@')[2] or 'o200k'}", 1), 1)
                      for r in v if m in r.get("ns", {})]
                if m in TEXT_METHODS:
                    for tk in ("o200k", "qwen3"):
                        tt = [(r["ns"][m] - r["ns"].get("tokenize@o200k", 0) + r["ns"][f"tokenize@{tk}"])
                              / max(r.get(f"tok_{tk}", 1), 1)
                              for r in v if m in r.get("ns", {}) and f"tokenize@{tk}" in r["ns"]]
                        if tt:
                            cell[m][f"decode_ns_per_token@{tk}"] = dist(tt)
                elif tm:
                    cell[m]["decode_ns_per_token"] = dist(tm)
            gres[b] = cell
        res["groups"][f"{ds}/{grp}"] = gres
        md += [f"## {ds} / {grp}", "", "| method | " + " | ".join(bks) + " | all |",
               "|---|" + "---:|" * (len(bks) + 1),
               "| n values | " + " | ".join(str(gres[b]["n"]) for b in bks + ["all"]) + " |",
               "| median o200k tokens | " + " | ".join(f"{gres[b]['tok_o200k_p50']:.0f}" for b in bks + ["all"]) + " |"]
        for m in methods:
            if not any(m in gres[b] for b in bks):
                continue
            md.append(f"| {label(m)} | " + " | ".join(
                f"{gres[b][m]['ratio']['p50']:.2f}" if m in gres[b] else "" for b in bks + ["all"]) + " |")
        md.append("")
        # decode time table, o200k
        trow = []
        for m in methods:
            key = "decode_ns_per_token@o200k" if m in TEXT_METHODS else "decode_ns_per_token"
            if m.endswith("@qwen3") or m == "schmidt16":
                continue
            if any(key in gres[b].get(m, {}) for b in bks):
                trow.append(f"| {label(m)} | " + " | ".join(
                    f"{gres[b][m][key]['p50']:.0f}" if key in gres[b].get(m, {}) else "" for b in bks + ["all"]) + " |")
        if trow:
            md += [f"Decode to o200k IDs, median ns per token (single core; text = decompress + HF "
                   f"tokenize):", "", "| method | " + " | ".join(bks) + " | all |",
                   "|---|" + "---:|" * (len(bks) + 1)] + trow + [""]

    # gate: WildChat messages, freq vs zstd-3 + dict on text at the median message length
    msgs = groups(rows).get(("wildchat", "messages"), [])
    if msgs and "freq_lz4@o200k" in msgs[0]["bytes"]:
        med = float(np.median([r["tok_o200k"] for r in msgs]))
        mb = bucket_of({"tok_o200k": int(med)})
        band = [r for r in msgs if 0.75 * med <= r["tok_o200k"] <= 1.25 * med]
        inb = [r for r in msgs if r["bucket"] == mb]

        def med_ratio(rs, m):
            return float(np.median([r["utf8"] / r["bytes"][m] for r in rs]))

        gate = {"median_message_tokens": med, "median_bucket": mb,
                "band": "0.75x to 1.25x the median length", "n_band": len(band), "n_bucket": len(inb)}
        band_en = [r for r in band if r.get("lang") == "English"]
        gate["n_band_english"] = len(band_en)
        for nm, rs in (("bucket", inb), ("band", band), ("band_english", band_en), ("all_messages", msgs)):
            gate[nm] = {m: med_ratio(rs, m) for m in ("freq_lz4@o200k", "freq@o200k", "zstd3_dict", "fsst", "zstd3_dict_ids@o200k", "lz4", "zstd3")}
            gate[nm]["wlt_freq_vs_zstd3_dict"] = wlt(rs, "freq_lz4@o200k", "zstd3_dict")
        gate["pass"] = gate["band"]["freq_lz4@o200k"] >= gate["band"]["zstd3_dict"]
        res["gate"] = gate
        md += ["## Gate: freq vs zstd-3 + dict on WildChat messages", "",
               f"Median message: {med:.0f} o200k tokens (bucket {mb}). Band = messages of "
               f"{0.75 * med:.0f} to {1.25 * med:.0f} tokens (n={len(band)}).", "",
               "| set | freq+lz4 | freq | zstd-3 + dict | FSST | zstd-3 dict on IDs | W/L/T freq vs zstd-dict |",
               "|---|---:|---:|---:|---:|---:|---|"]
        for nm in ("band", "band_english", "bucket", "all_messages"):
            g = gate[nm]
            w = g["wlt_freq_vs_zstd3_dict"]
            md.append(f"| {nm} | {g['freq_lz4@o200k']:.2f} | {g['freq@o200k']:.2f} | {g['zstd3_dict']:.2f} | "
                      f"{g['fsst']:.2f} | {g['zstd3_dict_ids@o200k']:.2f} | {w['wins']}/{w['losses']}/{w['ties']} "
                      f"(win {w['win_bytes_each']:.1f} B each, loss {w['loss_bytes_each']:.1f} B each, p={w['sign_p']:.2g}) |")
        md += ["", f"**Gate {'PASSES' if gate['pass'] else 'FAILS'}**: at the median message length freq "
               f"stores {gate['band']['freq_lz4@o200k']:.2f}x vs zstd-3 + dict {gate['band']['zstd3_dict']:.2f}x.", ""]

    # where freq loses, by bucket
    md += ["## Where freq+lz4 (o200k) loses, by bucket", "",
           "Share of values where freq+lz4 stores more bytes than zstd-3 + dict (text) / than FSST.", ""]
    res["loss_by_bucket"] = {}
    for (ds, grp), rs in sorted(groups(rows).items()):
        if "freq_lz4@o200k" not in rs[0]["bytes"] or "fsst" not in rs[0]["bytes"]:
            continue
        bks = [b for b in (NAT_LABELS if grp != "cut" else [str(x) for x in CUT_LENGTHS])
               if any(r["bucket"] == b for r in rs)]
        cells = {}
        for b in bks + ["all"]:
            sel = [r for r in rs if b == "all" or r["bucket"] == b]
            cells[b] = {m: wlt(sel, "freq_lz4@o200k", m) for m in ("zstd3_dict", "fsst")}
        res["loss_by_bucket"][f"{ds}/{grp}"] = cells
        md += [f"{ds} / {grp}:", "", "| vs | " + " | ".join(bks) + " | all |", "|---|" + "---:|" * (len(bks) + 1)]
        for m in ("zstd3_dict", "fsst"):
            md.append(f"| {label(m)} | " + " | ".join(
                f"{100 * cells[b][m]['losses'] / max(cells[b][m]['n'], 1):.1f}%" for b in bks + ["all"]) + " |")
        md.append("")

    # examples and losses
    rng = np.random.default_rng(5)
    md += ["## Examples (3 real values per dataset: short, median, long)", ""]
    res["examples"], res["losses"] = {}, {}
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds and r["kind"] == "natural" and r["role"] != "conversation"] \
            or [r for r in rows if r["dataset"] == ds]
        rs = sorted(rs, key=lambda r: r["tok_o200k"])
        ex = [rs[int(q * (len(rs) - 1))] for q in (0.1, 0.5, 0.95)]
        res["examples"][ds] = [{"preview": r["preview"][:150], "tok_o200k": r["tok_o200k"], "utf8": r["utf8"],
                                "bytes": r["bytes"], "role": r["role"]} for r in ex]
        md.append(f"### {ds}")
        md.append("")
        md.append("| value | tokens | " + " | ".join(label(m) for m in methods) + " |")
        md.append("|---|---:|" + "---:|" * len(methods))
        for r in ex:
            t = r["preview"][:150].replace("\n", "\\n").replace("|", "\\|")
            md.append(f"| {r['role']}: `{t}` | {r['tok_o200k']} | " + " | ".join(
                str(r["bytes"].get(m, "")) for m in methods) + " |")
        md.append("")
        # freq losses vs the best text compressor that a DB could use per value
        best_text = [m for m in ("zstd3_dict", "fsst") if m in methods]
        if "freq_lz4@o200k" in methods and best_text:
            losers = [r for r in rs if r["bytes"]["freq_lz4@o200k"] > min(r["bytes"][m] for m in best_text)]
            w = {m: wlt(rs, "freq_lz4@o200k", m) for m in best_text}
            pick = [losers[j] for j in rng.choice(len(losers), size=min(3, len(losers)), replace=False)] if losers else []
            res["losses"][ds] = {"n_values": len(rs), "n_freq_loses_to_best_text": len(losers), "wlt": w,
                                 "examples": [{"preview": r["preview"][:200], "tok_o200k": r["tok_o200k"],
                                               "utf8": r["utf8"], "bytes": r["bytes"]} for r in pick]}
            md.append(f"freq+lz4 (o200k) loses to the better of zstd-3 + dict and FSST on {len(losers)} of "
                      f"{len(rs)} values. " + "; ".join(
                          f"vs {label(m)}: W/L/T {x['wins']}/{x['losses']}/{x['ties']}" for m, x in w.items()))
            md.append("")
            for r in pick:
                t = r["preview"][:150].replace("\n", "\\n")
                md.append(f"- loss: `{t}` ({r['tok_o200k']} tokens, {r['utf8']} B): freq+lz4 "
                          f"{r['bytes']['freq_lz4@o200k']} B, zstd-3 + dict {r['bytes'].get('zstd3_dict')} B, "
                          f"FSST {r['bytes'].get('fsst')} B")
            md.append("")
    # default table
    md += ["## Default table: freq with the C4-trained table vs the in-domain table", "",
           "| dataset / group | in-domain freq+lz4 | C4-table freq+lz4 | loss |", "|---|---:|---:|---:|"]
    res["default_table"] = {}
    for (ds, grp), rs in sorted(groups(rows).items()):
        for tk in ("o200k", "qwen3"):
            a, b = f"freq_lz4@{tk}", f"freq_c4@{tk}"
            v = [r for r in rs if a in r["bytes"] and b in r["bytes"]]
            if not v or ds == "c4":
                continue
            ra = float(np.median([r["utf8"] / r["bytes"][a] for r in v]))
            rb = float(np.median([r["utf8"] / r["bytes"][b] for r in v]))
            tot = sum(r["bytes"][b] for r in v) / sum(r["bytes"][a] for r in v) - 1
            res["default_table"][f"{ds}/{grp}@{tk}"] = {"in_domain": ra, "c4_table": rb, "total_bytes_increase": tot}
            md.append(f"| {ds} / {grp} [{tk}] | {ra:.2f} | {rb:.2f} | {100 * (1 - rb / ra):.1f}% lower ratio, "
                      f"{100 * tot:+.1f}% bytes |")
    md.append("")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))
    log(f"wrote {out_dir}/summary.md and results.json")


# ── main ─────────────────────────────────────────────────────────────────────────────────────


def dataset_values(ds: str, args, rng) -> tuple[dict, list[dict]]:
    if ds == "wildchat":
        d = load_wildchat(args.wildchat_convs)
        nat = d["test"] + d["test_convs"]
        cuts = make_cuts([c["text"] for c in d["test_convs"]], args.cut_n, rng, boundary=True)
    elif ds == "dbpedia":
        d = load_dbpedia(args.dbpedia_rows)
        nat = d["test"]
        cuts = make_cuts([v["text"] for v in nat], args.cut_n, rng, boundary=True)
    elif ds == "c4":
        d = load_c4(args.c4_docs)
        nat = d["test"]
        cuts = make_cuts([v["text"] for v in nat], args.cut_n, rng, boundary=True)
    elif ds == "code":
        d = load_code(args.code_train_tokens, args.code_test_tokens)
        nat = []
        cuts = make_cuts([d["stream"]], args.cut_n, rng, boundary=False)
    else:
        raise SystemExit(f"unknown dataset {ds}")
    if args.max_natural and len(nat) > args.max_natural:
        keep = sorted(rng.choice(len(nat), size=args.max_natural, replace=False))
        nat = [nat[i] for i in keep]
    vals = nat + cuts
    o200k = tiktoken.get_encoding("o200k_base")
    for v, ids in zip(vals, o200k.encode_ordinary_batch([v["text"] for v in vals], num_threads=1)):
        v["tok_o200k"] = len(ids)
    vals = [v for v in vals if v["tok_o200k"] > 0]
    return d, vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="wildchat,dbpedia,c4,code")
    ap.add_argument("--tokenizers", default="o200k")
    ap.add_argument("--smoke", action="store_true", help="~500 values per dataset, own run dir")
    ap.add_argument("--resume", help="existing run dir: add passes to it, skipping finished ones")
    ap.add_argument("--summarize", help="only rebuild summary.md and results.json of this run dir")
    ap.add_argument("--wildchat-convs", type=int, default=40000)
    ap.add_argument("--dbpedia-rows", type=int, default=50000)
    ap.add_argument("--c4-docs", type=int, default=40000)
    ap.add_argument("--code-train-tokens", type=int, default=4_000_000)
    ap.add_argument("--code-test-tokens", type=int, default=1_500_000)
    ap.add_argument("--max-natural", type=int, default=0, help="subsample natural test values (0 = all)")
    ap.add_argument("--cut-n", type=int, default=1000, help="windows per cut length")
    ap.add_argument("--dict-kb", type=int, default=112)
    ap.add_argument("--dict-sample-mb", type=int, default=16)
    ap.add_argument("--time-frac", type=float, default=0.2)
    ap.add_argument("--progress-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--text-pass-tokenizer", default="o200k",
                    help="tokenizer pass that also computes the text methods (sizes are tokenizer-free)")
    args = ap.parse_args()
    if args.smoke:
        args.wildchat_convs, args.dbpedia_rows, args.c4_docs = 400, 1500, 1200
        args.code_train_tokens, args.code_test_tokens = 400_000, 150_000
        args.max_natural, args.cut_n, args.progress_every = 450, 8, 200
        args.dict_sample_mb = 4

    if args.summarize:
        with open(os.path.join(args.summarize, "config.json")) as f:
            summarize(args.summarize, json.load(f))
        return 0
    out_dir = args.resume or P.run_dir("compress-length", args.smoke)
    cfg_path = os.path.join(out_dir, "config.json")
    passes_done = set()
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)
        passes_done = {tuple(p["pass"]) for p in config.get("passes", [])}
    else:
        config = {"args": vars(args), "machine": P.machine(), "cache": CACHE, "fsst_commit": FSST_COMMIT,
                  "qwen3_tokenizer": QWEN3_REPO, "o200k_hf": C.hf_tokenizer_json(),
                  "pg_toast_threshold": PG_TOAST_THRESHOLD, "passes": [],
                  "sources": {"wildchat": "allenai/WildChat-1M (ODC-BY, not gated)",
                              "dbpedia": "KShivendu/dbpedia-entities-openai-1M (MIT)",
                              "c4": "allenai/c4 en/c4-validation.00000-of-00008.json.gz",
                              "code": "token-storage data/corpus/code_{train,test}.npy (r50k)"}}
    log(f"run dir: {out_dir}")
    log(f"machine: {json.dumps(P.machine())}")

    toks = {name: Tok(name) for name in args.tokenizers.split(",")}
    # The default-table check needs C4's train split rank table for every tokenizer.
    c4_tables = {}
    datasets = args.datasets.split(",")
    for tname, tok in toks.items():
        for ds in datasets:
            if (ds, tname) in passes_done:
                log(f"skip {ds}/{tname}: already in {out_dir}")
                continue
            if tname not in c4_tables:
                c4 = load_c4(args.c4_docs)
                c4_tables[tname] = rank_table(tok.ids_batch(c4["train"]), tok.vocab)[0]
            rng = np.random.default_rng(args.seed)
            t0 = time.perf_counter()
            data, vals = dataset_values(ds, args, rng)
            log(f"[{ds}] {len(data['train'])} train values, {len(vals)} test values "
                f"({sum(1 for v in vals if not v.get('cut'))} natural), loaded in {time.perf_counter() - t0:.0f} s")
            if tname != args.text_pass_tokenizer and (ds, args.text_pass_tokenizer) not in passes_done \
                    and args.text_pass_tokenizer not in toks:
                log(f"note: text methods come from the {args.text_pass_tokenizer} pass, which has not run")
            meta = run_pass(ds, data, vals, tok, None if ds == "c4" else c4_tables[tname], args, out_dir)
            meta["pass"] = [ds, tname]
            meta["loadavg_end"] = os.getloadavg()
            config["passes"].append(meta)
            with open(cfg_path, "w") as f:
                json.dump(config, f, indent=1)
            summarize(out_dir, config)
            if ds == "wildchat" and tname == "o200k":
                with open(os.path.join(out_dir, "results.json")) as f:
                    gate = json.load(f).get("gate")
                if gate:
                    b = gate["band"]
                    log(f"GATE ({tname}): median message {gate['median_message_tokens']:.0f} tokens; "
                        f"freq+lz4 {b['freq_lz4@o200k']:.2f}x vs zstd-3 + dict {b['zstd3_dict']:.2f}x "
                        f"-> {'PASS' if gate['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
