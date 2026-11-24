#!/usr/bin/env python3
"""Wave 1: the end-to-end RAG fetch. ANN top-k by vector, fetch the chunk bodies, hand token IDs
to the model as numpy arrays. Which storage format gets the IDs into the client fastest?

Hypothesis (PVLDB_PLAN.md, "Wave 1"): fetching k=100 chunks of 512 o200k tokens and producing
their IDs is >=2x faster at p50 with pgtoken freq + client decode than with text lz4 + client
tokenize, because tokenizing dominates the text path. Decision rule: >=2x p50 vs text (lz4) at
1 client, and not slower than bytea raw24 at p50. Under 1.3x, the paper's headline becomes
storage and operators, not latency.

One table per arm, same rows, same ids, same embeddings:

    CREATE TABLE rag_<arm> (id int PRIMARY KEY, emb vector(384), body <type>);
    CREATE INDEX ON rag_<arm> USING hnsw (emb vector_cosine_ops);

    arm                  body type                       client turns the fetched value into ids by
    text (pglz)          text                            tokenizing (fastest exact o200k tokenizer)
    text (lz4)           text COMPRESSION lz4            tokenizing
    text (lz4), batch    (same table as text (lz4))      tokenizing all k at once, library threads
    int4[]               int4[]                          numpy view of the binary array
    bytea raw24          bytea, 3 bytes per id (LE)      numpy unpack
    pgtoken raw          pgtoken.paper_o200k_raw (raw24) pgtoken_rs.decode
    pgtoken freq         pgtoken.o200k (freq / freq_lz4) pgtoken_rs.decode with the preset table
    oracle: in memory    rag_oracle (id, emb), no body   dict lookup: the ids are already cached

freq_lz4 is not a separate arm: the extension (and pgtoken_rs) writes freq_lz4 for a freq value
whenever lz4 makes it smaller, so it is reported as the share of freq rows stored that way.

The query, one prepared statement per arm, results in binary:

    SELECT id, body FROM rag_<arm> ORDER BY emb <=> $1::vector LIMIT k

Stages per query (microseconds):
    fetch    execute + fetchall of the query above: ANN, heap fetch, detoast/decompress, wire,
             psycopg row building. Also "the floor" for any token-ID format: stored bytes in
             the client, no decode.
    decode   client turns the k values into k numpy uint32 arrays (tokenize / decode / unpack)
    e2e      fetch + decode
    ann      the same query selecting only `id` (separate round trip, 1 client only)

Protocol: arms run round-robin in blocks with a rotating start, every arm runs the same query
vectors, the CPU cache is swept (64 MB) before every timed query at 1 client (so the tokenizer
and rank tables start cold, as in serving; bench_readwrite does the same), and every fetched
row is checked ID for ID against the ids the chunk was loaded with. With --clients N > 1, N
processes (one connection each) run each arm's block together; throughput, and the Little's-law
ratio (sum of query latencies / (N x wall)), are printed per arm. Sweeps are off by default at
N > 1: other clients pollute the cache anyway, and a sweep would be uncounted busy time.

Usage (inside the client container, see paper/README.md):
    python paper/rag_e2e.py --rows 20000 --k 100 --clients 1 4 --queries 300 --smoke
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import sys
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperlib as P  # noqa: E402

C = P.C
import tnbench as T  # noqa: E402

FREQ_VOCAB = "o200k"  # the preset: freq, compiled-in table, server tokenizer
RAW_VOCAB = "paper_o200k_raw"  # the same preset with compression => 'raw' (raw24)


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    table: str  # table suffix: rag_<table>
    coltype: str  # "" for the oracle (no body column)
    client: str  # tokenize | tokenize_batch | int4 | raw24 | pgtoken | oracle


ARMS = [
    Arm("text_pglz", "text (pglz)", "text_pglz", "text", "tokenize"),
    Arm("text_lz4", "text (lz4)", "text_lz4", "text COMPRESSION lz4", "tokenize"),
    Arm("text_lz4_batch", "text (lz4), batch tokenize", "text_lz4", "text COMPRESSION lz4",
        "tokenize_batch"),
    # Forced: lz4 on every row over 128 B (toast_tuple_target = 128) and kept inline
    # (STORAGE MAIN), so short values that the default 2 KB threshold never compresses are.
    Arm("text_lz4_forced", "text (lz4, forced)", "text_lz4_forced",
        "text COMPRESSION lz4", "tokenize"),
    Arm("int4_array", "int4[]", "int4_array", "int4[]", "int4"),
    Arm("bytea_raw24", "bytea raw24", "bytea_raw24", "bytea", "raw24"),
    # The extension's codec without the extension: pgtoken_rs freq bytes in a plain bytea
    # column (works on RDS). Same bytes as pgtoken freq, minus the type and its checks.
    Arm("bytea_freq", "bytea freq (client codec, no ext.)", "bytea_freq", "bytea", "pgtoken"),
    Arm("pgtoken_raw", "pgtoken raw (raw24)", "pgtoken_raw", f"pgtoken.{RAW_VOCAB}", "pgtoken"),
    Arm("pgtoken_freq", "pgtoken freq", "pgtoken_freq", f"pgtoken.{FREQ_VOCAB}", "pgtoken"),
    Arm("oracle_mem", "oracle: IDs already in memory", "oracle", "", "oracle"),
]
BASELINE = "text_lz4"
THREAT = "bytea_raw24"


# ── corpus and vectors ──────────────────────────────────────────────────────────────────────


def unit(rows: np.ndarray) -> np.ndarray:
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def vec_literal(v) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class Data:
    """Rows, ids, texts and vectors, from a prep_data.py directory or the small built-in corpus.

    Directory mode memory-maps everything, so 16 client processes share one page-cache copy and
    nothing is cycled. Built-in mode (no --data) is the 2,795 C4 test chunks of the smoke run,
    cycled to --rows with a warning, with seeded random unit vectors of --dim dimensions."""

    def __init__(self, path: str | None, rows: int, dim: int, seed: int, quiet: bool = False):
        self.path, self.seed = path, seed
        if path:
            self.meta = json.load(open(os.path.join(path, "meta.json")))
            self.tbin = np.memmap(os.path.join(path, "text.bin"), dtype=np.uint8, mode="r")
            self.toff = np.load(os.path.join(path, "text_off.npy"))
            self.ibin = np.memmap(os.path.join(path, "ids.bin"), dtype="<u4", mode="r")
            self.ioff = np.load(os.path.join(path, "ids_off.npy"))
            n = len(self.ioff) - 1
            if rows > n:
                raise SystemExit(f"--rows {rows} > {n} rows in {path}; no cycling in data mode")
            ep = os.path.join(path, "emb.npy")
            self.embs = np.load(ep, mmap_mode="r") if os.path.exists(ep) else None
            qp = os.path.join(path, "queries.npy")
            self.qvecs = np.load(qp) if os.path.exists(qp) else None
            self.dim = self.embs.shape[1] if self.embs is not None else dim
            self.n_unique = rows
            self.source = f"{self.meta['source']} ({path})"
        else:
            texts, id_lists = C.o200k_chunks("prose")
            self.texts, self.id_lists = texts, id_lists
            self.tb = np.array([len(t.encode()) for t in texts])
            self.embs, self.qvecs, self.dim = None, None, dim
            self.n_unique = len(texts)
            self.source = "token-storage prose_test (C4 English), 512-token o200k chunks, cycled"
            if rows > self.n_unique and not quiet:
                msg = (f"--rows {rows} > {self.n_unique} unique chunks: rows cycle the chunks "
                       f"{rows / self.n_unique:.1f}x.")
                warnings.warn(msg)
                print(f"WARNING: {msg}", flush=True)
        self.rows = rows

    def ids(self, i: int) -> np.ndarray:
        if self.path:
            return np.asarray(self.ibin[self.ioff[i]:self.ioff[i + 1]])
        return self.id_lists[i % self.n_unique]

    def text(self, i: int) -> str:
        if self.path:
            return bytes(self.tbin[self.toff[i]:self.toff[i + 1]]).decode()
        return self.texts[i % self.n_unique]

    def text_nbytes(self, i: int) -> int:
        if self.path:
            return int(self.toff[i + 1] - self.toff[i])
        return int(self.tb[i % self.n_unique])

    def token_lengths(self) -> np.ndarray:
        if self.path:
            return np.diff(self.ioff[: self.rows + 1])
        return np.array([len(self.ids(i)) for i in range(min(self.rows, self.n_unique))])

    def emb(self, lo: int, hi: int) -> np.ndarray:
        """Embeddings of rows lo..hi-1. Random ones: block b of 4096 rows is drawn from
        default_rng([seed, b]), so any slice is reproducible without generating the rest."""
        if self.embs is not None:
            return np.asarray(self.embs[lo:hi], dtype=np.float32)
        out = []
        for b in range(lo // 4096, (hi - 1) // 4096 + 1):
            blk = unit(np.random.default_rng([self.seed, b]).standard_normal((4096, self.dim)))
            s0, e0 = max(lo, b * 4096) - b * 4096, min(hi, (b + 1) * 4096) - b * 4096
            out.append(blk[s0:e0])
        return np.concatenate(out)

    def queries(self, n: int, offset: int) -> list[str]:
        """Query vectors as pgvector literals, formatted once before any timing. Real held-out
        entity embeddings when the dataset has them, else seeded random unit vectors."""
        if self.qvecs is not None:
            if offset + n > len(self.qvecs):
                raise SystemExit(f"need {offset + n} query vectors, dataset has {len(self.qvecs)}")
            return [vec_literal(v) for v in self.qvecs[offset:offset + n]]
        q = unit(np.random.default_rng([self.seed, 7_777_777]).standard_normal((offset + n, self.dim)))
        return [vec_literal(v) for v in q[offset:]]


# ── binary COPY ─────────────────────────────────────────────────────────────────────────────

COPY_HEADER = b"PGCOPY\n\xff\r\n\x00" + struct.pack(">ii", 0, 0)
COPY_TRAILER = struct.pack(">h", -1)


def field(b: bytes) -> bytes:
    return struct.pack(">i", len(b)) + b


def vec_bin(v: np.ndarray) -> bytes:
    return struct.pack(">hh", len(v), 0) + v.astype(">f4").tobytes()


def int4_array_bin(ids: np.ndarray) -> bytes:
    n = len(ids)
    a = np.empty((n, 2), dtype=">i4")
    a[:, 0] = 4
    a[:, 1] = ids
    return struct.pack(">iiiii", 1, 0, 23, n, 1) + a.tobytes()


def raw24(ids: np.ndarray) -> bytes:
    return np.ascontiguousarray(
        ids.astype("<u4").view(np.uint8).reshape(-1, 4)[:, :3]).tobytes()


# ── client decoders ─────────────────────────────────────────────────────────────────────────


class Client:
    """Everything a client process needs to turn fetched values into ids."""

    def __init__(self, tokenizer: str, ranking: bytes | None, data: "Data"):
        import pgtoken_rs as R

        self.R = R
        self.table = R.Table.from_bytes(ranking) if ranking is not None else None
        self.data = data
        self.tokenizer = tokenizer
        if tokenizer == "hf-o200k":
            hf = C._hf_file(C.hf_tokenizer_json(), "hf-o200k", n_vocab=200019).hf
            self.tok_one = lambda s: np.asarray(hf.encode(s, add_special_tokens=False).ids,
                                                dtype=np.uint32)
            self.tok_batch = lambda xs: [np.asarray(e.ids, dtype=np.uint32) for e in
                                         hf.encode_batch(xs, add_special_tokens=False)]
        elif tokenizer == "tiktoken-o200k":
            tt = C._tiktoken("o200k_base", "tiktoken-o200k").tt
            self.tok_one = lambda s: tt.encode_to_numpy(s, disallowed_special=()).astype(
                np.uint32, copy=False)
            self.tok_batch = lambda xs: [np.asarray(x, dtype=np.uint32) for x in
                                         tt.encode_ordinary_batch(xs, num_threads=len(os.sched_getaffinity(0)))]
        else:
            raise SystemExit(f"unknown tokenizer {tokenizer}")

    def decode(self, arm: Arm, rows: list) -> list[np.ndarray]:
        c = arm.client
        if c == "tokenize":
            return [self.tok_one(v) for _, v in rows]
        if c == "tokenize_batch":
            return self.tok_batch([v for _, v in rows])
        if c == "pgtoken":
            t = self.table if arm.key in ("pgtoken_freq", "bytea_freq") else None
            return [self.R.decode(v, t) for _, v in rows]
        if c == "int4":
            # 20-byte array header, then (len=4, value) pairs, big-endian.
            return [np.frombuffer(v, dtype=">i4", offset=20)[1::2].astype(np.uint32) for _, v in rows]
        if c == "raw24":
            # One vectorised pass over all k values: join, widen 3 -> 4 bytes, split by length.
            vals = [v for _, v in rows]
            a = np.frombuffer(b"".join(vals), dtype=np.uint8).reshape(-1, 3)
            x = np.zeros((a.shape[0], 4), dtype=np.uint8)
            x[:, :3] = a
            ids = x.view("<u4").ravel()
            return np.split(ids, np.cumsum([len(v) // 3 for v in vals[:-1]]))
        if c == "oracle":
            # The ids are already in client memory (the dataset's memory map, page-cache
            # resident): copy them out, as a cache of decoded ids would.
            return [np.array(self.data.ids(r[0])) for r in rows]
        raise ValueError(c)


def wire_bytes(arm: Arm, rows: list, data: "Data") -> int:
    if arm.client == "oracle":
        return 0
    if arm.client.startswith("tokenize"):
        return int(sum(data.text_nbytes(r[0]) for r in rows))
    return int(sum(len(r[1]) for r in rows))


# ── setup: vocabularies, tables, storage ────────────────────────────────────────────────────


def ensure_vocabs(conn) -> dict:
    out = {}
    for name, extra in ((FREQ_VOCAB, ""), (RAW_VOCAB, ", preset => 'o200k', compression => 'raw'")):
        row = conn.execute("SELECT id FROM pgtoken.vocabulary WHERE name = %s", (name,)).fetchone()
        out[name] = row[0] if row else conn.execute(
            f"SELECT pgtoken.create_vocabulary(%s{extra})", (name,)).fetchone()[0]
    info = {n: conn.execute("SELECT * FROM pgtoken.vocabulary_info(%s)", (n,)).fetchone()
            for n in out}
    print(f"  vocabularies: {out}; info {info}", flush=True)
    ranking = bytes(conn.execute("SELECT ranking FROM pgtoken.vocabulary_codec(%s)",
                                 (FREQ_VOCAB,)).fetchone()[0])
    return {"ids": out, "ranking": ranking}


def body_bin(arm: Arm, text: str, ids: np.ndarray, vocab: dict, R, table) -> bytes:
    if arm.coltype.startswith("text"):
        return text.encode()
    if arm.coltype == "int4[]":
        return int4_array_bin(ids)
    if arm.key == "bytea_raw24":
        return raw24(ids)
    if arm.key == "bytea_freq":
        return R.encode(ids, "freq", vocab["ids"][FREQ_VOCAB], table)
    if arm.key == "pgtoken_raw":
        return R.encode(ids, "raw24", vocab["ids"][RAW_VOCAB])
    if arm.key == "pgtoken_freq":
        return R.encode(ids, "freq", vocab["ids"][FREQ_VOCAB], table)
    raise ValueError(arm)


def table_names(tag: str, layout: str) -> dict:
    """Arm table suffix -> SQL table name. In the split layout every arm with a body has its own
    (id, body) table and all arms share one (id, emb) table with the HNSW index."""
    names = {a.table: f"{tag}_{a.table}" for a in ARMS}
    if layout == "split":
        names["oracle"] = f"{tag}_vec"
    return names


def arm_sql(a: Arm, tag: str, layout: str, k: int) -> tuple[str, str]:
    """(e2e query, ANN-only query) for an arm."""
    names = table_names(tag, layout)
    if layout == "single":
        t = names[a.table]
        body = ", body" if a.coltype else ""
        return (f"SELECT id{body} FROM {t} ORDER BY emb <=> %s::vector LIMIT {k}",
                f"SELECT id FROM {t} ORDER BY emb <=> %s::vector LIMIT {k}")
    ann = f"SELECT id FROM {tag}_vec ORDER BY emb <=> %s::vector LIMIT {k}"
    if not a.coltype:
        return ann, ann
    return (f"SELECT b.id, b.body FROM ({ann}) v JOIN {names[a.table]} b ON b.id = v.id", ann)


def copy_rows(conn, name: str, cols: str, n: int, row_fn, step: int = 2048) -> None:
    with conn.cursor() as cur:
        with cur.copy(f"COPY {name} ({cols}) FROM STDIN (FORMAT BINARY)") as cp:
            cp.write(COPY_HEADER)
            for lo in range(0, n, step):
                hi = min(n, lo + step)
                cp.write(b"".join(row_fn(lo, hi)))
            cp.write(COPY_TRAILER)


def load_tables(conn, data: Data, vocab: dict, tag: str, layout: str, reuse: bool) -> dict:
    """Create and fill the tables. Returns load stats per table."""
    import pgtoken_rs as R

    table = R.Table.from_bytes(vocab["ranking"])
    rows = data.rows
    names = table_names(tag, layout)
    conn.execute("CREATE TABLE IF NOT EXISTS rag_meta (tbl text PRIMARY KEY, rows int, seed int, "
                 "source text)")
    stats = {}

    def fresh(name: str) -> bool:
        meta = conn.execute("SELECT rows, seed, source FROM rag_meta WHERE tbl = %s",
                            (name,)).fetchone()
        if reuse and meta == (rows, data.seed, data.source):
            print(f"  {name}: reusing ({rows} rows)", flush=True)
            stats[name] = {"reused": True}
            return False
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute("DELETE FROM rag_meta WHERE tbl = %s", (name,))
        return True

    def done(name: str, st: dict) -> None:
        conn.execute(f"VACUUM (ANALYZE) {name}")
        conn.execute("INSERT INTO rag_meta VALUES (%s, %s, %s, %s)",
                     (name, rows, data.seed, data.source))
        stats[name] = st

    def hnsw(name: str) -> float:
        t = time.perf_counter()
        print(f"  {name}: building HNSW (m 16, ef_construction 64, cosine) ...", flush=True)
        conn.execute(f"CREATE INDEX ON {name} USING hnsw (emb vector_cosine_ops) "
                     "WITH (m = 16, ef_construction = 64)")
        return time.perf_counter() - t

    def rows_fn(arm: Arm | None, with_emb: bool, counter: dict):
        def fn(lo, hi):
            embs = data.emb(lo, hi) if with_emb else None
            out = []
            for i in range(lo, hi):
                parts = [field(struct.pack(">i", i))]
                if with_emb:
                    parts.append(field(vec_bin(embs[i - lo])))
                if arm is not None and arm.coltype:
                    b = body_bin(arm, data.text(i), data.ids(i), vocab, R, table)
                    if arm.key in ("pgtoken_freq", "bytea_freq") and (b[0] & 0x0F) == 4:
                        counter["lz4"] += 1
                    parts.append(field(b))
                out.append(struct.pack(">h", len(parts)) + b"".join(parts))
            if lo // 2048 % 50 == 0:
                print(f"      {lo}/{rows} rows", flush=True)
            return out
        return fn

    if layout == "split":
        name = names["oracle"]
        if fresh(name):
            t0 = time.perf_counter()
            conn.execute(f"CREATE TABLE {name} (id int PRIMARY KEY, emb vector({data.dim}))")
            copy_rows(conn, name, "id, emb", rows, rows_fn(None, True, {}))
            t1 = time.perf_counter()
            hs = hnsw(name)
            done(name, {"copy_s": t1 - t0, "hnsw_build_s": hs})
            print(f"  {name}: {rows} vectors in {t1 - t0:.0f} s, HNSW in {hs:.0f} s", flush=True)
    seen = set()
    for arm in ARMS:
        if arm.table in seen or (layout == "split" and not arm.coltype):
            continue
        seen.add(arm.table)
        name = names[arm.table]
        if not fresh(name):
            continue
        cnt = {"lz4": 0}
        t0 = time.perf_counter()
        body = f", body {arm.coltype}" if arm.coltype else ""
        single = layout == "single"
        emb_col = f", emb vector({data.dim})" if single else ""
        conn.execute(f"CREATE TABLE {name} (id int PRIMARY KEY{emb_col}{body})")
        if arm.key == "text_lz4_forced":  # PG16 has no STORAGE clause in CREATE TABLE
            conn.execute(f"ALTER TABLE {name} ALTER COLUMN body SET STORAGE MAIN, "
                         "SET (toast_tuple_target = 128)")
        cols = "id" + (", emb" if single else "") + (", body" if arm.coltype else "")
        copy_rows(conn, name, cols, rows, rows_fn(arm, single, cnt))
        t1 = time.perf_counter()
        hs = hnsw(name) if single else 0.0
        st = {"copy_s": t1 - t0, "hnsw_build_s": hs,
              "freq_lz4_rows": cnt["lz4"] if arm.key in ("pgtoken_freq", "bytea_freq") else None}
        done(name, st)
        print(f"  {name}: loaded {rows} rows in {t1 - t0:.0f} s"
              + (f", HNSW in {hs:.0f} s" if single else "")
              + (f", {cnt['lz4']}/{rows} rows stored as freq_lz4"
                 if arm.key in ("pgtoken_freq", "bytea_freq") else ""),
              flush=True)
    return stats


def storage(conn, rows: int, tag: str, layout: str) -> dict:
    """Per table: heap, TOAST and index bytes, rows per page, average stored body and embedding
    bytes, and how many values live out of line in TOAST."""
    out = {}
    names = table_names(tag, layout)
    for t, name in sorted(names.items()):
        if conn.execute("SELECT to_regclass(%s)", (name,)).fetchone()[0] is None:
            continue
        cols = [r[0] for r in conn.execute(
            "SELECT attname FROM pg_attribute WHERE attrelid = %s::regclass AND attnum > 0 "
            "AND NOT attisdropped", (name,))]
        heap, toast, idx, toast_name, pages = conn.execute(
            "SELECT pg_relation_size(c.oid), coalesce(pg_relation_size(c.reltoastrelid), 0), "
            "pg_indexes_size(c.oid), c.reltoastrelid::regclass::text, c.relpages "
            "FROM pg_class c WHERE c.oid = %s::regclass", (name,)).fetchone()
        d = {"table": name, "heap_bytes": heap, "toast_bytes": toast, "index_bytes": idx,
             "rows_per_page": rows / max(1, pages)}
        for col in ("body", "emb"):
            if col in cols:
                d[f"{col}_bytes_per_row"] = float(conn.execute(
                    f"SELECT avg(pg_column_size({col})) FROM {name}").fetchone()[0])
        if toast_name and toast_name != "-":
            d["toasted_values"] = conn.execute(
                f"SELECT count(DISTINCT chunk_id) FROM {toast_name}").fetchone()[0]
        out[t] = d
    return out


def probe_single(conn, tag: str, n: int = 20000) -> dict:
    """Split layout only: how would one table holding (id, emb, body) store its values? Builds
    a throwaway table of the first n rows for two arms and reports where emb and body went."""
    out = {}
    for t in ("text_lz4", "pgtoken_freq"):
        name = f"{tag}_probe_{t}"
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute(f"CREATE TABLE {name} AS SELECT v.id, v.emb, b.body FROM {tag}_vec v "
                     f"JOIN {tag}_{t} b ON b.id = v.id WHERE v.id < {n}")
        # CTAS copies values as they are (lz4 bodies stay lz4) and re-runs TOAST on each row.
        out[t] = storage_one(conn, name, n)
        conn.execute(f"DROP TABLE {name}")
    return out


def storage_one(conn, name: str, rows: int) -> dict:
    heap, toast, toast_name, pages = conn.execute(
        "SELECT pg_relation_size(c.oid), coalesce(pg_relation_size(c.reltoastrelid), 0), "
        "c.reltoastrelid::regclass::text, c.relpages FROM pg_class c WHERE c.oid = %s::regclass",
        (name,)).fetchone()
    conn.execute(f"VACUUM (ANALYZE) {name}")
    pages = conn.execute("SELECT relpages FROM pg_class WHERE oid = %s::regclass", (name,)).fetchone()[0]
    d = {"heap_bytes": heap, "toast_bytes": toast, "rows_per_page": rows / max(1, pages)}
    for col in ("body", "emb"):
        d[f"{col}_bytes_per_row"] = float(conn.execute(
            f"SELECT avg(pg_column_size({col})) FROM {name}").fetchone()[0])
    if toast_name and toast_name != "-":
        d["toasted_values"] = conn.execute(f"SELECT count(DISTINCT chunk_id) FROM {toast_name}").fetchone()[0]
    return d


def examples(conn, data: Data, tag: str, layout: str) -> list[dict]:
    """Rows at the 25th, 50th and 90th percentile of token length: text, UTF-8 bytes and the
    stored bytes per arm table."""
    names = table_names(tag, layout)
    lens = data.token_lengths()
    out = []
    for p in (25, 50, 90):
        target = np.percentile(lens, p)
        i = int(np.argmin(np.abs(lens - target)))
        sizes = {}
        for a in ARMS:
            if a.coltype and a.table not in sizes:
                sizes[a.table] = conn.execute(
                    f"SELECT pg_column_size(body) FROM {names[a.table]} WHERE id = %s", (i,)).fetchone()[0]
        out.append({"pct": p, "row": i, "tokens": int(lens[i]), "utf8_bytes": data.text_nbytes(i),
                    "text": data.text(i)[:300], "stored_bytes": sizes})
    return out


def prewarm(conn, tag: str, layout: str) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
    for name in sorted(set(table_names(tag, layout).values())):
        rels = conn.execute(
            "SELECT c.oid::regclass::text, c.reltoastrelid::regclass::text FROM pg_class c "
            "WHERE c.oid = to_regclass(%s)", (name,)).fetchone()
        if rels is None:
            continue
        idx = [r[0] for r in conn.execute(
            "SELECT indexrelid::regclass::text FROM pg_index WHERE indrelid = %s::regclass",
            (name,))]
        for rel in [rels[0]] + ([rels[1]] if rels[1] != "-" else []) + idx:
            conn.execute("SELECT pg_prewarm(%s)", (rel,))


def drop_caches() -> None:
    """Drop the host's page cache (needs a privileged container or root)."""
    os.sync()
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3\n")


def disk_stats(dev: str) -> tuple[int, int]:
    """(reads completed, sectors read) for `dev` from /proc/diskstats (host-wide)."""
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if p[2] == dev:
                return int(p[3]), int(p[5])
    return 0, 0


# ── the timed loop (one process per client) ─────────────────────────────────────────────────


def worker(w: int, n_clients: int, cfg: dict, shared: dict, barrier, out_q) -> None:
    """Run every arm on this worker's share of the queries. Phases are synchronised across
    workers with `barrier` so all N clients run the same arm at the same time."""
    import psycopg
    from psycopg.adapt import Loader
    from psycopg.pq import Format

    class RawBinary(Loader):
        format = Format.BINARY

        def load(self, data):
            return bytes(data)

    data = Data(cfg["data"], cfg["rows"], cfg["dim"], cfg["seed"], quiet=True)
    if cfg["cold"] and data.path:
        # The oracle's ids must stay "in memory" when the page cache is dropped: copy them to
        # process memory (2 GB for 1M C4 chunks; keep cold C4 runs to few clients).
        data.ibin = np.array(data.ibin)
    client = Client(cfg["tokenizer"], shared["ranking"], data)
    conn = C.connect()
    conn.adapters.register_loader(1007, RawBinary)  # int4[] -> raw bytes, parsed with numpy
    k = cfg["k"]
    conn.execute(f"SET hnsw.ef_search = {max(cfg['ef_search'], k)}")
    sql = {a.key: arm_sql(a, cfg["tag"], cfg["layout"], k)[0] for a in ARMS}
    sql_ann = {a.key: arm_sql(a, cfg["tag"], cfg["layout"], k)[1] for a in ARMS}
    qv = shared["queries"]
    warm_qv = shared["warm_queries"]
    sweep = cfg["sweep"]
    cold = cfg["cold"]
    stage_ann = cfg["stage_ann"]
    arms = [a for a in ARMS if a.key in cfg["arms"]]
    recs, phases = [], []
    mism = {a.key: 0 for a in arms}
    cur = conn.cursor(binary=True)

    # Server CPU per query: the backend's on-CPU nanoseconds from /proc/<pid>/schedstat. Needs
    # the server and client in the host PID namespace (docker-compose.bench.yml: pid: host);
    # otherwise server CPU is reported as missing.
    bpid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
    sched = f"/proc/{bpid}/schedstat"
    if not os.path.exists(sched):
        sched = None

    def srv_ns():
        if sched is None:
            return 0
        with open(sched) as f:
            return int(f.read().split()[0])

    def one(a: Arm, q: str):
        if cold:
            # Cold regime = "index warm, bodies cold": run this query's ANN untimed first, so the
            # HNSW pages it needs sit in shared_buffers (identical for every arm), then drop the
            # OS page cache so the body rows must come from the device (shared_buffers is small).
            cur.execute(sql_ann[a.key], (q,), prepare=True)
            cur.fetchall()
        if cold and n_clients == 1:
            drop_caches()
        if sweep:
            T.cache_pollute()
        s0 = srv_ns()
        c0 = time.process_time_ns()  # this process, all threads (tokenizer threads included)
        t0 = time.perf_counter_ns()
        cur.execute(sql[a.key], (q,), prepare=True)
        rows = cur.fetchall()
        t1 = time.perf_counter_ns()
        ids = client.decode(a, rows)
        t2 = time.perf_counter_ns()
        c1 = time.process_time_ns()
        s1 = srv_ns()
        one.cpu = ((c1 - c0) / 1e3, (s1 - s0) / 1e3 if sched else None)
        return rows, ids, (t1 - t0) / 1e3, (t2 - t1) / 1e3

    # warm-up: prepared statements, tokenizer init, buffers; not recorded
    for q in warm_qv[w::n_clients]:
        for a in arms:
            one(a, q)
            if stage_ann:
                cur.execute(sql_ann[a.key], (q,), prepare=True)
                cur.fetchall()

    my = list(range(w, len(qv), n_clients))
    block = cfg["block"]
    # Same block count on every worker, or the barrier would deadlock on an uneven split.
    per_worker = (len(qv) + n_clients - 1) // n_clients
    n_blocks = (per_worker + block - 1) // block
    for b in range(n_blocks):
        qs = my[b * block:(b + 1) * block]
        order = arms[b % len(arms):] + arms[:b % len(arms)]
        for a in order:
            if cold and n_clients > 1 and w == 0:
                drop_caches()  # at >1 client: once per phase, before all clients start
            if barrier is not None:
                barrier.wait()
            pending = []
            p0 = time.perf_counter_ns()
            busy = 0.0
            for qi in qs:
                rows, ids, f_us, d_us = one(a, qv[qi])
                cpu_c, cpu_s = one.cpu
                busy += f_us + d_us
                ann_us = None
                if stage_ann:
                    t0 = time.perf_counter_ns()
                    cur.execute(sql_ann[a.key], (qv[qi],), prepare=True)
                    cur.fetchall()
                    ann_us = (time.perf_counter_ns() - t0) / 1e3
                pending.append((qi, rows, ids, f_us, d_us, ann_us, cpu_c, cpu_s))
            p1 = time.perf_counter_ns()
            phases.append({"arm": a.key, "block": b, "client": w, "t0": p0, "t1": p1,
                           "busy_us": busy, "n": len(qs)})
            # The ID check runs after the phase clock stops, so it never counts as idle time in
            # the throughput (it would pull the Little's-law ratio below 1).
            for qi, rows, ids, f_us, d_us, ann_us, cpu_c, cpu_s in pending:
                bad = sum(not np.array_equal(x, data.ids(r[0])) for r, x in zip(rows, ids))
                mism[a.key] += bad
                recs.append({"arm": a.key, "q": qi, "client": w, "block": b,
                             "fetch_us": f_us, "decode_us": d_us, "e2e_us": f_us + d_us,
                             "ann_us": ann_us, "rows": len(rows),
                             "client_cpu_us": cpu_c, "server_cpu_us": cpu_s,
                             "wire_bytes": wire_bytes(a, rows, data),
                             "mismatched_rows": bad})
            pending.clear()
        if w == 0:
            line = "  ".join(
                f"{a.key} {np.median([r['e2e_us'] for r in recs if r['arm'] == a.key]) / 1e3:.2f}"
                for a in arms)
            print(f"    block {b + 1}/{n_blocks} (client 0, {len(recs)} queries): "
                  f"e2e p50 ms  {line}", flush=True)
    conn.close()
    out_q.put({"recs": recs, "phases": phases, "mism": mism})


def run_config(cfg: dict, shared: dict) -> dict:
    n = cfg["clients"]
    if n == 1:
        q = _ListQ()
        worker(0, 1, cfg, shared, None, q)
        parts = q.items
    else:
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(n)
        oq = ctx.Queue()
        procs = [ctx.Process(target=worker, args=(w, n, cfg, shared, barrier, oq)) for w in range(n)]
        for p in procs:
            p.start()
        parts = [oq.get() for _ in procs]
        for p in procs:
            p.join()
    recs = [r for p in parts for r in p["recs"]]
    phases = [ph for p in parts for ph in p["phases"]]
    mism = {k: sum(p["mism"][k] for p in parts) for k in parts[0]["mism"]}
    return {"recs": recs, "phases": phases, "mism": mism}


class _ListQ:
    def __init__(self):
        self.items = []

    def put(self, x):
        self.items.append(x)


# ── summary ─────────────────────────────────────────────────────────────────────────────────


def paired(recs, a: str, b: str) -> dict:
    """Per query: e2e of b / e2e of a (speedup of a over b), same query vector, same client."""
    x = {(r["q"], r["client"]): r["e2e_us"] for r in recs if r["arm"] == a}
    y = {(r["q"], r["client"]): r["e2e_us"] for r in recs if r["arm"] == b}
    keys = sorted(set(x) & set(y))
    if not keys:
        return {}
    xa, ya = np.array([x[k] for k in keys]), np.array([y[k] for k in keys])
    ratio = ya / xa
    boot = np.random.default_rng(0).choice(ratio, size=(2000, len(ratio)))
    lo, hi = np.percentile(np.median(boot, axis=1), [2.5, 97.5])
    wins, losses = int((xa < ya).sum()), int((xa > ya).sum())
    m = wins + losses
    z = (wins - m / 2) / np.sqrt(m / 4) if m else 0.0
    return {"n": len(keys), "faster_in": wins, "slower_in": losses, "sign_z": float(z),
            "median_speedup": float(np.median(ratio)), "speedup_ci95": [float(lo), float(hi)],
            "p50_ratio": float(np.median(ya) / np.median(xa))}


def summarize(cfg: dict, res: dict, store: dict) -> dict:
    recs, phases = res["recs"], res["phases"]
    out = {}
    for a in ARMS:
        if a.key not in cfg["arms"]:
            continue
        rs = [r for r in recs if r["arm"] == a.key]
        e2e = np.array([r["e2e_us"] for r in rs])
        dec = np.array([r["decode_us"] for r in rs])
        ph = [p for p in phases if p["arm"] == a.key]
        # Throughput per phase: queries finished by all clients / phase wall time.
        walls, nq, busy = 0.0, 0, 0.0
        for b in sorted({p["block"] for p in ph}):
            pb = [p for p in ph if p["block"] == b]
            walls += (max(p["t1"] for p in pb) - min(p["t0"] for p in pb)) / 1e9
            nq += sum(p["n"] for p in pb)
            busy += sum(p["busy_us"] for p in pb) / 1e6
        thr = nq / walls if walls else float("nan")
        st = store.get(a.table, {}) if a.coltype else {}
        out[a.key] = {
            "label": a.label,
            "e2e_us": P.dist(e2e),
            "fetch_us": P.dist([r["fetch_us"] for r in rs]),
            "decode_us": P.dist(dec),
            "ann_us": P.dist([r["ann_us"] for r in rs if r["ann_us"] is not None]),
            "decode_share_p50": float(np.median(dec / e2e)),
            "client_cpu_us_p50": P.pct([r["client_cpu_us"] for r in rs], 50),
            "server_cpu_us_p50": (P.pct([r["server_cpu_us"] for r in rs], 50)
                                  if rs and rs[0]["server_cpu_us"] is not None else None),
            "wire_bytes_per_query_p50": float(np.median([r["wire_bytes"] for r in rs])),
            "wire_bytes_per_row": float(np.sum([r["wire_bytes"] for r in rs]) /
                                        max(1, np.sum([r["rows"] for r in rs]))),
            "rows_per_query_min": int(min(r["rows"] for r in rs)),
            "body_bytes_per_row": st.get("body_bytes_per_row"),
            "throughput_qps": thr,
            # Little's law: mean queries in flight = throughput x mean latency. Busy time / wall
            # gives the same thing measured directly; both should be ~N if clients never idle.
            "littles_in_flight": thr * float(e2e.mean()) / 1e6,
            "littles_ratio": (thr * float(e2e.mean()) / 1e6) / cfg["clients"],
            "busy_ratio": busy / (walls * cfg["clients"]) if walls else float("nan"),
            "mismatched_rows": res["mism"][a.key],
        }
    for key in out:
        if key != BASELINE:
            out[key]["vs_text_lz4"] = paired(recs, key, BASELINE)
        if key != THREAT:
            out[key]["vs_bytea_raw24"] = paired(recs, key, THREAT)
    return out


def print_table(cfg: dict, summ: dict) -> None:
    print(f"\nk={cfg['k']}, {cfg['clients']} client(s), {cfg['rows']} rows, "
          f"{summ[next(iter(summ))]['e2e_us']['n']} queries per arm, tokenizer {cfg['tokenizer']}, "
          f"cache sweep {'on' if cfg['sweep'] else 'off'}, regime {'COLD' if cfg['cold'] else 'warm'}, "
          f"layout {cfg['layout']}, data {cfg['data'] or 'built-in'}", flush=True)
    hdr = (f"  {'arm':<30} {'e2e p50':>8} {'p90':>7} {'p99':>7} {'fetch p50':>9} {'decode p50':>10} "
           f"{'decode':>6} {'ann p50':>7} {'stored B':>8} {'wire B':>7} {'x vs lz4':>8} "
           f"{'x vs raw24':>10} {'qps':>7} {'Little':>6} {'busy':>5} {'cli CPU':>7} {'srv CPU':>7}")
    print(hdr + "\n  " + "-" * (len(hdr) - 2), flush=True)
    for key, s in summ.items():
        e, f, d = s["e2e_us"], s["fetch_us"], s["decode_us"]
        ann = s["ann_us"]["p50"] / 1e3 if s["ann_us"]["n"] else float("nan")
        v1 = s.get("vs_text_lz4", {}).get("median_speedup", 1.0)
        v2 = s.get("vs_bytea_raw24", {}).get("median_speedup", 1.0)
        sb = s["body_bytes_per_row"]
        print(f"  {s['label']:<30} {e['p50'] / 1e3:>8.2f} {e['p90'] / 1e3:>7.2f} {e['p99'] / 1e3:>7.2f} "
              f"{f['p50'] / 1e3:>9.2f} {d['p50'] / 1e3:>10.2f} {100 * s['decode_share_p50']:>5.0f}% "
              f"{ann:>7.2f} {(sb if sb is not None else 0):>8.0f} {s['wire_bytes_per_row']:>7.0f} "
              f"{v1:>7.2f}x {v2:>9.2f}x {s['throughput_qps']:>7.0f} {s['littles_ratio']:>6.2f} "
              f"{s['busy_ratio']:>5.2f} {s['client_cpu_us_p50'] / 1e3:>7.2f} "
              f"{(s['server_cpu_us_p50'] or float('nan')) / 1e3:>7.2f}",
              flush=True)
    bad = {k: s["mismatched_rows"] for k, s in summ.items() if s["mismatched_rows"]}
    short = {k: s["rows_per_query_min"] for k, s in summ.items() if s["rows_per_query_min"] < cfg["k"]}
    print("  check: " + (f"MISMATCHED ROWS {bad}" if bad else
                         "every arm returned exactly the loaded ids for every fetched row")
          + (f"; FEWER THAN k ROWS {short}" if short else ""), flush=True)
    print("  ms; 'decode' = decode p50 share of e2e (per query); 'x vs' = median per-query speedup "
          "(>1 = faster than that baseline); Little = throughput x mean e2e / clients, busy = "
          "time inside queries / (clients x wall); both should be ~1.00 at >1 client (at 1 client "
          "the cache sweep and the ANN-only query are uncounted time); cli/srv CPU = p50 CPU ms per "
          "query in the client process (all threads) and in its PostgreSQL backend", flush=True)
    f = summ.get("pgtoken_freq")
    for base, label in (("vs_text_lz4", "text (lz4)"), ("vs_bytea_raw24", "bytea raw24")):
        v = (f or {}).get(base) or {}
        if v:
            print(f"  pgtoken freq vs {label}: faster in {v['faster_in']}/{v['n']} queries, median "
                  f"per-query speedup {v['median_speedup']:.2f}x [{v['speedup_ci95'][0]:.2f}, "
                  f"{v['speedup_ci95'][1]:.2f}], ratio of p50s {v['p50_ratio']:.2f}x", flush=True)


def explain(conn, cfg: dict, queries: list[str], n: int = 20) -> dict:
    """Server-side execution time and buffers (EXPLAIN ANALYZE) of each arm's query. EXPLAIN
    neither detoasts nor sends the output column, so this is ANN + heap/PK visits only; fetch
    minus this is roughly detoast/decompress + wire + client row building. In the cold regime the
    OS cache is dropped before each EXPLAIN, so read blocks are real device reads (or OS cache
    misses at least; shared_buffers is small but not emptied)."""
    k = cfg["k"]
    conn.execute(f"SET hnsw.ef_search = {max(cfg['ef_search'], k)}")
    out = {}
    for a in ARMS:
        if a.key not in cfg["arms"]:
            continue
        q_sql = arm_sql(a, cfg["tag"], cfg["layout"], k)[0]
        ex, hit, read = [], [], []
        for q in queries[:n]:
            if cfg["cold"]:
                drop_caches()
            plan = conn.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {q_sql}",
                                (q,)).fetchone()[0][0]
            ex.append(plan["Execution Time"] * 1e3)
            hit.append(plan["Plan"].get("Shared Hit Blocks", 0))
            read.append(plan["Plan"].get("Shared Read Blocks", 0))
        out[a.key] = {"exec_us_p50": P.pct(ex, 50), "shared_hit_blocks_p50": P.pct(hit, 50),
                      "shared_read_blocks_p50": P.pct(read, 50)}
    return out


def capacity(results: list, k: int, targets: list, quiet: bool = False) -> dict:
    """Throughput at a fixed p99 target: for each arm, the highest qps over the unswept client
    counts whose e2e p99 stays under the target."""
    rs = [r for r in results if r["cfg"]["k"] == k and not r["cfg"]["sweep"]]
    out = {}
    for key in (rs[0]["summary"] if rs else {}):
        pts = sorted((r["cfg"]["clients"], r["summary"][key]["throughput_qps"],
                      r["summary"][key]["e2e_us"]["p99"] / 1e3) for r in rs)
        best = {}
        for t in targets:
            ok = [(q, c) for c, q, p in pts if p <= t]
            best[t] = max(ok) if ok else None
        out[key] = {"points": pts, "best_qps_under_p99": best}
    if not quiet and out:
        print(f"\nk={k}: best throughput (queries/s) with e2e p99 under a target, over "
              f"{sorted({r['cfg']['clients'] for r in rs})} clients (clients used in brackets)",
              flush=True)
        print(f"  {'arm':<34}" + "".join(f"{'p99<' + str(int(t)) + 'ms':>16}" for t in targets),
              flush=True)
        lab = {a.key: a.label for a in ARMS}
        for key, v in out.items():
            cells = "".join(f"{(f'{b[0]:.0f} ({b[1]})' if b else '-'):>16}"
                            for b in v["best_qps_under_p99"].values())
            print(f"  {lab[key]:<34}{cells}", flush=True)
    return out


def pick_tokenizer() -> tuple[str, str]:
    import glob

    hits = sorted(glob.glob(os.path.join(P.HERE, "runs", "*o200k-check", "result.json"))
                  + glob.glob(os.path.join(P.HERE, "runs", "smoke", "*o200k-check", "result.json")),
                  key=os.path.getmtime)
    for path in reversed(hits):
        with open(path) as f:
            r = json.load(f)
        if r.get("pgtoken_tokenizer_env") in ("hf-o200k", "tiktoken-o200k"):
            return r["pgtoken_tokenizer_env"], f"fastest exact in {os.path.relpath(path, P.HERE)}"
    return "hf-o200k", "no check_o200k.py result found; defaulting"


# ── main ────────────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None, help="prep_data.py directory; default: built-in C4 chunks")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=384, help="random-vector dims when the data has none")
    ap.add_argument("--layout", choices=["single", "split"], default="single",
                    help="single: one (id, emb, body) table + HNSW per arm; split: one shared "
                         "(id, emb) table + HNSW, and an (id, body) table per arm")
    ap.add_argument("--tag", default="rag", help="table name prefix")
    ap.add_argument("--k", type=int, nargs="+", default=[100])
    ap.add_argument("--clients", type=int, nargs="+", default=[1])
    ap.add_argument("--queries", type=int, default=300, help="timed queries per arm per config")
    ap.add_argument("--warmup", type=int, default=20, help="untimed queries per arm per client")
    ap.add_argument("--block", type=int, default=10, help="queries per arm per round-robin turn")
    ap.add_argument("--ef-search", type=int, default=100, help="hnsw.ef_search; raised to k if lower")
    ap.add_argument("--tokenizer", default="auto", choices=["auto", "hf-o200k", "tiktoken-o200k"],
                    help="client tokenizer for the text arms; auto = the fastest exact one in the "
                         "newest check_o200k.py result")
    ap.add_argument("--arms", nargs="+", default=[a.key for a in ARMS])
    ap.add_argument("--sweep", choices=["auto", "on", "off", "both"], default="both",
                    help="64 MB CPU-cache sweep before each timed query. both = at 1 client run "
                         "swept (serving-cold tables) and unswept (steady-state serving); >1 "
                         "client always unswept. auto = swept at 1 client only")
    ap.add_argument("--cold", action="store_true",
                    help="index warm, bodies cold: run each query's ANN untimed, then drop the OS "
                         "page cache before the timed query (1 client) or each phase (>1 client); "
                         "run the server with a small shared_buffers (128MB)")
    ap.add_argument("--disk", default="nvme0n1", help="block device for read counters")
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--p99-targets", type=float, nargs="+", default=[10, 25, 50, 100],
                    help="ms; report the best throughput over client counts with p99 under each")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--no-reuse", action="store_true", help="reload tables even if they match")
    ap.add_argument("--smoke", action="store_true", help="write under runs/smoke/")
    ap.add_argument("--name", default="rag-e2e")
    args = ap.parse_args()

    out_dir = P.run_dir(args.name, args.smoke)
    print(f"run dir: {out_dir}", flush=True)
    if args.tokenizer == "auto":
        args.tokenizer, src = pick_tokenizer()
        print(f"tokenizer for the text arms: {args.tokenizer} ({src})", flush=True)
    data = Data(args.data, args.rows, args.dim, args.seed)
    lens = data.token_lengths()
    tok_dist = {p: float(np.percentile(lens, p)) for p in (1, 10, 25, 50, 75, 90, 99)}
    print(f"data: {data.source}; {args.rows} rows, dim {data.dim}; o200k tokens per row "
          + ", ".join(f"p{p} {v:.0f}" for p, v in tok_dist.items())
          + f", mean {lens.mean():.1f}", flush=True)

    with C.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
        settings = {k: conn.execute(f"SHOW {k}").fetchone()[0] for k in (
            "server_version", "shared_buffers", "work_mem", "maintenance_work_mem",
            "max_connections", "synchronous_commit", "jit", "default_toast_compression",
            "max_parallel_workers_per_gather", "max_parallel_maintenance_workers",
            "effective_cache_size")}
        for ext in ("vector", "pgtoken"):
            settings[ext] = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = %s", (ext,)).fetchone()[0]
        print(f"server: {settings}", flush=True)
        if args.cold and settings["shared_buffers"] not in ("128MB", "256MB"):
            print(f"WARNING: --cold with shared_buffers {settings['shared_buffers']}: PostgreSQL's "
                  "own cache stays warm; restart the server with PG_SHARED_BUFFERS=128MB", flush=True)
        vocab = ensure_vocabs(conn)
        load = load_tables(conn, data, vocab, args.tag, args.layout, not args.no_reuse)
        store = storage(conn, args.rows, args.tag, args.layout)
        for t, st in store.items():
            print(f"  storage {t}: " + ", ".join(
                f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in st.items()),
                flush=True)
        probe = probe_single(conn, args.tag) if args.layout == "split" else None
        if probe:
            for t, st in probe.items():
                print(f"  if emb and body shared one table ({t}, first 20000 rows): " + ", ".join(
                    f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in st.items()),
                    flush=True)
        ex_rows = examples(conn, data, args.tag, args.layout)
        for e in ex_rows:
            print(f"  example p{e['pct']} row {e['row']}: {e['tokens']} tokens, {e['utf8_bytes']} B "
                  f"UTF-8, stored {e['stored_bytes']}: {e['text'][:160]!r}", flush=True)
        if not args.cold:
            prewarm(conn, args.tag, args.layout)

    config = {"args": vars(args), "settings": settings, "machine": P.machine(),
              "data": {"source": data.source, "rows": args.rows, "dim": data.dim,
                       "tokens": tok_dist, "tokens_mean": float(lens.mean()),
                       "meta": getattr(data, "meta", None)},
              "arms": [asdict(a) for a in ARMS], "load": load, "storage": store,
              "single_table_probe": probe, "examples": ex_rows,
              "cpus": {"client_affinity": sorted(os.sched_getaffinity(0)),
                       "server_cpuset": os.environ.get("SERVER_CPUS_RECORDED")}}
    P.write_json(os.path.join(out_dir, "config.json"), config)
    if args.load_only:
        return 0

    nq_warm = args.warmup * max(args.clients)
    shared = {
        "ranking": vocab["ranking"],
        "warm_queries": data.queries(nq_warm, 0),
        "queries": data.queries(args.queries, nq_warm),
    }
    results = []
    jsonl = open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8")
    for k in args.k:
        for n, sweep in [(n, sw) for n in args.clients for sw in (
                [True, False] if (args.sweep == "both" and n == 1) else
                [n == 1] if args.sweep in ("auto", "both") else [args.sweep == "on"])]:
            cfg = {"rows": args.rows, "k": k, "clients": n, "tokenizer": args.tokenizer,
                   "ef_search": args.ef_search, "sweep": sweep, "cold": args.cold,
                   "stage_ann": n == 1 and not args.cold, "block": args.block,
                   "arms": args.arms, "data": args.data, "dim": data.dim, "seed": args.seed,
                   "tag": args.tag, "layout": args.layout}
            print(f"\n== k={k}, clients={n}: {args.queries} queries per arm, blocks of "
                  f"{args.block}, sweep {'on' if sweep else 'off'}, "
                  f"{'COLD' if args.cold else 'warm'}", flush=True)
            t0 = time.time()
            d0 = disk_stats(args.disk)
            res = run_config(cfg, shared)
            d1 = disk_stats(args.disk)
            wall = time.time() - t0
            summ = summarize(cfg, res, store)
            with C.connect() as conn:
                ex = explain(conn, cfg, shared["queries"])
            for key, s_ in summ.items():
                s_["server_exec"] = ex.get(key)
            disk = {"device": args.disk, "reads": d1[0] - d0[0],
                    "read_mb": (d1[1] - d0[1]) * 512 / 1e6, "wall_s": wall,
                    "read_iops": (d1[0] - d0[0]) / wall,
                    "read_mb_s": (d1[1] - d0[1]) * 512 / 1e6 / wall,
                    "reads_per_query": (d1[0] - d0[0]) / max(1, len(res["recs"]))}
            print_table(cfg, summ)
            print(f"  disk {args.disk}: {disk['reads']} reads, {disk['read_mb']:.0f} MB in "
                  f"{wall:.0f} s = {disk['read_iops']:.0f} read IOPS, {disk['read_mb_s']:.1f} MB/s, "
                  f"{disk['reads_per_query']:.1f} reads per query (gp3 baseline: 3000 IOPS, "
                  f"125 MB/s)", flush=True)
            print("  EXPLAIN (server, ANN + row visits, no detoast): " + ", ".join(
                f"{key} {v['exec_us_p50'] / 1e3:.2f} ms/{v['shared_read_blocks_p50']:.0f} reads"
                for key, v in ex.items()), flush=True)
            for r in res["recs"]:
                jsonl.write(json.dumps({"k": k, "clients": n, "cold": args.cold, "sweep": sweep,
                                        **r}) + "\n")
            jsonl.flush()
            results.append({"cfg": cfg, "summary": summ, "disk": disk, "phases": res["phases"]})
            P.write_json(os.path.join(out_dir, "results.json"),
                         {"config": config, "results": results})
    for k in args.k:
        capacity(results, k, args.p99_targets)
    P.write_json(os.path.join(out_dir, "results.json"), {"config": config, "results": results,
                 "capacity": {k: capacity(results, k, args.p99_targets, quiet=True) for k in args.k}})
    jsonl.close()
    print(f"\nwrote {out_dir}/results.json and queries.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
