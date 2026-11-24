#!/usr/bin/env python3
"""Build a dataset directory for rag_e2e.py once, so every wave reads the same rows.

    dbpedia   KShivendu/dbpedia-entities-openai-1M (HF): 1M DBpedia entities, title + abstract,
              with real OpenAI text-embedding-ada-002 vectors (1536 dims). `--holdout` entities
              (random, seeded) are removed from the table and their embeddings become the query
              vectors, so a query is a real entity embedding that is not itself in the table.
    c4        allenai/c4 'en' train shards (HF): documents joined with a blank line, encoded with
              o200k, cut into consecutive non-overlapping windows of `--chunk-tokens` ids, each
              window decoded to text and re-encoded (so stored ids == tokenize(text)). Every
              chunk is distinct: nothing is cycled. No embeddings; rag_e2e.py draws seeded random
              unit vectors for these rows.

Output, in <out>/<name>/:
    meta.json          what was built, from what, token-length distribution
    text.bin           UTF-8 bodies, concatenated;  text_off.npy (int64, n + 1)
    ids.bin            o200k ids (uint32 LE);         ids_off.npy  (int64, n + 1)
    emb.npy            float32 (n, dim)                       (dbpedia only)
    queries.npy        float32 (q, dim) query vectors         (dbpedia only)
    queries.jsonl      the held-out entities the queries came from

Tokenizer: HF `tokenizers` with the pinned o200k tokenizer.json (pgcommon.hf_tokenizer_json),
which paper/check_o200k.py showed ID-for-ID equal to tiktoken o200k_base; encode_batch uses all
the client's cores.

Usage:
    python paper/prep_data.py dbpedia --out /data
    python paper/prep_data.py c4 --out /data --rows 1000000 --shards 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperlib as P  # noqa: E402

C = P.C

DBPEDIA = ("KShivendu/dbpedia-entities-openai-1M", "af9b8869cc2d8debbd254d77737865bb09a2067f")


def download(url: str, path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
            return path
        except Exception as e:  # noqa: BLE001
            print(f"    retry {attempt + 1} {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"could not download {url}")


def hf_tok():
    return C._hf_file(C.hf_tokenizer_json(), "hf-o200k", n_vocab=200019).hf


class Writer:
    def __init__(self, d: str):
        self.d = d
        os.makedirs(d, exist_ok=True)
        self.ft = open(os.path.join(d, "text.bin"), "wb")
        self.fi = open(os.path.join(d, "ids.bin"), "wb")
        self.toff, self.ioff = [0], [0]

    def add(self, texts, id_lists):
        for t, ids in zip(texts, id_lists):
            b = t.encode()
            self.ft.write(b)
            self.toff.append(self.toff[-1] + len(b))
            a = np.asarray(ids, dtype="<u4")
            self.fi.write(a.tobytes())
            self.ioff.append(self.ioff[-1] + len(a))

    @property
    def n(self):
        return len(self.toff) - 1

    def close(self):
        self.ft.close()
        self.fi.close()
        np.save(os.path.join(self.d, "text_off.npy"), np.asarray(self.toff, dtype=np.int64))
        np.save(os.path.join(self.d, "ids_off.npy"), np.asarray(self.ioff, dtype=np.int64))
        lens = np.diff(np.asarray(self.ioff))
        tb = np.diff(np.asarray(self.toff))
        return {"rows": self.n,
                "tokens": {p: float(np.percentile(lens, p)) for p in (1, 10, 25, 50, 75, 90, 99)},
                "tokens_mean": float(lens.mean()), "text_bytes_mean": float(tb.mean()),
                "text_bytes": {p: float(np.percentile(tb, p)) for p in (25, 50, 90)}}


def prep_dbpedia(args) -> dict:
    import pyarrow.parquet as pq

    repo, rev = DBPEDIA
    files = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}"))
    shards = sorted(s["rfilename"] for s in files["siblings"] if s["rfilename"].endswith(".parquet"))
    raw = os.path.join(args.out, "raw", "dbpedia")
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        paths = list(ex.map(lambda f: download(
            f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{f}",
            os.path.join(raw, os.path.basename(f))), shards))
    print(f"  downloaded {len(paths)} shards in {time.time() - t0:.0f} s", flush=True)

    n_total = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    rng = np.random.default_rng(args.seed)
    held = set(rng.choice(n_total, size=args.holdout, replace=False).tolist())
    tok = hf_tok()
    d = os.path.join(args.out, args.name or "dbpedia")
    w = Writer(d)
    dim = None
    emb_f = None
    queries, qmeta = [], []
    g = 0
    for p in paths:
        tbl = pq.read_table(p, columns=["_id", "title", "text", "openai"])
        ids_ = tbl.column("_id").to_pylist()
        titles = tbl.column("title").to_pylist()
        texts = tbl.column("text").to_pylist()
        col = tbl.column("openai").combine_chunks()
        dim = dim or len(col[0])
        emb = col.values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, dim)
        if emb_f is None:
            emb_f = np.lib.format.open_memmap(os.path.join(d, "emb.npy"), mode="w+",
                                              dtype=np.float32, shape=(n_total - len(held), dim))
        keep = []
        for i in range(len(texts)):
            if g + i in held:
                queries.append(emb[i])
                qmeta.append({"_id": ids_[i], "title": titles[i]})
            else:
                keep.append(i)
        bodies = [f"{titles[i]}\n{texts[i]}".replace("\x00", "") for i in keep]
        encs = tok.encode_batch(bodies, add_special_tokens=False)
        w.add(bodies, [e.ids for e in encs])
        emb_f[w.n - len(keep):w.n] = emb[keep]
        g += len(texts)
        print(f"  {os.path.basename(p)}: {len(keep)} rows kept, total {w.n}, "
              f"{time.time() - t0:.0f} s", flush=True)
    emb_f.flush()
    del emb_f
    np.save(os.path.join(d, "queries.npy"), np.stack(queries).astype(np.float32))
    with open(os.path.join(d, "queries.jsonl"), "w") as f:
        for m in qmeta:
            f.write(json.dumps(m) + "\n")
    stats = w.close()
    return {"name": "dbpedia", "source": f"{repo}@{rev}", "dim": dim, "queries": len(queries),
            "queries_from": f"{args.holdout} entities held out at random (seed {args.seed})",
            "body": "title + '\\n' + text", **stats}


def prep_c4(args) -> dict:
    tok = hf_tok()
    tt = C._tiktoken("o200k_base", "tiktoken-o200k").tt
    d = os.path.join(args.out, args.name or "c4")
    w = Writer(d)
    raw = os.path.join(args.out, "raw", "c4")
    ct = args.chunk_tokens
    t0 = time.time()
    carry: list[int] = []
    for s in range(args.shards):
        url = (f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
               f"c4-train.{s:05d}-of-01024.json.gz")
        path = download(url, os.path.join(raw, os.path.basename(url)))
        docs = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line)["text"].replace("\x00", ""))
        for lo in range(0, len(docs), 20000):
            encs = tok.encode_batch([x + "\n\n" for x in docs[lo:lo + 20000]],
                                    add_special_tokens=False)
            stream = carry + [i for e in encs for i in e.ids]
            n_win = len(stream) // ct
            wins = [stream[k * ct:(k + 1) * ct] for k in range(n_win)]
            carry = stream[n_win * ct:]
            # tokenizers 1.x has no decode_batch; tiktoken's threaded byte decode gives the same
            # bytes (ids are identical), and a character split at a window edge is dropped.
            texts = [b.decode("utf-8", errors="ignore")
                     for b in tt.decode_bytes_batch(wins, num_threads=len(os.sched_getaffinity(0)))]
            texts = [t for t in texts if t]
            ids = [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
            take = min(len(texts), args.rows - w.n)
            w.add(texts[:take], ids[:take])
            if w.n >= args.rows:
                break
        print(f"  shard {s}: {len(docs)} docs, total {w.n} chunks, {time.time() - t0:.0f} s",
              flush=True)
        if w.n >= args.rows:
            break
    stats = w.close()
    if stats["rows"] < args.rows:
        print(f"WARNING: only {stats['rows']} chunks from {args.shards} shards", flush=True)
    return {"name": "c4", "source": f"allenai/c4 en train shards 0..{s}", "dim": None,
            "chunk_tokens": ct, **stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["dbpedia", "c4"])
    ap.add_argument("--out", default=os.environ.get("PGTOKEN_DATA", "/data"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--holdout", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()
    t0 = time.time()
    meta = prep_dbpedia(args) if args.dataset == "dbpedia" else prep_c4(args)
    meta.update({"built_s": time.time() - t0, "machine": P.machine(),
                 "tokenizer": "hf-o200k " + C.hf_tokenizer_json()})
    d = os.path.join(args.out, args.name or args.dataset)
    P.write_json(os.path.join(d, "meta.json"), meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "machine"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
