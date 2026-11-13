#!/usr/bin/env python3
"""A BPE tokenizer trained straight on the corpus text, instead of the model's gpt2 ids.

The alternative to bench_token_bpe.py (BPE on top of gpt2 ids). Here the stored ids belong to a
new tokenizer that the model does not know, so a model-facing read must turn them back into text
and run the model's tokenizer again. This measures what that buys in size and costs in speed.

Variants, each trained per domain (prose, code) on the same 8000 train chunks:
    gpt2-style   byte-level BPE with gpt2's word-splitting regex (merges never cross a word)
    superword    byte-level BPE with no word split (merges may span spaces, e.g. " of the")
at vocab 50257 (gpt2's size) and 65536 (the largest that fits raw16, 2 bytes an id).

Ratio = UTF-8 bytes / stored bytes, including pgtoken's 12-byte header, on the same 200 held-out
test chunks per domain as bench_token_bpe.py. Ids are stored raw16, or freq ranked on train.

Two steps, because tokenizers v1 cannot train and 0.21 is slow to encode:
  1) train + sizes (tokenizers 0.21):
       uv run --no-project --with 'tokenizers<0.22' --with numpy --with tiktoken --with ./py \\
           python benchmarks/bench_text_bpe.py train
  2) times (project env, tokenizers v1, loads the saved tokenizer.json files):
       uv run python benchmarks/bench_text_bpe.py time
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time

import numpy as np
import pgtoken_rs as R
import tiktoken
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(os.environ.get("TOKEN_STORAGE_REPO", os.path.expanduser("~/projects/token-storage")),
                      "data", "corpus")
SAVE = os.path.join(HERE, "fixture", "text_bpe")
OUT = os.path.join(HERE, "text_bpe_results.json")
N_GPT2 = 50257


def load(name):
    return np.load(os.path.join(CORPUS, f"{name}.npy")).astype(np.int64)


def test_chunks(n=200):
    # Same chunks as bench_token_bpe.py: one rng, prose drawn first, then code.
    rng = np.random.default_rng(9012)
    out = {}
    for d in ("prose", "code"):
        arr = load(f"{d}_test")
        m = len(arr) // 512
        out[d] = [arr[s:s + 512] for s in rng.choice(m, size=min(n, m), replace=False) * 512]
    return out


def rank_table(ids: np.ndarray, size: int) -> R.Table:
    counts = np.bincount(ids, minlength=size)
    seen = np.nonzero(counts)[0]
    order = seen[np.argsort(-counts[seen], kind="stable")].tolist()
    return R.Table.from_bytes(b"TNTT" + bytes([1, 1, 0, 0]) + struct.pack("<I", len(order))
                              + struct.pack(f"<{len(order)}I", *order))


def p50(xs):
    return float(np.median(xs))


def best_of(fn, n=5):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def train_step(args) -> int:
    from tokenizers import decoders, models, pre_tokenizers, trainers

    enc = tiktoken.get_encoding("r50k_base")
    tests = test_chunks(args.chunks)
    texts = {d: [enc.decode(c.tolist()) for c in cs] for d, cs in tests.items()}
    utf8 = {d: [len(t.encode()) for t in ts] for d, ts in texts.items()}
    os.makedirs(SAVE, exist_ok=True)
    res = {"baselines": {}, "bpe": {}, "examples": {}}

    for d, cs in tests.items():
        train = load(f"{d}_train")[: args.train_chunks * 512]
        ft = rank_table(train, N_GPT2)
        res["baselines"][d] = {
            "gpt2 ids per chunk": 512.0,
            "gpt2 raw16": p50([u / len(R.encode(c.astype(np.uint32), "raw16", 1)) for u, c in zip(utf8[d], cs)]),
            "gpt2 freq": p50([u / len(R.encode(c.astype(np.uint32), "freq", 1, ft)) for u, c in zip(utf8[d], cs)]),
        }
        print(f"{d}: gpt2 raw16 {res['baselines'][d]['gpt2 raw16']:.2f}  gpt2 freq {res['baselines'][d]['gpt2 freq']:.2f}",
              flush=True)

    for d_train in ("prose", "code"):
        train = load(f"{d_train}_train")[: args.train_chunks * 512]
        train_texts = [enc.decode(train[i:i + 512].tolist()) for i in range(0, len(train) - 511, 512)]
        for style in ("gpt2-style", "superword"):
            for V in args.vocab:
                tok = Tokenizer(models.BPE())
                tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=(style == "gpt2-style"))
                tok.decoder = decoders.ByteLevel()
                trainer = trainers.BpeTrainer(vocab_size=V, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                              show_progress=False)
                t0 = time.perf_counter()
                tok.train_from_iterator(train_texts, trainer=trainer)
                train_s = time.perf_counter() - t0
                name = f"{d_train}_{style}_{V}"
                tok.save(os.path.join(SAVE, f"{name}.json"))
                size = tok.get_vocab_size()
                train_ids = np.concatenate([tok.encode(t).ids for t in train_texts[:2000]]).astype(np.int64)
                ft = rank_table(train_ids, size)
                key = f"{d_train} text, {style}, vocab {V}"
                res["bpe"][key] = {"train_seconds": train_s, "file": f"{name}.json"}
                print(f"\n{key}: trained in {train_s:.0f} s", flush=True)
                for d_test, ts in texts.items():
                    ids = [np.asarray(tok.encode(t).ids, dtype=np.uint32) for t in ts]
                    bad = sum(tok.decode(i.tolist()) != t for i, t in zip(ids, ts))
                    row = {"ids_per_chunk_p50": p50([len(i) for i in ids]), "text_roundtrip_failures": bad}
                    if size <= 65536:
                        row["ratio raw16"] = p50([u / len(R.encode(i, "raw16", 1)) for u, i in zip(utf8[d_test], ids)])
                    row["ratio freq"] = p50([u / len(R.encode(i, "freq", 1, ft)) for u, i in zip(utf8[d_test], ids)])
                    res["bpe"][key][d_test] = row
                    print(f"  test {d_test:<6} " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()), flush=True)
                # Concrete tokens: the longest frequent ones on its own domain's test set.
                ids_all = np.concatenate([tok.encode(t).ids for t in texts[d_train]])
                counts = np.bincount(ids_all, minlength=size)
                lens = np.array([len(tok.decode([i]).encode()) for i in range(size)])
                top = np.argsort(-(counts * np.maximum(lens - 1, 0)))[:8]
                res["examples"][key] = [{"text": tok.decode([int(i)]), "uses": int(counts[i])} for i in top]
                print("  most bytes covered: " + "  ".join(f"{e['text']!r}x{e['uses']}" for e in res["examples"][key]),
                      flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), **res}, f, indent=1)
    print(f"\nwrote {OUT}")
    return 0


def time_step(args) -> int:
    """v1 times: store (text -> new ids -> bytes) and model read (bytes -> new ids -> text -> gpt2 ids)."""
    enc = tiktoken.get_encoding("r50k_base")
    with open(OUT, encoding="utf-8") as f:
        res = json.load(f)
    tests = test_chunks(args.chunks)
    n = args.time_chunks
    res["times_v1"] = {}
    for d, cs in tests.items():
        cs = cs[:n]
        texts = [enc.decode(c.tolist()) for c in cs]
        # How often does re-tokenizing the text give back the exact gpt2 ids that were stored?
        same = sum(enc.encode(t, disallowed_special=()) == c.tolist() for t, c in zip(texts, cs))
        gp = {"gpt2 ids exactly recovered from text": f"{same}/{len(cs)}",
              "tiktoken gpt2 encode text": p50([best_of(lambda t=t: enc.encode(t, disallowed_special=())) for t in texts])}
        ft = rank_table(load(f"{d}_train")[:2_000_000], N_GPT2)
        blobs = [R.encode(c.astype(np.uint32), "freq", 1, ft) for c in cs]
        gp["gpt2 freq encode"] = p50([best_of(lambda c=c: R.encode(c.astype(np.uint32), "freq", 1, ft)) for c in cs])
        gp["gpt2 freq decode"] = p50([best_of(lambda b=b: R.decode(b, ft)) for b in blobs])
        res["times_v1"][f"{d}: gpt2 baseline"] = gp
        print(f"{d}: " + "  ".join(f"{k} {v if isinstance(v, str) else f'{v:.1f} us'}" for k, v in gp.items()), flush=True)
        for key, row in res["bpe"].items():
            if not key.startswith(d):
                continue
            tok = Tokenizer.from_file(os.path.join(SAVE, row["file"]))
            # First-seen encode: v1 caches words, so time each chunk once on a fresh tokenizer copy
            # is not possible; best_of on repeats measures the warm-cache cost, first call the cold.
            cold = []
            for t in texts:
                t0 = time.perf_counter()
                tok.encode(t)
                cold.append((time.perf_counter() - t0) * 1e6)
            warm = [best_of(lambda t=t: tok.encode(t)) for t in texts]
            ids = [tok.encode(t).ids for t in texts]
            dec = [best_of(lambda i=i: tok.decode(i)) for i in ids]
            back = [best_of(lambda i=i: enc.encode(tok.decode(i), disallowed_special=())) for i in ids]
            tr = {"encode first seen": p50(cold), "encode repeat": p50(warm), "decode to text": p50(dec),
                  "decode to text + gpt2 retokenize": p50(back)}
            res["times_v1"][key] = tr
            print(f"  {key:<38} " + "  ".join(f"{k} {v:.1f} us" for k, v in tr.items()), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "time"])
    ap.add_argument("--train-chunks", type=int, default=8000)
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--time-chunks", type=int, default=50)
    ap.add_argument("--vocab", type=int, nargs="+", default=[50257, 65536])
    args = ap.parse_args()
    return train_step(args) if args.step == "train" else time_step(args)


if __name__ == "__main__":
    raise SystemExit(main())
