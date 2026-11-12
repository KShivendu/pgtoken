#!/usr/bin/env python3
"""Export a trained token-ID BPE and test chunks as flat binary files for the Rust bench.

Reads a tokenizer.json written by `bench_token_bpe.py --save-dir`, and writes, next to it:

    <name>.merges.u32   [n_gpt2] gpt2 id -> symbol id, then rank-ordered (left, right, result)
    <dom>_test.u32      every 512-token test chunk, back to back, as gpt2 ids
    <name>.v1_out.u32   tokenizers v1's encoding of each chunk: per chunk a length, then symbols

The v1 output is the reference the Rust encoder must match exactly.

Usage (project env, so tokenizers v1 does the reference encode):
    uv run python benchmarks/export_token_bpe_fixture.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "fixture", "token_bpe")
CORPUS = os.path.join(os.environ.get("TOKEN_STORAGE_REPO", os.path.expanduser("~/projects/token-storage")),
                      "data", "corpus")
BASE = 0xF0000
N_GPT2 = 50257


def main() -> int:
    for dom in ("prose", "code"):
        name = f"token_bpe_{dom}_65536"
        spec = json.load(open(os.path.join(DIR, f"{name}.json"), encoding="utf-8"))["model"]
        vocab = spec["vocab"]
        base = np.array([vocab[chr(BASE + g)] for g in range(N_GPT2)], dtype=np.uint32)
        merges = np.array([(vocab[a], vocab[b], vocab[a + b]) for a, b in spec["merges"]], dtype=np.uint32)
        with open(os.path.join(DIR, f"{name}.merges.u32"), "wb") as f:
            f.write(np.array([N_GPT2, len(merges)], dtype=np.uint32).tobytes())
            f.write(base.tobytes())
            f.write(merges.tobytes())

        test = np.load(os.path.join(CORPUS, f"{dom}_test.npy")).astype(np.uint32)
        test = test[: (len(test) // 512) * 512]
        test.tofile(os.path.join(DIR, f"{dom}_test.u32"))

        tok = Tokenizer.from_file(os.path.join(DIR, f"{name}.json"))
        out = []
        for i in range(0, len(test), 512):
            ids = tok.encode("".join(chr(BASE + int(x)) for x in test[i:i + 512]), add_special_tokens=False).ids
            out.append(len(ids))
            out.extend(ids)
        np.array(out, dtype=np.uint32).tofile(os.path.join(DIR, f"{name}.v1_out.u32"))
        print(f"{name}: {len(merges)} merges, {len(test) // 512} test chunks exported", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
