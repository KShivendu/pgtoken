#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.51", "datasets", "numpy", "huggingface_hub"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Are a model's own output token IDs recoverable from the text? Storing text says yes. This checks.

A chat model emits token IDs. A text column stores decode(ids). The next turn, a training job or a
prefix cache then needs IDs again and runs encode(decode(ids)). That is lossless as TEXT, but the
IDs can differ: BPE has one canonical split per string, and sampling can pick a non-canonical one
(e.g. ' Rot' + 'tn' where the tokenizer would write ' Rott' + 'n'). Storing the IDs keeps exactly
what the model produced.

For each model, N prompts (Alpaca instructions, seeded sample) go through the chat template and the
model samples a reply on CPU. Per reply we record:
    reply_exact   encode(decode(reply_ids)) == reply_ids
    convo_exact   encode(decode(prompt_ids + reply_ids)) == prompt_ids + reply_ids
                  (the multi-turn case: the whole history re-tokenized as one string)
    text_lossless decode(encode(decode(ids))) == decode(ids)
plus the first differing position and the token strings around it.

Usage (CPU, ~10-20 min per model on 4 cores):
    uv run paper/exactness.py --n 500 --threads 2
    uv run paper/exactness.py --smoke          # 16 prompts, writes to runs/<date>-exactness-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def git_sha() -> str:
    if os.environ.get("PGTOKEN_GIT_SHA"):
        return os.environ["PGTOKEN_GIT_SHA"]
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def first_diff(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def window(tok, ids: list[int], pos: int, w: int = 3) -> list[str]:
    return tok.convert_ids_to_tokens(ids[max(0, pos - w): pos + w + 1])


def prompts(n: int, seed: int) -> list[str]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)
    out = []
    for i in idx:
        r = ds[i]
        out.append(r["instruction"] + (("\n\n" + r["input"]) if r["input"] else ""))
    return out


def chat_ids(tok, text: str) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    kw = {"enable_thinking": False} if "qwen3" in tok.name_or_path.lower() else {}
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
    if not isinstance(ids, list):  # newer transformers return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def run_model(key: str, ps: list[str], args, out_f) -> dict:
    name = MODELS[key]
    sha = model_info(name).sha
    tok = AutoTokenizer.from_pretrained(name, revision=sha)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, revision=sha, dtype=torch.float32)
    model.eval()
    stop = {tok.eos_token_id, tok.pad_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            stop.add(tid)
    print(f"\n== {key} ({name}@{sha[:10]}), {len(ps)} prompts, sample={not args.greedy}, "
          f"max_new_tokens={args.max_new}", flush=True)
    n = n_reply = n_convo = n_text = 0
    toks_total = 0
    t_start = time.perf_counter()
    for b in range(0, len(ps), args.batch):
        batch = ps[b: b + args.batch]
        pids = [chat_ids(tok, p) for p in batch]
        L = max(len(x) for x in pids)
        inp = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in pids])
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in pids])
        gen_kw = dict(max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id)
        if args.greedy:
            gen_kw.update(do_sample=False)
        else:
            gen_kw.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=att, **gen_kw)
        for j, p_ids in enumerate(pids):
            r = out[j, L:].tolist()
            # cut at the first stop token; keep it out of the stored reply (a chat store keeps
            # the message text, the template adds the end marker again)
            cut = next((k for k, t in enumerate(r) if t in stop), len(r))
            r = r[:cut]
            if not r:
                continue
            text = tok.decode(r, skip_special_tokens=False)
            re_ids = tok(text, add_special_tokens=False)["input_ids"]
            conv = p_ids + r
            conv_text = tok.decode(conv, skip_special_tokens=False)
            re_conv = tok(conv_text, add_special_tokens=False)["input_ids"]
            reply_exact = re_ids == r
            convo_exact = re_conv == conv
            text_lossless = tok.decode(re_ids, skip_special_tokens=False) == text
            n += 1
            n_reply += reply_exact
            n_convo += convo_exact
            n_text += text_lossless
            toks_total += len(r)
            rec = {"model": key, "i": b + j, "prompt": batch[j][:300], "reply_tokens": len(r),
                   "reply_exact": reply_exact, "convo_exact": convo_exact,
                   "text_lossless": text_lossless, "reply_text": text[:2000]}
            if not reply_exact:
                d = first_diff(r, re_ids)
                rec.update(diff_pos=d, len_model=len(r), len_retok=len(re_ids),
                           model_tokens=window(tok, r, d), retok_tokens=window(tok, re_ids, d),
                           n_diff_positions=sum(1 for x, y in zip(r, re_ids) if x != y))
            if not convo_exact:
                d = first_diff(conv, re_conv)
                rec.update(convo_diff_pos=d, convo_in_prompt=d < len(p_ids),
                           convo_model_tokens=window(tok, conv, d),
                           convo_retok_tokens=window(tok, re_conv, d))
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if not reply_exact and n - n_reply <= 5:
                print(f"   mismatch #{n - n_reply}: model {rec['model_tokens']} vs re-tokenized "
                      f"{rec['retok_tokens']}", flush=True)
        el = time.perf_counter() - t_start
        print(f"  {n:4d} replies ({toks_total} tokens, {toks_total / el:.0f} tok/s): reply exact "
              f"{100 * n_reply / n:.1f}%  conversation exact {100 * n_convo / n:.1f}%  "
              f"text lossless {100 * n_text / n:.1f}%", flush=True)
    return {"model": name, "revision": sha, "replies": n, "reply_tokens": toks_total,
            "reply_exact": n_reply, "convo_exact": n_convo, "text_lossless": n_text,
            "seconds": time.perf_counter() - t_start}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--models", default="qwen3-0.6b,smollm2-360m")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.n, args.max_new = 16, 64
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    name = args.name or ("exactness-smoke" if args.smoke else
                         f"exactness{'-greedy' if args.greedy else ''}")
    run = os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S-") + name)
    os.makedirs(run, exist_ok=False)
    cfg = {"args": vars(args), "git": git_sha(), "host": platform.node(), "python": sys.version,
           "torch": torch.__version__, "loadavg": os.getloadavg(),
           "prompts": "tatsu-lab/alpaca train, seeded sample"}
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"), indent=1)
    print(f"run dir: {run}", flush=True)
    ps = prompts(args.n, args.seed)
    res = {}
    with open(os.path.join(run, "replies.jsonl"), "w") as f:
        for key in args.models.split(","):
            torch.manual_seed(args.seed)
            res[key] = run_model(key, ps, args, f)
    json.dump(res, open(os.path.join(run, "results.json"), "w"), indent=1)
    print("\nsummary:", flush=True)
    for k, r in res.items():
        print(f"  {k}: {r['replies']} replies, reply exact {100 * r['reply_exact'] / r['replies']:.1f}%, "
              f"conversation exact {100 * r['convo_exact'] / r['replies']:.1f}%, text lossless "
              f"{100 * r['text_lossless'] / r['replies']:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
