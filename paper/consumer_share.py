#!/usr/bin/env python3
"""What share of a model consumer's work is tokenizing? Stored token IDs save exactly that share.

Tokenizer: Qwen3's (shared by Qwen3-Embedding, Qwen3-Reranker and the Qwen3 LLMs), HF
`tokenizers` fast. Rows: DBpedia abstracts (title + text, from prep_data.py) and C4 English
chunks of 512 o200k tokens (prep_data.py), re-tokenized with Qwen3. Every consumer is timed
warm, both ways, with the same batches:

    (a) text path     tokenize the text, then run the model
    (b) stored IDs    build the model input from IDs already stored, then run the model

Consumers:
    1. Qwen3-Embedding-0.6B: embed DBpedia docs in batches of 32, and embed 100 retrieved C4
       chunks (batches of 25).
    2. Qwen3-Reranker-0.6B: score 100 (query, abstract) pairs. The stored-ID input is
       prefix + tokenize(instruction and query part) + stored doc IDs + suffix, and is checked ID
       for ID against tokenizing the whole pair string as the model card does.
    3. A static embedder (model2vec distill of Qwen3-Embedding, futur/...-model2vec-onnx, if its
       tokenizer is Qwen3's; otherwise Qwen3-Embedding's own input embedding table): tokenize vs
       lookup + mean pool.
    4. Token budgeting: how many of 100 chunks fit in 8,192 tokens: tokenize-to-count vs stored
       lengths (pgtoken.token_count, which reads only the header).
Also: the same rows stored as Qwen3 IDs in a pgtoken vocabulary (151,669 ids, freq trained on
other rows) vs text lz4 and vs the o200k preset, bytes per row.

Usage (ML client image, see docker/Dockerfile.client-ml):
    python paper/consumer_share.py --dbpedia /data/dbpedia --c4 /data/c4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperlib as P  # noqa: E402

C = P.C
from rag_e2e import Data  # noqa: E402

EMB = ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
RERANK = ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473")
STATIC = ("futur/Qwen3-Embedding-0.6B-model2vec-onnx", "a3f60a85a2570e7976ea7f2d6505ad3e5152a96f")
QWEN_VOCAB = 151669


def timeit(fn, reps: int):
    out, ts = None, []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return out, float(np.median(ts))


def report(name: str, tok_ms: float, model_ms: float, n_tok: int, extra: str = "") -> dict:
    share = tok_ms / (tok_ms + model_ms)
    r = {"tokenize_ms": tok_ms, "model_ms": model_ms, "tokenize_share": share,
         "tokens": n_tok, "tokenize_tokens_per_s": n_tok / (tok_ms / 1e3),
         "model_tokens_per_s": n_tok / (model_ms / 1e3)}
    print(f"  {name:<44} tokenize {tok_ms:>9.2f} ms  model {model_ms:>10.2f} ms  "
          f"tokenize share {100 * share:>6.2f}%  ({n_tok} tokens; tokenize "
          f"{r['tokenize_tokens_per_s'] / 1e6:.1f} M tok/s, model "
          f"{r['model_tokens_per_s'] / 1e3:.1f} k tok/s) {extra}", flush=True)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbpedia", default="/data/dbpedia")
    ap.add_argument("--c4", default="/data/c4")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--embed-docs", type=int, default=320, help="DBpedia docs to embed (batch 32)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoModel, AutoModelForCausalLM

    threads = len(os.sched_getaffinity(0))
    torch.set_num_threads(threads)
    out_dir = P.run_dir("consumer-share", args.smoke)
    print(f"run dir: {out_dir}; torch {torch.__version__}, {threads} threads", flush=True)
    res = {"config": vars(args) | {"machine": P.machine(), "threads": threads,
                                   "models": {"embedding": EMB, "reranker": RERANK,
                                              "static": STATIC}}}

    tok = Tokenizer.from_file(hf_hub_download(EMB[0], "tokenizer.json", revision=EMB[1]))
    rr_tok = Tokenizer.from_file(hf_hub_download(RERANK[0], "tokenizer.json", revision=RERANK[1]))

    def enc(texts):
        return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]

    rng = np.random.default_rng(5)
    dbp = Data(args.dbpedia, len(np.load(os.path.join(args.dbpedia, "ids_off.npy"))) - 1, 0, 0, True)
    c4 = Data(args.c4, len(np.load(os.path.join(args.c4, "ids_off.npy"))) - 1, 0, 0, True)
    dbp_rows = rng.choice(dbp.rows, size=args.rows, replace=False)
    c4_rows = rng.choice(c4.rows, size=args.rows, replace=False)
    dbp_text = [dbp.text(int(i)) for i in dbp_rows]
    c4_text = [c4.text(int(i)) for i in c4_rows]
    # "Stored" Qwen3 ids: what a pgtoken column with a Qwen3 vocabulary would hand back.
    dbp_ids, c4_ids = enc(dbp_text), enc(c4_text)
    for name, ids in (("DBpedia", dbp_ids), ("C4", c4_ids)):
        n = np.array([len(x) for x in ids])
        print(f"  {name}: {len(ids)} rows, Qwen3 tokens p25/p50/p90 {np.percentile(n, 25):.0f}/"
              f"{np.median(n):.0f}/{np.percentile(n, 90):.0f}", flush=True)

    # ── 1. embedding ──
    print("\n1. Qwen3-Embedding-0.6B (last-token pooling, left padding)", flush=True)
    model = AutoModel.from_pretrained(EMB[0], revision=EMB[1], torch_dtype=torch.float32).eval()
    pad_id = tok.token_to_id("<|endoftext|>")
    full = [e.ids for e in tok.encode_batch(dbp_text[:200], add_special_tokens=True)]
    suffix = full[0][len(dbp_ids[0]):]
    eq = sum(f == s + suffix for f, s in zip(full, dbp_ids[:200]))
    print(f"  special tokens the tokenizer adds after the text: {suffix}; stored ids + that == "
          f"tokenizer output on {eq}/200 docs", flush=True)

    def batch_tensors(id_lists):
        id_lists = [x + suffix for x in id_lists]
        L = max(len(x) for x in id_lists)
        ids = torch.full((len(id_lists), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(id_lists), L), dtype=torch.long)
        for r, x in enumerate(id_lists):
            ids[r, L - len(x):] = torch.tensor(x)
            att[r, L - len(x):] = 1
        return ids, att

    @torch.inference_mode()
    def embed(ids, att):
        h = model(input_ids=ids, attention_mask=att).last_hidden_state[:, -1]
        return torch.nn.functional.normalize(h, dim=-1)

    def run_embed(texts, stored, bs, label):
        tok_ms = model_ms = from_ids_ms = 0.0
        n_tok = 0
        for lo in range(0, len(texts), bs):
            tb, sb = texts[lo:lo + bs], stored[lo:lo + bs]
            (ids_a, att_a), t_tok = timeit(lambda: batch_tensors(enc(tb)), args.reps)
            (ids_b, att_b), t_ids = timeit(lambda: batch_tensors(sb), args.reps)
            assert torch.equal(ids_a, ids_b)
            _, t_model = timeit(lambda: embed(ids_b, att_b), 1)
            tok_ms += t_tok
            from_ids_ms += t_ids
            model_ms += t_model
            n_tok += int(att_b.sum())
        r = report(label, tok_ms, model_ms, n_tok,
                   f"[stored-ID tensor build {from_ids_ms:.2f} ms]")
        r["from_ids_ms"] = from_ids_ms
        r["saving_share"] = (tok_ms - from_ids_ms) / (tok_ms + model_ms)
        return r

    embed(*batch_tensors(dbp_ids[:4]))  # warm
    res["embedding_docs"] = run_embed(dbp_text[:args.embed_docs], dbp_ids[:args.embed_docs], 32,
                                      f"embed {args.embed_docs} DBpedia docs, batch 32")
    res["embedding_100_chunks"] = run_embed(c4_text[:100], c4_ids[:100], 25,
                                            "embed 100 C4 chunks, batch 25")
    del model

    # ── 2. reranker ──
    print("\n2. Qwen3-Reranker-0.6B (yes/no logit, model-card template)", flush=True)
    rr = AutoModelForCausalLM.from_pretrained(RERANK[0], revision=RERANK[1],
                                              torch_dtype=torch.float32).eval()
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
              "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
              "<|im_end|>\n<|im_start|>user\n")
    suffix_rr = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    instr = "Given a web search query, retrieve relevant passages that answer the query"
    import json as _json

    qtitles = [_json.loads(x)["title"] for x in open(os.path.join(args.dbpedia, "queries.jsonl"))]
    pairs = [(qtitles[i % len(qtitles)], i) for i in range(100)]
    pre_ids = rr_tok.encode(prefix, add_special_tokens=False).ids
    suf_ids = rr_tok.encode(suffix_rr, add_special_tokens=False).ids
    yes_id, no_id = rr_tok.token_to_id("yes"), rr_tok.token_to_id("no")

    def pair_text(q, d):
        return f"<Instruct>: {instr}\n<Query>: {q}\n<Document>: {d}"

    def ids_text_path(batch):
        return [pre_ids + e.ids + suf_ids for e in rr_tok.encode_batch(
            [pair_text(q, dbp_text[d]) for q, d in batch], add_special_tokens=False)]

    def ids_stored_path(batch):
        heads = rr_tok.encode_batch([f"<Instruct>: {instr}\n<Query>: {q}\n<Document>: "
                                     for q, _ in batch], add_special_tokens=False)
        return [pre_ids + h.ids + dbp_ids[d] + suf_ids for h, (_, d) in zip(heads, batch)]

    a_ids, b_ids = ids_text_path(pairs), ids_stored_path(pairs)
    mism = [i for i in range(100) if a_ids[i] != b_ids[i]]
    ex = None
    if mism:
        i = mism[0]
        j = next(k for k in range(min(len(a_ids[i]), len(b_ids[i]))) if a_ids[i][k] != b_ids[i][k])
        ex = {"pair": i, "pos": j, "pair_tokenizer": [rr_tok.decode([t]) for t in a_ids[i][j - 1:j + 3]],
              "stored_concat": [rr_tok.decode([t]) for t in b_ids[i][j - 1:j + 3]]}
    print(f"  ID check: stored-ID input == tokenizing the whole pair on {100 - len(mism)}/100 pairs"
          + (f"; first difference {ex}" if ex else ""), flush=True)

    def to_tensors(lists):
        L = max(len(x) for x in lists)
        ids = torch.full((len(lists), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(lists), L), dtype=torch.long)
        for r, x in enumerate(lists):
            ids[r, L - len(x):] = torch.tensor(x)
            att[r, L - len(x):] = 1
        return ids, att

    @torch.inference_mode()
    def score(ids, att):
        lg = rr(input_ids=ids, attention_mask=att).logits[:, -1, :]
        return torch.log_softmax(torch.stack([lg[:, no_id], lg[:, yes_id]], 1), 1)[:, 1].exp()

    score(*to_tensors(b_ids[:2]))
    tok_ms = ids_ms = model_ms = 0.0
    n_tok = 0
    sc_a, sc_b = [], []
    for lo in range(0, 100, 25):
        bt = pairs[lo:lo + 25]
        ta, t_a = timeit(lambda: to_tensors(ids_text_path(bt)), args.reps)
        tb_, t_b = timeit(lambda: to_tensors(ids_stored_path(bt)), args.reps)
        s_b, t_m = timeit(lambda: score(*tb_), 1)
        sc_b.append(s_b)
        if not torch.equal(ta[0], tb_[0]):
            sc_a.append(score(*ta))
        else:
            sc_a.append(s_b)
        tok_ms, ids_ms, model_ms = tok_ms + t_a, ids_ms + t_b, model_ms + t_m
        n_tok += int(tb_[1].sum())
    diff = float((torch.cat(sc_a) - torch.cat(sc_b)).abs().max())
    r = report("score 100 (query, abstract) pairs, batch 25", tok_ms, model_ms, n_tok,
               f"[stored-ID path {ids_ms:.2f} ms (still tokenizes the query part)]")
    r.update({"from_ids_ms": ids_ms, "id_mismatched_pairs": len(mism), "mismatch_example": ex,
              "max_score_diff": diff, "saving_share": (tok_ms - ids_ms) / (tok_ms + model_ms)})
    print(f"  max |score(text path) - score(stored ids)| = {diff:.2e}", flush=True)
    res["reranker"] = r
    del rr

    # ── 3. static embedder ──
    print("\n3. static embedder (lookup + mean pool)", flush=True)
    from safetensors.numpy import load_file

    st_tok = Tokenizer.from_file(hf_hub_download(STATIC[0], "tokenizer.json", revision=STATIC[1]))
    same = [st_tok.encode(t, add_special_tokens=False).ids for t in dbp_text[:200]] == dbp_ids[:200]
    if same:
        W = next(iter(load_file(hf_hub_download(STATIC[0], "model.safetensors",
                                                revision=STATIC[1])).values()))
        st_name = f"{STATIC[0]} (model2vec, {W.shape[0]} x {W.shape[1]})"
    else:
        m = AutoModel.from_pretrained(EMB[0], revision=EMB[1], torch_dtype=torch.float32)
        W = m.get_input_embeddings().weight.detach().numpy()
        st_name = (f"Qwen3-Embedding-0.6B input embedding table ({W.shape[0]} x {W.shape[1]}); "
                   f"{STATIC[0]}'s tokenizer differs from Qwen3's")
        del m
    print(f"  model: {st_name}; tokenizer identical to Qwen3 on 200 docs: {same}", flush=True)

    def pool(lists):
        return np.stack([W[np.asarray(x)].mean(0) for x in lists])

    for label, texts, ids in (("DBpedia", dbp_text, dbp_ids), ("C4", c4_text, c4_ids)):
        _, t_tok = timeit(lambda: enc(texts), args.reps)
        _, t_pool = timeit(lambda: pool(ids), args.reps)
        res[f"static_{label}"] = report(f"static embed {len(texts)} {label} rows", t_tok, t_pool,
                                        sum(len(x) for x in ids))
        res[f"static_{label}"]["model"] = st_name

    # ── 4 and storage: Qwen3 ids in pgtoken ──
    print("\n4. token budgeting and Qwen3 ids in pgtoken", flush=True)
    import pgtoken_rs as R

    with C.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgtoken")
        name = "paper_qwen3"
        row = conn.execute("SELECT id FROM pgtoken.vocabulary WHERE name = %s", (name,)).fetchone()
        vid = row[0] if row else conn.execute(
            "SELECT pgtoken.create_vocabulary(%s, %s, compression => 'freq')",
            (name, QWEN_VOCAB)).fetchone()[0]
        if conn.execute("SELECT ranked FROM pgtoken.vocabulary_info(%s)", (name,)).fetchone()[0] is None:
            # train on rows NOT measured below: 20k other DBpedia rows + 4k other C4 rows
            held = set(dbp_rows.tolist())
            tr_d = [i for i in range(0, dbp.rows, max(1, dbp.rows // 25000)) if i not in held][:20000]
            heldc = set(c4_rows.tolist())
            tr_c = [i for i in range(0, c4.rows, max(1, c4.rows // 5000)) if i not in heldc][:4000]
            train_ids = enc([dbp.text(i) for i in tr_d]) + enc([c4.text(i) for i in tr_c])
            conn.execute("DROP TABLE IF EXISTS paper_qwen3_train")
            conn.execute("CREATE TABLE paper_qwen3_train (ids int[])")
            with conn.cursor() as cur:
                with cur.copy("COPY paper_qwen3_train (ids) FROM STDIN") as cp:
                    for x in train_ids:
                        cp.write_row(("{" + ",".join(map(str, x)) + "}",))
            conn.execute("SELECT pgtoken.train(%s, 'SELECT ids FROM paper_qwen3_train')", (name,))
        ranking = bytes(conn.execute("SELECT ranking FROM pgtoken.vocabulary_codec(%s)",
                                     (name,)).fetchone()[0])
        tbl = R.Table.from_bytes(ranking)
        sizes = {}
        for label, texts, ids in (("dbpedia", dbp_text, dbp_ids), ("c4", c4_text, c4_ids)):
            t = f"paper_qwen3_{label}"
            conn.execute(f"DROP TABLE IF EXISTS {t}")
            conn.execute(f"CREATE TABLE {t} (id int PRIMARY KEY, txt text COMPRESSION lz4, "
                         f"txt_forced text COMPRESSION lz4, qwen pgtoken.{name}, "
                         f"o200k pgtoken.o200k)")
            conn.execute(f"ALTER TABLE {t} ALTER COLUMN txt_forced SET STORAGE MAIN, "
                         "SET (toast_tuple_target = 128)")
            o200k = C._hf_file(C.hf_tokenizer_json(), "hf-o200k", n_vocab=200019).hf
            o_ids = [e.ids for e in o200k.encode_batch(texts, add_special_tokens=False)]
            # Text and o200k ids via COPY (the server encodes the o200k int[] with its preset).
            with conn.cursor() as cur:
                with cur.copy(f"COPY {t} (id, txt, txt_forced, o200k) FROM STDIN") as cp:
                    for i, (tx, o) in enumerate(zip(texts, o_ids)):
                        cp.write_row((i, tx, tx, "{" + ",".join(map(str, o)) + "}"))
            # Qwen3 values encoded on the client (pgtoken_rs freq with the trained table), sent
            # in binary as pgtoken.tokens, as bench_readwrite does.
            from psycopg.adapt import Dumper
            from psycopg.pq import Format

            class Tok(bytes):
                pass

            class TokDumper(Dumper):
                format = Format.BINARY
                oid = conn.execute("SELECT 'pgtoken.tokens'::regtype::oid").fetchone()[0]

                def dump(self, obj):
                    return obj

            conn.adapters.register_dumper(Tok, TokDumper)
            with conn.cursor() as cur:
                cur.executemany(f"UPDATE {t} SET qwen = %s WHERE id = %s",
                                [(Tok(R.encode(np.asarray(q, np.uint32), "freq", vid, tbl)), i)
                                 for i, q in enumerate(ids)])
            ok = conn.execute(f"SELECT count(*) FROM {t} WHERE qwen::int[] IS NOT NULL").fetchone()[0]
            s_ = conn.execute(f"SELECT avg(octet_length(txt)), avg(pg_column_size(txt)), "
                              f"avg(pg_column_size(txt_forced)), avg(pg_column_size(qwen)), "
                              f"avg(pg_column_size(o200k)) FROM {t}").fetchone()
            sizes[label] = {"utf8": float(s_[0]), "text_lz4": float(s_[1]),
                            "text_lz4_forced": float(s_[2]), "qwen3_freq": float(s_[3]),
                            "o200k_freq": float(s_[4]), "rows": ok}
            print(f"  {label}: bytes/row UTF-8 {s_[0]:.0f}, text lz4 {s_[1]:.0f}, text lz4 forced "
                  f"{s_[2]:.0f}, pgtoken Qwen3 freq {s_[3]:.0f}, pgtoken o200k freq {s_[4]:.0f} "
                  f"({ok} rows)", flush=True)
        res["storage_bytes_per_row"] = sizes

        # token budgeting: 100 C4 chunks, how many fit in 8192 Qwen3 tokens
        picks = list(range(100))

        def count_tok():
            n = np.cumsum([len(e.ids) for e in tok.encode_batch([c4_text[i] for i in picks],
                                                                add_special_tokens=False)])
            return int((n <= 8192).sum())

        def count_db():
            n = np.cumsum([r[0] for r in conn.execute(
                "SELECT pgtoken.token_count(qwen) FROM paper_qwen3_c4 WHERE id = ANY(%s) ORDER BY id",
                (picks,))])
            return int((n <= 8192).sum())

        def count_text_fetch():
            txt = [r[0] for r in conn.execute(
                "SELECT txt FROM paper_qwen3_c4 WHERE id = ANY(%s) ORDER BY id", (picks,))]
            n = np.cumsum([len(e.ids) for e in tok.encode_batch(txt, add_special_tokens=False)])
            return int((n <= 8192).sum())

        a, t_a = timeit(count_tok, 20)
        b, t_b = timeit(count_db, 20)
        c, t_c = timeit(count_text_fetch, 20)
        print(f"  fit 100 C4 chunks into 8192 Qwen3 tokens: tokenize text in hand {t_a:.2f} ms "
              f"({a} fit); fetch text + tokenize {t_c:.2f} ms ({c}); SELECT pgtoken.token_count "
              f"{t_b:.2f} ms ({b})", flush=True)
        res["budgeting"] = {"tokenize_in_hand_ms": t_a, "fetch_text_tokenize_ms": t_c,
                            "token_count_sql_ms": t_b, "fit": [a, b, c]}
    P.write_json(os.path.join(out_dir, "result.json"), res)
    print(f"wrote {out_dir}/result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
