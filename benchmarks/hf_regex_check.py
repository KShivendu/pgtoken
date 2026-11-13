"""Is Hugging Face slow here because of the lookaround regex? Same BPE model, the pretokenizer
regex swapped for GPT-2's lookahead-free core (`\\s+(?!\\S)` and `(?!\\p{L})` removed). Token
counts change slightly; this measures speed only. One thread, 4 KB docs, best of 3."""
import json, os, sys, time
import tokenizers
from tokenizers import Tokenizer
import isbpe_storage as S

text = S.utf8_prefix(open(f"{S.RESEARCH}/data/code_tr.txt", "rb").read(), 4_000_000)
docs = [c.decode() for c in S.chunks(text, 4096, 10**9)]
mb = len(text) / 1e6
cfg = json.load(open(f"{S.RESEARCH}/models/exp138/bpe_code_same_65536.json"))
orig = cfg["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
simple = r" ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"
print("tokenizers", tokenizers.__version__, flush=True)
for name, pat in (("original regex (lookaround)", orig), ("no lookaround", simple)):
    cfg["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = pat
    try:
        path = os.path.join(os.environ.get("SCRATCH", "."), "hf_regex_check_tok.json")
        open(path, "w").write(json.dumps(cfg))
        tok = Tokenizer.from_file(path)
    except Exception as e:
        print(f"  {name:28s} cannot load: {e}", flush=True); continue
    tok.encode_batch(docs[:20])
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for d in docs:
            tok.encode(d, add_special_tokens=False)
        best = min(best, time.perf_counter() - t0)
    print(f"  {name:28s} {mb / best:6.1f} MB/s one thread", flush=True)
