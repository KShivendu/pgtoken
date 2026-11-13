"""tokenizers 1.0 rc speed on the exp138 BPE trained with GPT-2's pretokenizer (the one rc2 can
load), against isbpe_rs, same 4 MB of code in 4 KB docs, one thread, best of 3."""
import time
import isbpe_rs, tokenizers
from tokenizers import Tokenizer
import isbpe_storage as S

text = S.utf8_prefix(open(f"{S.RESEARCH}/data/code_tr.txt", "rb").read(), 4_000_000)
docs = [c.decode() for c in S.chunks(text, 4096, 10**9)]
mb = len(text) / 1e6
# Same model, its pretokenizer written as ByteLevel(use_regex=True), which is GPT-2's regex:
# the form 1.0 rc can run. Token counts must match 0.22.1's run of the original file.
import json, os
cfg = json.load(open(f"{S.RESEARCH}/models/exp138/bpe_code_gpt2_65536.json"))
cfg["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                        "use_regex": True}
path = os.path.join(os.environ.get("SCRATCH", "."), "bpe_code_gpt2_bytelevel.json")
json.dump(cfg, open(path, "w"))
bpe = Tokenizer.from_file(path)
isb = isbpe_rs.ISBPE.load(f"{S.RESEARCH}/models/exp138/isbpe_code_dual_65536.json")
print("tokenizers", tokenizers.__version__, flush=True)
for name, fn in (("hf bpe (gpt2 pretok)", lambda d: bpe.encode(d, add_special_tokens=False).ids),
                 ("isbpe", isb.encode)):
    fn(docs[0]); best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        n = sum(len(fn(d)) for d in docs)
        best = min(best, time.perf_counter() - t0)
    print(f"  {name:22s} {mb / best:6.1f} MB/s one thread, {n:,} tokens", flush=True)
