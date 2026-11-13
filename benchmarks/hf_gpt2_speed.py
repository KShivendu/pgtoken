"""Hugging Face's own GPT-2 tokenizer (from the hub) against tiktoken's GPT-2 (r50k), same
4 MB of code in 4 KB docs, one thread, best of 3. Is HF slow in general or only on our model?"""
import time
import tiktoken, tokenizers
from tokenizers import Tokenizer
import isbpe_storage as S

text = S.utf8_prefix(open(f"{S.RESEARCH}/data/code_tr.txt", "rb").read(), 4_000_000)
docs = [c.decode() for c in S.chunks(text, 4096, 10**9)]
mb = len(text) / 1e6
hf = Tokenizer.from_pretrained("gpt2")
tk = tiktoken.get_encoding("r50k_base")
print("tokenizers", tokenizers.__version__, flush=True)
for name, fn in (("hf gpt2", lambda d: hf.encode(d).ids), ("tiktoken r50k", tk.encode_ordinary)):
    fn(docs[0]); best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        n = sum(len(fn(d)) for d in docs)
        best = min(best, time.perf_counter() - t0)
    print(f"  {name:14s} {mb / best:6.1f} MB/s one thread, {n:,} tokens", flush=True)
