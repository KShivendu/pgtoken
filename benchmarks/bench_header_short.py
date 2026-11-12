import struct, sys, statistics as st, json
import numpy as np, tiktoken
import pgtoken_rs

# Size model for candidate value-header layouts on short texts, using the bundled o200k ranking.
# "cur12+svb" is the old pre-v1 format (12-byte header, streamvbyte ranks), kept for comparison.
# "v1 committed" is the format the extension writes today (core/src/header.rs): tag, flags,
# u16 vocabulary id, LEB128 token count, then one LEB128 rank per token. The check below
# asserts the real Rust codec matches the v1 model byte for byte.
# Usage: uv run python benchmarks/bench_header_short.py
import os, zstandard
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = zstandard.ZstdDecompressor().decompress(open(os.path.join(_ROOT, "ext/presets/o200k.tntt.zst"), "rb").read())
K = struct.unpack_from("<I", raw, 8)[0]
tor = np.frombuffer(raw, "<u4", offset=12, count=K)
rank_of = {int(t): r for r, t in enumerate(tor)}
table = pgtoken_rs.Table.from_bytes(raw)
enc = tiktoken.get_encoding("o200k_base")

def rank(t): return rank_of.get(t, t + K)

def svb(rs):
    if not rs: return 0
    return (len(rs) + 3) // 4 + sum(1 if r < 256 else 2 if r < 65536 else 3 if r < 1 << 24 else 4 for r in rs)

def leb(v):
    n = 1
    while v >= 128: v >>= 7; n += 1
    return n

def pg(n):  # varlena: 1-byte header if total <= 127, else 4
    return n + 1 if n + 1 <= 127 else n + 4

ex = ["Hello, world!", '{"role": "user"}', "नमस्ते 🦜",
      "Can you summarise my last three invoices and flag anything that looks unusual compared to previous months?"]

chat = """hi
thanks!
ok
yes please
Can you help me write an email to my landlord about the broken heater?
What's the capital of Australia?
Summarize this in two sentences.
Make it shorter and more formal.
Translate to French: where is the train station?
Why does my Python script say ModuleNotFoundError: No module named 'numpy'?
That didn't work, same error.
Explain recursion like I'm five.
Give me 5 name ideas for a coffee shop near the beach.
Sure! Here are a few options you might like.
I'm sorry, I can't help with that request.
Great question! The short answer is yes, but there are some caveats.
Can you rewrite the second paragraph so it sounds less robotic?
what time zone is IST
convert 72F to celsius
How do I center a div?
Remind me tomorrow at 9am to call the dentist.
Order status for #48213?
My package hasn't arrived yet, tracking says delivered.
Please cancel my subscription.
Is it safe to take ibuprofen with coffee?
Write a haiku about Postgres.
Done. Anything else?
Could you list the pros and cons of renting versus buying in Berlin?
lol
Let me check that for you.
The meeting has been moved to Thursday at 3pm.
I updated the README and fixed the failing test in test_api.py.
SELECT * FROM users WHERE email = 'a@b.com';
git push -f origin main
Your refund of $42.99 has been processed.
Sounds good, go ahead.
What are the side effects of metformin?
Generate a SQL query to find duplicate rows by email.
How many tokens is this sentence?
Please respond only in JSON.""".split("\n")

titles = """Quarterly Business Review Q3 2026
How to Train Your Dragon
Fix flaky test in auth middleware
Re: Invoice #10234 overdue
Untitled document
Meeting notes 2026-11-12
Best pizza in Naples
Introduction to Vector Databases
Bump serde from 1.0.210 to 1.0.214
Weekly sync
Trip to Lisbon
Python asyncio tutorial for beginners
Annual Report 2025
iPhone 16 Pro Max review: worth the upgrade?
Customer churn analysis
README.md
New chat
Debugging memory leak in Rust service
Recipe: vegan banana bread
Onboarding checklist for new engineers""".split("\n")

jsons = ['{"role": "assistant"}', '{"role": "system", "content": "You are a helpful assistant."}',
         '{"ok": true}', '{"id": 12345, "status": "active"}', '{"error": "not_found"}', '[]', '{}',
         '{"name": "get_weather", "arguments": {"city": "Paris"}}', '{"lat": 52.52, "lon": 13.405}',
         '{"type": "text", "text": "Hello"}', '{"finish_reason": "stop"}', '{"tags": ["urgent", "billing"]}',
         '{"user_id": "u_8f3a2c", "plan": "pro", "seats": 12}', '{"temperature": 0.7, "max_tokens": 256}',
         '{"tool_call_id": "call_abc123", "content": "72F and sunny"}', '{"score": 0.93}',
         '{"model": "gpt-4o", "usage": {"prompt_tokens": 12, "completion_tokens": 40}}']

multi = ["Bonjour, comment ça va ?", "¿Dónde está la biblioteca?", "Danke schön!", "こんにちは、元気ですか？",
         "你好，世界", "Привет, как дела?", "مرحبا بالعالم", "안녕하세요", "मुझे मदद चाहिए", "Olá, tudo bem?",
         "🔥🔥🔥", "Γειά σου Κόσμε", "Merhaba dünya", "Xin chào", "שלום עולם"]

misc = ["ERROR 2026-11-12T10:15:03Z connection refused (port 5432)", "user@example.com", "+1 (415) 555-0199",
        "INFO request_id=7f3a latency_ms=42 status=200", "https://example.com/docs/getting-started?ref=nav",
        "urgent", "billing", "positive", "en-US", "a3f9c2e1-7b4d-4e8a-9c1f-2d3e4f5a6b7c",
        "TODO: handle the empty case", "pip install numpy==2.1.0", "Approved", "N/A", "SKU-88213-XL"]

cats = {"examples": ex, "chat": chat, "titles": titles, "json": jsons, "multilingual": multi, "misc/logs/tags": misc}

def layouts(ids):
    rs = [rank(int(t)) for t in ids]
    n = len(rs); p_svb = svb(rs); p_leb = sum(leb(r) for r in rs)
    return {
        "text": None,
        "v1 committed": 1 + 1 + 2 + leb(n) + p_leb,  # tag, flags, u16 vocab, LEB128 n, LEB128 ranks
        "cur12+svb": 12 + p_svb,  # old pre-v1 format
        "8B+svb": 8 + p_svb,        # magic/ver/codec, u16 vocab, u32 n... (compacted fixed)
        "tag+v2+n16+svb": 5 + p_svb,
        "tag+v2+vn+svb": 1 + 2 + leb(n) + p_svb,
        "tag+v2+leb": 1 + 2 + p_leb,
        "tag+v1+leb": 1 + 1 + p_leb,
    }

# check the v1 model matches the real codec (the old 12 + streamvbyte format is no longer written)
for t in ex + chat + jsons + multi:
    ids = enc.encode(t)
    rs = [rank(i) for i in ids]
    real = pgtoken_rs.encode(np.array(ids, dtype=np.uint32), "freq", 65280, table)
    assert len(real) == 4 + leb(len(ids)) + sum(leb(r) for r in rs), (t, len(real))

allrows = []
for c, texts in cats.items():
    rows = []
    for t in texts:
        ids = enc.encode(t); L = layouts(ids)
        L["text"] = len(t.encode())
        r = {k: pg(v) for k, v in L.items()}; r["n"] = len(ids); r["t"] = t
        rows.append(r)
    allrows += rows if c != "examples" else []
    keys = [k for k in rows[0] if k not in ("n", "t")]
    print(f"\n== {c}: {len(rows)} texts, tokens p50={st.median([r['n'] for r in rows])} max={max(r['n'] for r in rows)}")
    print("mean PG bytes: " + "  ".join(f"{k}={st.mean([r[k] for r in rows]):.1f}" for k in keys))
    if c == "examples":
        for r in rows: print(f"  n={r['n']:3d} " + " ".join(f"{k}={r[k]}" for k in keys) + f"  {r['t'][:40]!r}")

ns = [r["n"] for r in allrows]
print("\nall non-example:", len(ns), "texts; token count p25/p50/p90/max", np.percentile(ns, [25, 50, 90]), max(ns))
keys = [k for k in allrows[0] if k not in ("n", "t")]
for k in keys:
    worse = [r for r in allrows if r[k] > r["text"]]
    print(f"{k:16s} mean={st.mean(r[k] for r in allrows):5.1f}  larger-than-text={len(worse):3d}/{len(allrows)}  e.g. {[ (r['t'][:25], r['text'], r[k]) for r in worse[:2]]}")
print()
for r in allrows:
    if r["tag+v1+leb"] > r["text"]: print(f"LOSS n={r['n']} text={r['text']} new={r['tag+v1+leb']} svbnew={r['tag+v2+vn+svb']-1} cur={r['cur12+svb']} {r['t']!r}")
