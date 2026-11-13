# pgtoken

[![CI](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml/badge.svg)](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml)

Store text in PostgreSQL as LLM token IDs instead of UTF-8.

An LLM reads and writes text as token IDs. A tokenizer, such as OpenAI's `tiktoken`, turns
`'Hello, world!'` into `[13225, 11, 2375, 0]` and back. If your app feeds stored text to a model,
for example chat history for an agent or chunks for RAG, it tokenizes the same text on every
read. `pgtoken` stores the token IDs instead, compressed:

- **Smaller.** A 183-token English paragraph takes 870 B as `text` and 327 B as `tokens`, 2.66x
  smaller. That holds on disk, in WAL and in the buffer cache.
- **No tokenizing on read.** The IDs come straight back. Reading 100 chunks of 512 tokens is 2.8x
  faster than reading `text` and tokenizing it.
- **Text when you want it.** `body::text` gives the characters back, and you can insert plain text
  and let the server tokenize it.

**When not to use it.** Values of a few tokens, like `'hi'`, are bigger as tokens (8 B against
3 B). The column can't be sorted or carry a `UNIQUE` constraint. And you need to install a
compiled extension on the server, which hosted services such as RDS or Supabase don't allow.

<sub>Background: [blog](https://www.kshivendu.dev/blog/token-storage) ·
[paper](https://arxiv.org/abs/2608.02376)</sub>

> Early days. Version 0.1.0, PostgreSQL 14 to 18, Linux. The API may still change.

## Install

You need:

- Rust: [rustup.rs](https://rustup.rs)
- PostgreSQL with its development files, so that `pg_config` exists. On Ubuntu that is
  `postgresql-server-dev-16` for PostgreSQL 16.
- [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx), the tool that builds Rust extensions.

```sh
cargo install cargo-pgrx --version 0.19.2 --locked
cargo pgrx init --pg16 $(which pg_config)     # use your version: --pg14 ... --pg18

git clone https://github.com/KShivendu/pgtoken.git
cd pgtoken/ext && cargo pgrx install --release --sudo
```

`--sudo` is needed when PostgreSQL's folders belong to root, as with the Ubuntu packages. To try it
without root, `bash setup_pg.sh` installs PostgreSQL 14 under `~/.local/share`, then
`bash setup_pg.sh --start` starts it and `bash setup_pg.sh --psql` opens a shell.

`CREATE EXTENSION pgtoken` needs a superuser, because the extension is compiled C code. There is
nothing to configure. `DROP EXTENSION pgtoken CASCADE` removes it, along with every
column of a `tokens` type.

## Quick start

```sql
CREATE EXTENSION pgtoken;
SELECT pgtoken.create_vocabulary('o200k');   -- OpenAI's o200k_base tokenizer, bundled

CREATE TABLE documents (id bigserial PRIMARY KEY, body tokens.o200k);

INSERT INTO documents (body) VALUES ('{13225,11,2375,0}');   -- token IDs from tiktoken
INSERT INTO documents (body) VALUES ('Hello, world!');       -- or text: the server tokenizes it
SELECT body FROM documents;                                  -- {13225,11,2375,0}, twice
SELECT body::text FROM documents;                            -- Hello, world!, twice
SELECT * FROM documents WHERE body = 'Hello, world!';        -- both rows
```

A **vocabulary** is one tokenizer's set of IDs. `create_vocabulary('o200k')` creates the column
type `tokens.o200k`. Each vocabulary gets its own type, so PostgreSQL won't let you mix IDs from
two tokenizers. o200k is a **preset**, which means its frequency table, its ID-to-bytes table and its
tokenizer are built into the extension. The server tokenizer matches `tiktoken`'s `o200k_base` ID for ID.

Input is read as token IDs when it looks like `'{1,2,3}'`: braces around numbers and commas.
Anything else is text, so `'{"role": "user"}'` is text. Special tokens are not recognised in text:
`'<|endoftext|>'` is stored as the tokens of those 13 characters. `pgtoken.tokenize('Hello,
world!', 'o200k')` returns the IDs without storing anything.

`SELECT body` and `pg_dump` show token IDs, so restoring a dump never tokenizes anything.

## Other tokenizers

Any tokenizer works if you send its IDs. Tell pgtoken how many IDs it has:

```sql
SELECT pgtoken.create_vocabulary('gpt2', vocab_size => 50257);
```

This stores each ID in 2 bytes. Only the o200k preset accepts text. Any other vocabulary refuses
it with `vocabulary "gpt2" has no server tokenizer; send token ids`.

**To get text back**, load the tokenizer's ID-to-bytes table once. `scripts/load_mapping.py` does
this for any tiktoken encoding:

```sh
uv run python scripts/load_mapping.py --encoding cl100k_base --vocabulary cl100k
```

**To store less**, use `freq` compression and train it on a sample of your data. It gives common
tokens 1 byte (more in [Compression](#compression)):

```sql
SELECT pgtoken.create_vocabulary('corpus', vocab_size => 200019, compression => 'freq');
SELECT pgtoken.train('corpus', 'SELECT ids FROM my_corpus');   -- ids is an int[] column
```

The mapping and the training are write-once. A vocabulary can't change after that, so to change
one, create a new vocabulary and `ALTER TABLE ... TYPE tokens.<new>`.

## Coming from a text column

These work the same as on `text`:

```sql
SELECT DISTINCT body FROM chunks;
SELECT body, count(*) FROM chunks GROUP BY body;
SELECT * FROM a JOIN b ON a.body = b.body;
SELECT * FROM chunks WHERE body = '{24912,2375}';                    -- 'hello world'
SELECT * FROM chunks WHERE body::text LIKE '%invoice%';
CREATE INDEX ON chunks USING gin (to_tsvector('english', body::text));
```

`=` compares token IDs. So a tokenizer that lowercases treats `'Hello'` and `'hello'` as equal.

What doesn't work:

- **`ORDER BY body` and `UNIQUE`.** Both need the values to have an order, and compressed token IDs
  have none. Sort on `body::text` instead.
- **An `IN` list of nine or more literals** needs a cast on each value:
  `body IN ('{1}'::tokens.o200k, ...)`. PostgreSQL hashes a long list before it knows the
  column's type. Shorter lists and `body = '...'` need no cast.

## Use from Python

`pgtoken_rs` (in `py/`) reads and writes the column as numpy arrays of token IDs. It runs the
extension's own Rust code on your machine, so the server sends the stored bytes as they are. It
needs Rust to build:

```sh
uv pip install "./py[psycopg]"      # or: pip install "./py[psycopg]"
```

```python
import psycopg
from pgtoken_rs.psycopg import register

conn = psycopg.connect("dbname=mydb", autocommit=True)
vocab = register(conn, "o200k")   # fetches the vocabulary's settings from the server

conn.execute("INSERT INTO documents (body) VALUES (%s)", [vocab.encode([24912, 2375])])
ids = conn.execute("SELECT body FROM documents", binary=True).fetchone()[0]
print(ids)                        # [24912  2375], a numpy uint32 array
```

Binary `COPY` works too. See [py/README.md](py/README.md) for `COPY`, several vocabularies and
errors.

## Size and speed

**Size by value length.** Each row is the average `pg_column_size` of 300 values with the o200k
preset, on PostgreSQL 14. The script is `benchmarks/bench_length_buckets.py`. PostgreSQL only
compresses `text` once a row passes about 2 KB, using pglz or lz4. `tokens` are compressed at every
length:

| content | tokens | `text` | `text` pglz | `text` lz4 | `tokens.o200k` | smaller by |
| --- | --: | --: | --: | --: | --: | --: |
| English | 15 | 74 B | 74 B | 74 B | 33 B | 2.23x |
| English | 183 | 870 B | 870 B | 870 B | 327 B | 2.66x |
| English | 488 | 2314 B | 1830 B | 1815 B | 847 B | 2.14x |
| English | 1952 | 9249 B | 5538 B | 6344 B | 3306 B | 1.67x |
| code | 68 | 281 B | 281 B | 281 B | 114 B | 2.45x |
| code | 278 | 1130 B | 1130 B | 1130 B | 393 B | 2.88x |
| code | 1102 | 4508 B | 1723 B | 2039 B | 1325 B | 1.30x |
| Hindi | 30 | 220 B | 220 B | 220 B | 56 B | 3.92x |
| Hindi | 447 | 3446 B | 1213 B | 1816 B | 718 B | 1.69x |

"Smaller by" compares `tokens.o200k` with the smallest `text` column. Four values from this run:

- **183 tokens of English**, starting `old copy of Mr. Popper’s Penguins and we are having a lot
  of fun`. `text` stores 859 B and `tokens.o200k` 330 B, 2.60x smaller.
- **1466 tokens of one repeated code line**, `CThostFtdcLimitPosiSField["OpenVolume"] = "int"`.
  pglz stores 871 B, lz4 1149 B and `tokens.o200k` 888 B. The repeats are found by lz4 run inside
  the value (see [Compression](#compression)). Without it this value took 3121 B.
- **1642 tokens of math code**, `assert modgcd_multivariate(f, g) == (x + y, 2*x - 3, 4*y - 2)`.
  pglz stores 969 B and `tokens.o200k` 1240 B. Short symbols like `x`, `*` and `2` cost about a
  token each. pglz still wins on 10 of the 300 values in the 1102-token code row.
- **14 tokens**, `4): p. 332.\nGobburu, J.V.,`. `text` stores 27 B and `tokens.o200k` 32 B.
  Digits and punctuation cost about one token per two characters, and each value has a 5-byte
  header.

**A whole table.** 20,000 chunks of C4 English, 512 tokens each, GPT-2 IDs (PostgreSQL 14):

| column | per row | table | WAL per row | rows per page | rows moved to TOAST |
| --- | --: | --: | --: | --: | --: |
| `text` pglz | 1893 B | 45.9 MB | 2091 B | 6.32 | 7000 |
| `text` lz4 | 1832 B | 42.6 MB | 1953 B | 4.32 | 1500 |
| `tokens` freq | **836 B** | **18.7 MB** | **927 B** | **9.00** | **0** |

TOAST is PostgreSQL's side table for large values. Each value moved there costs an extra lookup on
every read. lz4 moves fewer rows out than pglz, so more large rows stay on the main page and fewer
fit per page. No `tokens` value is moved out. A 4 KB embedding column in the same table brings the
gap down to about 1.2x, since the embedding goes to TOAST in every case.

**Read and write speed.** 2000 chunks of 512 tokens, median of 300 repeats, end to end as a Python
client sees it, with GPT-2 IDs. A `text` read includes tokenizing the text with the Hugging Face
GPT-2 tokenizer. A `tokens` read decodes with `pgtoken_rs`:

| column | read 1 row | read 100 rows | write 1 row | write 100 rows, per row |
| --- | --: | --: | --: | --: |
| `text` lz4 | 349 µs | 3369 µs | 319 µs | 56.8 µs |
| `text` pglz | 362 µs | 3804 µs | 386 µs | 104.7 µs |
| `tokens` raw | 307 µs | 1307 µs | **267 µs** | **12.6 µs** |
| `tokens` freq | **304 µs** | **1199 µs** | 271 µs | 21.4 µs |

For 100 rows, the lz4 `text` column spends 1.10 ms fetching 233 KB and 2.25 ms tokenizing it.
`tokens` freq spends 0.71 ms fetching 83 KB and 0.49 ms decoding it. A single row gains less,
because the round trip to the server dominates. Reading `body::int[]` instead, with no client
library, takes 7.0 ms for 100 rows, because psycopg builds 512 Python ints per row.

Scripts for every number are in `benchmarks/`.

## Compression

| method | size | decode a 512-token value | needs training |
| --- | --: | --: | --- |
| `raw` (default without a preset) | 1-3 B per token | 0.1-0.4 µs | no |
| `freq` (default for a preset) | ~1.85 B per token | 1.6 µs | yes, or bundled in a preset |

`raw` stores each ID in a fixed width of 1, 2 or 3 bytes, set by `vocab_size`. `freq`
replaces each ID with its rank in a frequency table, so the most common token is 0. Ranks under
128 take 1 byte, under 16,384 take 2 bytes. When it makes a value smaller, `freq` also runs lz4
over these bytes to catch repeated lines, like the `CThostFtdc...` value above.

The o200k preset's table was counted over equal parts English, code and Hindi. Train your own when
your data looks different. Tokens the table never saw still work, just a byte or two wider.

## Limitations

- **Text needs an ID-to-bytes table.** Presets have one. Without one, reads give token IDs, and
  `psql` shows numbers.
- **No sorting and no `UNIQUE`** on the column. See [Coming from a text column](#coming-from-a-text-column).
- **A column must name a vocabulary.** A column typed as bare `pgtoken.tokens` accepts inserts but
  fails on read. Use `tokens.<name>`.
- **Binary writes are trusted.** Text and `int[]` input check every ID. Binary `COPY` and the
  `bytea` cast check only the header, to keep writes fast.
- **One value holds at most 67,108,864 tokens** (2^26). That is far past any model's context
  window, and it keeps a small corrupt value from claiming a huge size.
- **A vocabulary's name and number are reserved forever**, even after you drop it.
- **The frequency table is tuned to its data.** Sizes grow as your data drifts from what it was
  trained on.

## Reference

All functions live in the `pgtoken` schema.

| function | returns | |
| --- | --- | --- |
| `create_vocabulary(name [, vocab_size] [, compression] [, id] [, preset])` | `int` | also creates `tokens.<name>` |
| `train(name, query [, max_ranks])` | `text` | frequency table for `freq`, write-once |
| `load_mapping(name, query)` | `text` | ID-to-bytes table, write-once |
| `text(tokens)` | `text` | same as `body::text`, `IMMUTABLE` so it can back an index |
| `tokenize(text, vocabulary)` | `int[]` | server tokenizer, o200k preset only |
| `vocabulary_info(name)` | record | size, compression, width, preset |
| `vocabulary_codec(name)` | record | what a client library needs to encode |
| `drop_vocabulary(name)` | | drops the type, the number stays reserved |
| `token_count(tokens)` | `int` | reads only the header |
| `describe(tokens)` | record | codec, vocabulary, sizes |

Casts: `int[] → tokens` and `text → tokens` happen on assignment. `tokens → int[]` and
`tokens → text` need an explicit `::`. Write `body::text` rather than `text(body)`, which only
works with `pgtoken` on `search_path`. `pgtoken.tokens_send(body)` gives the raw stored bytes.
Use `create_vocabulary('chat', preset => 'o200k')` for a preset under another name, and
`compression => 'raw'` to skip compression.

<details>
<summary>How values and tables are stored</summary>

Every value is a short header and then the payload:

```
off  size  field
  0     1  tag = (version 1 << 4) | codec    codec 0=raw16 1=raw24 2=freq 3=raw8 4=freq_lz4
  1     1  flags, all zero for now
  2     2  vocabulary id (u16 little-endian)
  4   1-5  token count (LEB128)
  *     -  payload: raw = count x width bytes, freq = one LEB128 rank per token,
                    freq_lz4 = the freq payload as one lz4 block
```

The header is 5 bytes under 128 tokens and 6 bytes up to 16,383. `'Hello, world!'` is 13 bytes as
`tokens.o200k` and 14 as `text`. A reader refuses an unknown tag or flag. lz4 can write the same
data in more than one way, so `=`, hashing, `GROUP BY` and `DISTINCT` decompress a `freq_lz4`
value before comparing. Equal token IDs always compare equal. Details are in `core/src/header.rs`.

A preset's tables are compiled into the extension, and its vocabularies take numbers from a reserved
range (65280-65535 for o200k), so a value reads back with nothing else in the database. `train` and
`load_mapping` store their tables as SQL functions, `pgtoken.artefact_<id>_ranking()` and
`pgtoken.artefact_<id>_mapping()`. So they are transactional, copied to replicas and included in
`pg_dump`, and a dump restores them before any table data, even with `pg_restore -j`. Do not edit
or drop these functions.

`pgtoken_rs` wraps the extension's Rust codec and writes exactly its bytes.
`benchmarks/pgtoken_client.py` is a pure-Python version that reads every codec but always writes
plain `freq`, which compares equal to the `freq_lz4` form. `benchmarks/test_client.py` checks both.

</details>

## Tests

```sh
cd core && cargo test                            # codecs, no PostgreSQL
cd ext  && cargo pgrx test pg14
uv run python py/tests/test_psycopg.py           # Python client, needs a running server
bash scripts/test_dump_restore.sh                # pg_dump and pg_restore -j, needs a running server
uv run python scripts/test_tokenizer_parity.py   # server tokenizer vs tiktoken, 600 docs
```

CI runs both cargo suites on PostgreSQL 14 to 18, plus `cargo fmt`, `clippy -D warnings` and an
install check.

## License

Apache-2.0
