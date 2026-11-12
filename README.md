# pgtoken

[![CI](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml/badge.svg)](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml)

Store text in PostgreSQL as token IDs instead of UTF-8.

A model has to tokenize text before it can use it. `pgtoken` stores the text already tokenized, as
a compressed column of token IDs. That column is about 2.1x smaller than the text on disk, in WAL,
and in the buffer cache, and it reads straight back as IDs, with nothing to re-tokenize. When you
need the characters, `pgtoken.text()` hands them back.

You bring the tokenizer: tiktoken, HuggingFace, SentencePiece, or your own. The database only needs
its vocabulary size, plus a `token_id -> bytes` table if you want text back. It never tokenizes
anything itself. For OpenAI's `o200k_base`, both come bundled, so there is nothing to set up.

<sub>Background: [blog](https://www.kshivendu.dev/blog/token-storage) ·
[paper](https://arxiv.org/abs/2608.02376)</sub>

> Early days; the API may change. CI covers PostgreSQL 14 through 18.

## Install

Needs Rust and [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx).

```sh
cargo install cargo-pgrx --locked
cargo pgrx init --pg14 $(which pg_config)

git clone https://github.com/KShivendu/pgtoken.git
cd pgtoken/ext && cargo pgrx install --release
```

No PostgreSQL to hand? `setup_pg.sh` installs one under `~/.local/share`, no root needed. There is
nothing to configure: trained rankings and token mappings are stored in the database itself.

## Usage

```sql
CREATE EXTENSION pgtoken;
SELECT pgtoken.create_vocabulary('o200k');   -- bundled: o200k_base, freq compression, text()

CREATE TABLE documents (id bigserial PRIMARY KEY, body tokens.o200k);

INSERT INTO documents (body) VALUES ('{13225,11,2375,0}');   -- ids from tiktoken's o200k_base
SELECT body FROM documents;                                  -- {13225,11,2375,0}
SELECT body::text FROM documents;                            -- Hello, world!
```

`SELECT body` shows token IDs, and so does `pg_dump`. `body::text` (or `pgtoken.text(body)`, the
same function) gives the text back.

`create_vocabulary` creates the domain `tokens.o200k` that you put on the column, and it sets
`STORAGE EXTERNAL` for you. Each vocabulary is its own type, so PostgreSQL won't let you mix values
between them. That is on purpose: a token ID means nothing outside the tokenizer that produced it.

`o200k` is a **preset**: the extension ships its frequency ranking and its `token_id -> bytes` table
inside the binary. Use it under another name with `create_vocabulary('chat', preset => 'o200k')`,
or store plain 3-byte ids with `compression => 'raw'`. Any other tokenizer declares its ID space,
and the storage width follows (50257 ids need 2 bytes):

```sql
SELECT pgtoken.create_vocabulary('gpt2', vocab_size => 50257);
```

### Getting text back

`pgtoken.text()` needs a `token_id -> bytes` table. A preset has one built in. For anything else,
export it from your tokenizer. `scripts/load_mapping.py` builds one for any tiktoken encoding:

```sh
uv run python scripts/load_mapping.py --encoding cl100k_base --vocabulary cl100k
```

The table maps IDs to bytes rather than strings, because one character can span two tokens. pgtoken
joins the bytes and decodes them as UTF-8 once, at the end. Bytes that do not form whole characters, such as a chunk
cut in the middle of 🦜 (three o200k tokens), come back as U+FFFD (`�`), never as an empty string. The mapping is write-once, which is what
lets `pgtoken.text` be `IMMUTABLE` and back an index:

```sql
CREATE INDEX ON documents USING gin (to_tsvector('english', pgtoken.text(body)));
```

### Which read to use

| you want | write | cost |
| --- | --- | --- |
| stored bytes, no server work | `SELECT body`, binary mode | none, the fast path |
| token IDs for SQL or `train` | `body::int[]` | 4 B/token |
| text for a human | `body::text` | needs a mapping |

### Coming from a text column

Dedup, group and join are the same SQL, since `=` works:

```sql
SELECT DISTINCT body FROM chunks;
SELECT * FROM a JOIN b ON a.body = b.body;
```

To match on content, pass the token IDs your tokenizer makes. For anything that reads the text, cast
the column with `body::text` (presets have a mapping; otherwise load one first):

```sql
SELECT * FROM chunks WHERE body = '{24912,2375}';   -- 'hello world'
SELECT * FROM chunks WHERE body::text LIKE '%invoice%';
CREATE INDEX ON chunks USING gin (to_tsvector('english', body::text));
```

Dedup and join compare token IDs, so a tokenizer that lowercases treats `'Hello'` and `'hello'` as
the same. A `UNIQUE` constraint on the column is not supported.

A literal compared with a column takes the column's vocabulary, so `body = '{24912,2375}'` needs
no cast. The one exception is an `IN` list of nine or more literals, which PostgreSQL hashes before
it knows the column: cast those, as in `body IN ('{1}'::tokens.o200k, ...)`.

## Use from Python

`pgtoken-rs` (in `py/`) reads and writes these columns as numpy arrays of token IDs. It encodes on
your machine with the extension's own Rust codec. Install it from this checkout, with the psycopg
extra:

```sh
uv pip install "./py[psycopg]"
```

```python
import psycopg
from pgtoken_rs.psycopg import register

conn = psycopg.connect("dbname=mydb", autocommit=True)
vocab = register(conn, "o200k")   # asks the server for the vocabulary's codec and ranking

conn.execute("INSERT INTO documents (body) VALUES (%s)", [vocab.encode([24912, 2375])])
ids = conn.execute("SELECT body FROM documents", binary=True).fetchone()[0]
print(ids)                        # [24912  2375], a numpy uint32 array
```

`register` needs only a normal connection. It works on a hosted server too, because the ranking
comes back as bytes from `pgtoken.vocabulary_codec`. Binary `COPY` works as well. See
[py/README.md](py/README.md) for `COPY`, several vocabularies, and the errors it raises.

## Compression

| method | size | decode | training |
| --- | --: | --: | --- |
| `raw` (default without a preset) | 1-3 B/token | 0.2-0.4 µs | no |
| `freq` (default for a preset) | ~1.9 B/token | 3.6 µs | yes, or bundled in a preset |

`raw` packs IDs at the fixed width `vocab_size` implies. `freq` remaps them to frequency rank and
varint-packs, so common tokens cost a single byte:

```sql
SELECT pgtoken.create_vocabulary('corpus', vocab_size => 200019, compression => 'freq');
SELECT pgtoken.train('corpus', 'SELECT ids FROM my_corpus');
```

The o200k preset ships a ranking counted over an equal mix of English prose, code and Hindi. Train
your own when your data looks different. A ranking only stores the tokens the corpus actually used.
For a skewed corpus that is tens of bytes, not the 800 KB a full vocabulary would need. Tokens it
never saw still encode fine, just a little wider. A vocabulary is fixed once created, so to change one, make a new vocabulary and
`ALTER TABLE ... TYPE tokens.<new>`.

## Benchmarks

C4 English text, 512-token chunks, gpt2 ids from HuggingFace `tokenizers` v1, PostgreSQL 14. Every
pgtoken number below uses a real `tokens.<name>` column. Regenerate with the scripts in
`benchmarks/`.

**Storage.** With 20000 rows, pgtoken is about 2.1x smaller on disk, in WAL, and in the cache:

| column | payload/row | relation | WAL/row | docs/page | past TOAST |
| --- | --: | --: | --: | --: | --: |
| `text` (pglz) | 1893 B | 45.9 MB | 2091 B | 6.32 | 7000 / 20000 |
| `text` (lz4) | 1832 B | 42.6 MB | 1953 B | 4.32 | 1500 / 20000 |
| `pgtoken` freq | **901 B** (2.10x) | **21.0 MB** (2.19x) | **993 B** (2.10x) | **8.00** | **0** / 20000 |

Every token value stays inline. A third of the text rows spill to TOAST instead, and each spill
costs an extra fetch on every read. A 4 KB embedding column dilutes the ratios to about 1.2x, since
it is TOASTed in every table anyway.

**Codec cost** with no database in the way (`cargo run --release --example codec_bench`, one
dedicated CPU):

| codec | encode | decode | bytes/token |
| --- | --: | --: | --: |
| `raw16` | 0.36 µs | 0.22 µs | 2.02 |
| `raw24` | 0.98 µs | 0.39 µs | 3.02 |
| `freq` | 5.12 µs | **3.60 µs** | **1.89** |

**Read and write speed.** pgtoken suits agent workloads, where an LLM reads and writes token IDs.
With a `text` column, every write has to detokenize and every read has to tokenize again. With
pgtoken, the client sends and receives the stored bytes in binary mode. It encodes and decodes them
with `pgtoken_rs`, a Python module in `py/` that runs the extension's own Rust codec.

Setup: 2000 chunks, 300 repeats, `synchronous_commit = off`, client pinned to one core, CPU cache
flushed before every timed call. Times are the median, end to end, as the client sees them:

| column | read 1 row | read 10 rows | read 100 rows | write 1 row | write 100 rows, per row |
| --- | --: | --: | --: | --: | --: |
| `text` (lz4) | 332 µs | 734 µs | 3311 µs | 341 µs | 55.6 µs |
| `text` (pglz) | 346 µs | 755 µs | 3705 µs | 414 µs | 104.5 µs |
| `pgtoken` raw | 290 µs | 417 µs | 1273 µs | 280 µs | **12.6 µs** |
| `pgtoken` freq | **286 µs** | **415 µs** | **1265 µs** | **276 µs** | 19.2 µs |

Reading 100 rows shows where the time goes. The lz4 text column spends 1.06 ms fetching 233 KB and
2.24 ms tokenizing it. pgtoken freq spends 0.68 ms fetching 90 KB and 0.58 ms decoding it. That is
2.6x faster, and it was faster in all 300 repeats. A single row gains less (1.16x), because the round
trip to the server dominates.

The client codec matters. The same freq reads through the pure-Python reference codec
(`benchmarks/pgtoken_client.py`) take 6.1 ms for 100 rows, slower than text. Reading `body::int[]`
with no client codec takes 7.0 ms. The server builds an array (about 14 µs a row), and psycopg then
makes 512 Python ints a row (about 34 µs). Use `pgtoken_rs` or your own binding of `pgtoken-core`.
`uv sync` builds `pgtoken_rs`, or run `pip install ./py`.

## Limitations

- **Text needs a mapping.** A preset has one. Otherwise, without `load_mapping`, reads give you
  token IDs, and `psql` shows integers.
- **No `ORDER BY`** on the column. A compressed value has no meaningful byte order, because the
  `freq` codec remaps ids before packing. `=`, `<>`, `DISTINCT`, and `GROUP BY` do work (equality is
  a byte compare within one vocabulary), so sort on `pgtoken.text(body)` when you need alphabetical.
- **A column must name a vocabulary.** A column typed as bare `pgtoken.tokens` accepts inserts but
  errors on read, because PostgreSQL applies the type modifier too late to reject them up front. Fix
  it with `ALTER TABLE ... TYPE tokens.<name>`.
- **Binary writes are trusted.** Text and `int[]` input bounds-check every id. `COPY BINARY` and the
  `bytea` cast check only the 12-byte header, to keep the write path fast.
- **A vocabulary's name and id are reserved forever**, even after its domain is dropped.
- **Rankings are corpus-specific.** The ratios drop as the ranking and your data drift apart.

## Reference

| function | returns | |
| --- | --- | --- |
| `create_vocabulary(name [, vocab_size] [, compression] [, id] [, preset])` | `int` | also creates `tokens.<name>` |
| `train(name, query [, max_ranks])` | `text` | ranking for `freq`, write-once; not for presets |
| `load_mapping(name, query)` | `text` | `token_id -> bytes`, write-once; not for presets |
| `text(tokens)` | `text` | detokenize, same as `body::text`; needs a mapping; `IMMUTABLE` |
| `vocabulary_info(name)` | record | size, compression, width, preset, per-artefact stats |
| `vocabulary_codec(name)` | record | id, codec and ranking bytes, for a client codec |
| `drop_vocabulary(name)` | | drops the domain; the id stays reserved |
| `token_count(tokens)` | `int` | header only, no decode |
| `describe(tokens)` | record | codec, vocabulary, sizes |

Everything lives in the `pgtoken` schema. The casts are `int[] → tokens` (assignment),
`tokens → int[]` (explicit) and `tokens → text` (explicit, detokenizes like `pgtoken.text`). A
bare `text(body)` works only with `pgtoken` on `search_path`; without it PostgreSQL reports that
`text(tokens.o200k)` does not exist, so write `body::text`. For the raw stored bytes, call `pgtoken.tokens_send(body)`. The `=` and
`<>` operators compare two values within one vocabulary, and a default hash operator class powers
`DISTINCT`, `GROUP BY`, hash joins and hash indexes. The two operators are created in `pg_catalog`,
so they work under any `search_path`.

**Where rankings and mappings live.** A preset's are compiled into the extension, and its
vocabularies take ids from a reserved range (65280-65535 for o200k), so a value is readable with
nothing else in the database. `train` and `load_mapping` store theirs as SQL functions,
`pgtoken.artefact_<id>_ranking()` and `pgtoken.artefact_<id>_mapping()`, that return the bytes. So
they are transactional, WAL-logged, copied to replicas, private to their database, and included in
`pg_dump`. A dump restores functions before any table data, so a `freq` column restores even with
`pg_restore -j`. Do not edit or drop these functions.

`benchmarks/pgtoken_client.py` is a Python reference codec, and `pgtoken_rs` (in `py/`) wraps the
Rust one. Both are byte-identical to the extension in both directions, and
`benchmarks/test_client.py` asserts that.

## Tests

```sh
cd core && cargo test          # codecs, no PostgreSQL
cd ext  && cargo pgrx test pg14
uv run python py/tests/test_psycopg.py   # Python client, against a running server
bash scripts/test_dump_restore.sh        # pg_dump and pg_restore -j, against a running server
```

CI runs both cargo suites across PostgreSQL 14 through 18, plus `cargo fmt`, `clippy -D warnings`,
and an install check.

## License

Apache-2.0
