# pgtoken

[![CI](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml/badge.svg)](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml)

Store text in PostgreSQL as token IDs instead of UTF-8.

A model has to tokenize text before it can use it. `pgtoken` stores the text already tokenized, as
a compressed column of token IDs. For 512-token chunks of English, that column is about 2.3x smaller
than a compressed `text` column on disk, in WAL, and in the buffer cache. It reads straight back as
IDs, with nothing to re-tokenize. When you need the characters, `pgtoken.text()` hands them back.

You bring the tokenizer: tiktoken, HuggingFace, SentencePiece, or your own. The database only needs
its vocabulary size, plus a `token_id -> bytes` table if you want text back. For OpenAI's
`o200k_base`, both come bundled, and so does a server-side tokenizer, so you can also insert plain
text and let the database tokenize it.

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
INSERT INTO documents (body) VALUES ('Hello, world!');       -- or text: the server tokenizes it
SELECT body FROM documents;                                  -- {13225,11,2375,0}, twice
SELECT body::text FROM documents;                            -- Hello, world!, twice
SELECT * FROM documents WHERE body = 'Hello, world!';        -- both rows
```

`SELECT body` shows token IDs, and so does `pg_dump`, so restoring a dump never tokenizes anything.
`body::text` (or `pgtoken.text(body)`, the same function) gives the text back.

Input is read as token IDs only when it is a plain ID list: `{`, numbers separated by commas, `}`,
spaces allowed. Anything else is text, so `'{1,2,3}'` is three IDs while `'{"role": "user"}'` is
text. Only a vocabulary with a server tokenizer accepts text; today that is the o200k preset, which
matches tiktoken's `o200k_base` ID for ID (`scripts/test_tokenizer_parity.py`). Special tokens are
not recognised in text: `'<|endoftext|>'` is stored as the tokens of those 13 characters. Any other
vocabulary refuses text with `vocabulary "gpt2" has no server tokenizer; send token ids`.
`pgtoken.tokenize('Hello, world!', 'o200k')` returns the IDs without storing anything.

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
| `raw` (default without a preset) | 1-3 B/token | 0.1-0.4 µs | no |
| `freq` (default for a preset) | ~1.85 B/token | 1.6 µs | yes, or bundled in a preset |

`raw` packs IDs at the fixed width `vocab_size` implies. `freq` remaps them to frequency rank and
packs each rank as a LEB128 varint, so common tokens cost a single byte:

```sql
SELECT pgtoken.create_vocabulary('corpus', vocab_size => 200019, compression => 'freq');
SELECT pgtoken.train('corpus', 'SELECT ids FROM my_corpus');
```

The o200k preset ships a ranking counted over an equal mix of English prose, code and Hindi. Train
your own when your data looks different. A ranking only stores the tokens the corpus actually used.
For a skewed corpus that is tens of bytes, not the 800 KB a full vocabulary would need. Tokens it
never saw still encode fine, just a little wider. A vocabulary is fixed once created, so to change one, make a new vocabulary and
`ALTER TABLE ... TYPE tokens.<new>`.

### Value format

Every value is a small header and then the payload:

```
off  size  field
  0     1  tag = (version 1 << 4) | codec    codec 0=raw16 1=raw24 2=freq 3=raw8
  1     1  flags, all zero for now
  2     2  vocabulary id (u16 little-endian)
  4   1-5  token count (LEB128)
  *     -  payload: raw = count x width bytes, freq = one LEB128 rank per token
```

The header is 5 bytes for a value under 128 tokens and 6 bytes up to 16383. `'Hello, world!'` in
an o200k column is 13 bytes on disk, against 14 for `text`. A reader refuses an unknown tag or flag,
and every field has exactly one encoding, so equal token sequences are equal bytes. See
`core/src/header.rs` for the details.

## Benchmarks

C4 English text, 512-token chunks, gpt2 ids from HuggingFace `tokenizers` v1, PostgreSQL 14. Every
pgtoken number below uses a real `tokens.<name>` column. Regenerate with the scripts in
`benchmarks/`.

**Storage.** With 20000 rows, pgtoken is about 2.3x smaller on disk, in WAL, and in the cache. The
`text` rows are what PostgreSQL really stores, after its own pglz or lz4 compression:

| column | payload/row | relation | WAL/row | docs/page | past TOAST |
| --- | --: | --: | --: | --: | --: |
| `text` (pglz) | 1893 B | 45.9 MB | 2091 B | 6.32 | 7000 / 20000 |
| `text` (lz4) | 1832 B | 42.6 MB | 1953 B | 4.32 | 1500 / 20000 |
| `pgtoken` freq | **836 B** (2.27x) | **18.7 MB** (2.45x) | **927 B** (2.26x) | **9.00** | **0** / 20000 |

Every token value stays inline. A third of the text rows spill to TOAST instead, and each spill
costs an extra fetch on every read. A 4 KB embedding column dilutes the ratios to about 1.2x, since
it is TOASTed in every table anyway.

**Size by value length.** PostgreSQL compresses a `text` value only once its row passes about 2 KB.
Below that it stores the plain UTF-8, with any compression setting. pglz also keeps a value
uncompressed unless it saves at least 25%. A tokens column is compressed at every length. So the
gap depends on length and content. The table gives the mean `pg_column_size` per value, over 300
values per row, with the o200k preset. It comes from `benchmarks/bench_length_buckets.py`.

| content | o200k tokens | `text` uncompressed | `text` pglz | `text` lz4 | `tokens.o200k` | vs smaller text |
| --- | --: | --: | --: | --: | --: | --: |
| English | 15 | 74 B | 74 B | 74 B | 33 B | 2.23x |
| English | 123 | 591 B | 591 B | 591 B | 226 B | 2.61x |
| English | 183 | 870 B | 870 B | 870 B | 335 B | 2.60x |
| English | 488 | 2314 B | 1830 B | 1815 B | 878 B | 2.07x |
| English | 1952 | 9249 B | 5538 B | 6344 B | 3468 B | 1.60x |
| code | 68 | 281 B | 281 B | 281 B | 130 B | 2.16x |
| code | 1102 | 4508 B | 1723 B | 2039 B | 1986 B | **0.87x** |
| Hindi | 30 | 220 B | 220 B | 220 B | 56 B | 3.92x |
| Hindi | 447 | 3446 B | 1213 B | 1816 B | 759 B | 1.60x |

Three values from that run:

- 183 tokens of English, starting `old copy of Mr. Popper’s Penguins and we are having a lot of fun`:
  855 B of UTF-8. `text` stores 859 B under all three options, since the row is under 2 KB.
  `tokens.o200k` stores 330 B, 2.60x smaller.
- 1466 tokens of code, line after line of
  `CThostFtdcLimitPosiSField["OpenVolume"] = "int"`: 4398 B of UTF-8. pglz stores 871 B, lz4
  1149 B, `tokens.o200k` 3121 B, 3.6x larger than pglz. pglz and lz4 find the repeated lines. The
  freq codec packs each token on its own and cannot see repeats. In the 1102-token code row,
  tokens lose to pglz on 167 of 300 values.
- 14 tokens, `4): p. 332.\nGobburu, J.V.,`: 26 B of UTF-8. `text` stores 27 B, `tokens.o200k`
  32 B. Short strings of digits and punctuation cost about one token per two characters, and
  the header adds 5 bytes. A one-token value like `'hi'` is 3 B as `text` and 8 B as tokens.

**Codec cost** with no database in the way (`cargo run --release --example codec_bench`, one
dedicated CPU):

| codec | encode | decode | bytes/token |
| --- | --: | --: | --: |
| `raw16` | 0.52 µs | 0.10 µs | 2.01 |
| `raw24` | 1.15 µs | 0.35 µs | 3.01 |
| `freq` | 4.93 µs | **1.56 µs** | **1.85** |

**Read and write speed.** pgtoken suits agent workloads, where an LLM reads and writes token IDs.
With a `text` column, every write has to detokenize and every read has to tokenize again. With
pgtoken, the client sends and receives the stored bytes in binary mode. It encodes and decodes them
with `pgtoken_rs`, a Python module in `py/` that runs the extension's own Rust codec.

Setup: 2000 chunks, 300 repeats, `synchronous_commit = off`, client pinned to one core, CPU cache
flushed before every timed call. Times are the median, end to end, as the client sees them:

| column | read 1 row | read 10 rows | read 100 rows | write 1 row | write 100 rows, per row |
| --- | --: | --: | --: | --: | --: |
| `text` (lz4) | 349 µs | 769 µs | 3369 µs | 319 µs | 56.8 µs |
| `text` (pglz) | 362 µs | 823 µs | 3804 µs | 386 µs | 104.7 µs |
| `pgtoken` raw | 307 µs | 446 µs | 1307 µs | **267 µs** | **12.6 µs** |
| `pgtoken` freq | **304 µs** | **432 µs** | **1199 µs** | 271 µs | 21.4 µs |

Reading 100 rows shows where the time goes. The lz4 text column spends 1.10 ms fetching 233 KB and
2.25 ms tokenizing it. pgtoken freq spends 0.71 ms fetching 83 KB and 0.49 ms decoding it. That is
2.8x faster, and it was faster in all 300 repeats. A single row gains less (1.16x), because the round
trip to the server dominates.

The client codec matters. The same freq reads through the pure-Python reference codec
(`benchmarks/pgtoken_client.py`) take 8.2 ms for 100 rows, slower than text. Reading `body::int[]`
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
  `bytea` cast check only the header, to keep the write path fast.
- **A vocabulary's name and id are reserved forever**, even after its domain is dropped.
- **Rankings are corpus-specific.** The ratios drop as the ranking and your data drift apart.

## Reference

| function | returns | |
| --- | --- | --- |
| `create_vocabulary(name [, vocab_size] [, compression] [, id] [, preset])` | `int` | also creates `tokens.<name>` |
| `train(name, query [, max_ranks])` | `text` | ranking for `freq`, write-once; not for presets |
| `load_mapping(name, query)` | `text` | `token_id -> bytes`, write-once; not for presets |
| `text(tokens)` | `text` | detokenize, same as `body::text`; needs a mapping; `IMMUTABLE` |
| `tokenize(text, vocabulary)` | `int[]` | server tokenizer; o200k preset only |
| `vocabulary_info(name)` | record | size, compression, width, preset, per-artefact stats |
| `vocabulary_codec(name)` | record | id, codec and ranking bytes, for a client codec |
| `drop_vocabulary(name)` | | drops the domain; the id stays reserved |
| `token_count(tokens)` | `int` | header only, no decode |
| `describe(tokens)` | record | codec, vocabulary, sizes |

Everything lives in the `pgtoken` schema. The casts are `int[] → tokens` (assignment),
`text → tokens` (assignment, tokenizes like a text literal), `tokens → int[]` (explicit) and
`tokens → text` (explicit, detokenizes like `pgtoken.text`). A
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
uv run python scripts/test_tokenizer_parity.py   # server o200k tokenizer vs tiktoken, 600 docs
```

CI runs both cargo suites across PostgreSQL 14 through 18, plus `cargo fmt`, `clippy -D warnings`,
and an install check.

## License

Apache-2.0
