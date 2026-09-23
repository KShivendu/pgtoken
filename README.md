# pgtoken

[![CI](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml/badge.svg)](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml)

Store text in PostgreSQL as token IDs instead of UTF-8.

A model has to tokenize text before it can use it. `pgtoken` stores the text already tokenized, as
a compressed column of token IDs. That column is about 2.1x smaller than the text on disk, in WAL,
and in the buffer cache, and it reads straight back as IDs, with nothing to re-tokenize. When you
need the characters, `pgtoken.text()` hands them back.

You bring the tokenizer: tiktoken, HuggingFace, SentencePiece, or your own. The database only needs
its vocabulary size, plus a `token_id -> bytes` table if you want text back. It never tokenizes
anything itself.

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

No PostgreSQL to hand? `setup_pg.sh` installs one under `~/.local/share`, no root needed. Then set
`pgtoken.table_dir` in `postgresql.conf`. Trained rankings and token mappings live there.

## Usage

```sql
CREATE EXTENSION pgtoken;

-- Declare the ID space once. The storage width follows: 200019 ids need 3 bytes.
SELECT pgtoken.create_vocabulary('o200k', vocab_size => 200019);

CREATE TABLE documents (id bigserial PRIMARY KEY, body tokens.o200k);

INSERT INTO documents (body) VALUES ('{24912,2375}');   -- ids from your tokenizer
SELECT body FROM documents;                             -- {24912,2375}
```

`create_vocabulary` creates the domain `tokens.o200k` that you put on the column, and it sets
`STORAGE EXTERNAL` for you. Each vocabulary is its own type, so PostgreSQL won't let you mix values
between them. That is on purpose: a token ID means nothing outside the tokenizer that produced it.

### Getting text back

`pgtoken.text()` needs a `token_id -> bytes` table exported from your tokenizer.
`scripts/load_mapping.py` builds one for any tiktoken encoding:

```sh
uv run python scripts/load_mapping.py --encoding o200k_base --vocabulary o200k
```

```sql
SELECT pgtoken.text(body) FROM documents;   -- 'Hello, world!'
```

The table maps IDs to bytes rather than strings, because one character can span two tokens. pgtoken
joins the bytes and decodes them as UTF-8 once, at the end. The mapping is write-once, which is what
lets `pgtoken.text` be `IMMUTABLE` and back an index:

```sql
CREATE INDEX ON documents USING gin (to_tsvector('english', pgtoken.text(body)));
```

### Which read to use

| you want | write | cost |
| --- | --- | --- |
| stored bytes, no server work | `SELECT body`, binary mode | none, the fast path |
| token IDs for SQL or `train` | `body::int[]` | 4 B/token |
| text for a human | `pgtoken.text(body)` | needs a mapping |

## Compression

| method | size | decode | training |
| --- | --: | --: | --- |
| `raw` (default) | 1-3 B/token | 0.2-0.4 µs | no |
| `freq` | ~1.9 B/token | 3.6 µs | yes |

`raw` packs IDs at the fixed width `vocab_size` implies. `freq` remaps them to frequency rank and
varint-packs, so common tokens cost a single byte:

```sql
SELECT pgtoken.create_vocabulary('corpus', vocab_size => 200019, compression => 'freq');
SELECT pgtoken.train('corpus', 'SELECT ids FROM my_corpus');
```

The ranking only stores the tokens your corpus actually used. For a skewed corpus that is tens of
bytes, not the 800 KB a full vocabulary would need. Tokens it never saw still encode fine, just a
little wider. A vocabulary is fixed once created, so to change one, make a new vocabulary and
`ALTER TABLE ... TYPE tokens.<new>`.

## Benchmarks

C4 English prose, 512-token chunks, gpt2 ids via HuggingFace `tokenizers` v1, PostgreSQL 14, 20000
rows. Regenerate with the scripts in `benchmarks/`.

**Storage** is smaller on disk, in WAL, and in the cache, by about 2.1x, whatever tokenizer you use:

| column | payload/row | relation | WAL/row | docs/page | past TOAST |
| --- | --: | --: | --: | --: | --: |
| `text` (pglz) | 1893 B | 45.9 MB | 2098 B | 6.32 | 7000 / 20000 |
| `text` (lz4) | 1832 B | 42.6 MB | 1955 B | 4.32 | 1500 / 20000 |
| `pgtoken` freq | **901 B** (2.10x) | **21.0 MB** (2.19x) | **995 B** (2.11x) | **8.00** | **0** / 20000 |

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

**Latency.** pgtoken suits agent workloads, where an LLM is the main reader or writer and already
works in token IDs. With a plain `text` column, every read has to tokenize and every write has to
detokenize. pgtoken skips both, because it stores and returns the IDs directly.

How much that saves depends on your tokenizer, so use a fast one. HuggingFace `tokenizers` v1
tokenizes a 512-token chunk in single-digit microseconds, where an older or cold tiktoken table
takes hundreds. pgtoken itself decodes in 0.22 µs for `raw16` and 3.6 µs for `freq`. Either way, the
steady win is the 2.1x compression, and it compounds when a single answer touches hundreds of rows.

## Limitations

- **Text needs a mapping.** Without `load_mapping`, reads give you token IDs, and `psql` shows
  integers.
- **No `=`, `ORDER BY`, `GROUP BY`, or `DISTINCT`** on the column. A compressed value has no
  meaningful byte order, and there is no equality operator yet.
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
| `create_vocabulary(name, vocab_size [, compression] [, id])` | `int` | also creates `tokens.<name>` |
| `train(name, query [, max_ranks])` | `text` | ranking for `freq`, write-once |
| `load_mapping(name, query)` | `text` | `token_id -> bytes`, write-once |
| `text(tokens)` | `text` | detokenize; needs a mapping; `IMMUTABLE` |
| `vocabulary_info(name)` | record | size, compression, width, per-artefact stats |
| `drop_vocabulary(name)` | | drops the domain; the id stays reserved |
| `token_count(tokens)` | `int` | header only, no decode |
| `describe(tokens)` | record | codec, vocabulary, sizes |

Everything lives in the `pgtoken` schema. The casts are `int[] → tokens` (assignment) and
`tokens → int[]` (explicit). For the raw stored bytes, call `pgtoken.tokens_send(body)`.
`pgtoken.table_dir` reloads on `SIGHUP` and is not session-settable, so two sessions can never
decode the same value differently.

`benchmarks/pgtoken_client.py` is a Python reference codec, byte-identical to the extension in both
directions. `benchmarks/test_client.py` asserts that.

## Tests

```sh
cd core && cargo test          # codecs, no PostgreSQL
cd ext  && cargo pgrx test pg14
```

Clear `$pgtoken.table_dir` between runs. Rankings and mappings are files, so they survive the
rollback that resets everything else, and both refuse to overwrite an existing one. CI runs both
suites across PostgreSQL 14 through 18, plus `cargo fmt`, `clippy -D warnings`, and an install check.

## License

Apache-2.0
