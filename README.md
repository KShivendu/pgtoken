# pgtoken

[![CI](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml/badge.svg)](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml)

Store text in PostgreSQL as token IDs instead of UTF-8.

Agents read and write token IDs, not characters. A `text` column makes them re-tokenize on every
read; `pgtoken` stores the IDs directly, compressed, and hands them back as-is. When something
downstream needs prose, `pgtoken.text()` gives it to you.

**No tokenizer.** You tokenize with whatever you already use — tiktoken, HuggingFace,
SentencePiece, your own. The database needs two things from it: how many token IDs it has, and
optionally a `token_id -> bytes` table if you want prose back. It never sees a merge table and
never spends a cycle tokenizing.

<sub>Background: [blog](https://www.kshivendu.dev/blog/token-storage) ·
[paper](https://arxiv.org/abs/2608.02376)</sub>

> Early days. The API may still change. CI covers PostgreSQL 14 through 18.

## Install

Needs Rust and [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx).

```sh
cargo install cargo-pgrx --locked
cargo pgrx init --pg14 $(which pg_config)

git clone https://github.com/KShivendu/pgtoken.git
cd pgtoken/ext && cargo pgrx install --release
```

No PostgreSQL to hand? `setup_pg.sh` installs one under `~/.local/share`, no root needed.

Set `pgtoken.table_dir` in `postgresql.conf` — it is where trained rankings and token mappings
live.

## Usage

```sql
CREATE EXTENSION pgtoken;

-- Declare the ID space once. The storage width follows: 200019 ids need 3 bytes.
SELECT pgtoken.create_vocabulary('o200k', vocab_size => 200019);

CREATE TABLE documents (id bigserial PRIMARY KEY, body tokens.o200k);

INSERT INTO documents (body) VALUES ('{24912,2375}');   -- ids from your tokenizer
SELECT body FROM documents;                             -- {24912,2375}
```

`create_vocabulary` also creates the domain `tokens.o200k`, which is what you put on the column.
Two vocabularies are two types, so PostgreSQL refuses to move values between them — token IDs
mean nothing outside the tokenizer that produced them.

The type sets `STORAGE EXTERNAL` itself, so there is no `ALTER TABLE` to remember.

### Getting prose back

Load a `token_id -> bytes` mapping once, exported from the same tokenizer:

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
rows = [(i, enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)]
# COPY rows into vocab_staging(id int, bytes bytea)
```

```sql
SELECT pgtoken.load_mapping('o200k', 'SELECT id, bytes FROM vocab_staging');
SELECT pgtoken.text(body) FROM documents;   -- 'hello world'
```

The mapping is write-once, which is what lets `pgtoken.text` be `IMMUTABLE` and back an index:

```sql
CREATE INDEX ON documents USING gin (to_tsvector('english', pgtoken.text(body)));
```

Write it schema-qualified. On a `tokens.<name>` column a bare `text(body)` is PostgreSQL's
cast-to-`text` syntax, not a call to this function, and it silently returns the ID list.

### Which read to use

| you want | write | cost |
| --- | --- | --- |
| the stored bytes, no server work | `SELECT body`, binary mode | none — this is the fast path |
| the same, from a driver that makes binary awkward | `body::bytea` | hex, 2× on the wire |
| token IDs for SQL-side work | `body::int[]` | 4 B/token |
| prose, for a human | `pgtoken.text(body)` | needs a mapping |

## Compression

| method | size | decode | needs training |
| --- | --: | --: | --- |
| `raw` (default) | 1–3 B/token | 0.3–0.4 µs | no |
| `freq` | ~1.9 B/token | 4 µs | yes |

Per 512-token chunk over a 200k vocabulary. `raw` packs IDs at the fixed width `vocab_size`
implies. `freq` remaps them to frequency rank and varint-packs, so common tokens cost one byte.

```sql
SELECT pgtoken.create_vocabulary('corpus', vocab_size => 200019, compression => 'freq');
SELECT pgtoken.train('corpus', 'SELECT ids FROM my_corpus');
```

The ranking holds only the tokens your corpus contained — 28 bytes for one skewed corpus, not the
800 KB a full vocabulary would need. Tokens it never saw still encode losslessly, just wider.

A vocabulary is immutable: size, compression, ranking and mapping are fixed once set, because
stored values reference its id. Changing your mind means a new one:

```sql
SELECT pgtoken.create_vocabulary('corpus_v2', vocab_size => 200019);
ALTER TABLE documents ALTER COLUMN body TYPE tokens.corpus_v2 USING body::tokens.corpus_v2;
```

If the column backs a `pgtoken.text` index, load the new vocabulary's mapping *before* the
`ALTER`: rebuilding the index detokenizes every row, and it will fail on a vocabulary that has
none yet.

## Benchmarks

C4 English, 512-token chunks, o200k, PostgreSQL 14, one pinned core. The workload is an agent: it
wants token IDs back, so a `text` column pays a tokenize on every read.

**Read**, µs per query:

| column | fan-out 1 | fan-out 10 | fan-out 100 | wire @100 |
| --- | --: | --: | --: | --: |
| `text` (pglz) | 1176 | 4066 | 19258 | 233 KB |
| `pgtoken` raw | **748** (1.6×) | **903** (4.5×) | **2588** (7.4×) | 147 KB |
| `pgtoken` freq | 938 (1.3×) | 1888 (2.2×) | 10485 (1.8×) | **88 KB** |

At fan-out 100 the `text` column spends 16.5 ms of its 19.3 ms tokenizing. That is the cost
`pgtoken` removes, and it recurs on every read. Storage is ~2.1× smaller in payload, relation
size and WAL.

The codec alone, no database in the way (`cd core && cargo run --release --example codec_bench`):

| codec | encode | decode | bytes/token |
| --- | --: | --: | --: |
| `raw16` | 0.57 µs | 0.26 µs | 2.02 |
| `raw24` | 1.11 µs | 0.42 µs | 3.02 |
| `freq` | 5.38 µs | **4.06 µs** | **1.89** |

So `freq` decodes in ~4 µs against ~250 µs to tokenize the same text; the end-to-end figures
understate it because the Python client pays numpy overhead on 512-element arrays.

> The end-to-end numbers predate the type, and `benchmarks/bench_readwrite.py` still calls removed
> functions — it needs porting to vocabularies before it runs again.

## Limitations

- **Prose needs a mapping.** Without `load_mapping`, reads give you token IDs and `psql` shows
  integers.
- **No `=`, `ORDER BY`, `GROUP BY` or `DISTINCT`** on the column. Byte order of a compressed value
  is meaningless, and there is no equality operator yet.
- **A column must name a vocabulary.** A bare `pgtoken.tokens` column accepts inserts and fails on
  read — PostgreSQL applies a type modifier after the input function runs, so there is nowhere
  earlier to refuse. Recoverable with `ALTER TABLE ... TYPE tokens.<name>`.
- **Binary writes are trusted.** Text and `int[]` input bounds-check every id; `COPY BINARY` and
  the `bytea` cast check only the 12-byte header, because scanning the payload would cost the
  write path the speed it exists for.
- **A vocabulary's name and id are reserved forever**, even after its domain is dropped.
- **Rankings are corpus-specific.** Ratios drop as the ranking and your data diverge.

## Reference

| function | returns | |
| --- | --- | --- |
| `create_vocabulary(name, vocab_size [, compression] [, id])` | `int` | also creates `tokens.<name>` |
| `train(name, query [, max_ranks])` | `text` | ranking for `freq`, write-once |
| `load_mapping(name, query)` | `text` | `token_id -> bytes`, write-once |
| `text(tokens)` | `text` | detokenize; needs a mapping; `IMMUTABLE` |
| `vocabulary_info(name)` | record | size, compression, width, and per artefact: fill, sha256, bytes |
| `drop_vocabulary(name)` | | drops the domain; the id stays reserved |
| `token_count(tokens)` | `int` | header only, no decode |
| `describe(tokens)` | record | codec, vocabulary, sizes |

All in the `pgtoken` schema. Casts: `int[] → tokens` (assignment); `tokens → int[]`,
`tokens → bytea`, `bytea → tokens` (explicit).

Setting: `pgtoken.table_dir`, where rankings and mappings live (`SIGHUP`). Not session-settable on
purpose — two sessions must never decode one value differently.

`benchmarks/pgtoken_client.py` is a reference codec in Python, byte-compatible with the extension
in both directions (`benchmarks/test_client.py` asserts it).

## Tests

```sh
cd core && cargo test          # codecs, no PostgreSQL needed
cd ext  && cargo pgrx test pg14
```

Clear `$pgtoken.table_dir` between runs: rankings and mappings are files, they survive the
rollback that resets everything else, and both refuse to overwrite.

CI runs both across PostgreSQL 14–18, plus `cargo fmt`, `clippy -D warnings`, and an install check.

## License

Apache-2.0
