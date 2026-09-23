# pgtoken

[![CI](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml/badge.svg)](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml)

Store text in PostgreSQL as token IDs instead of UTF-8.

A RAG chunk or an agent's memory is text that a model will tokenize before it can use it.
`pgtoken` stores it already tokenized: an entropy-coded column of IDs that runs ~2.1x smaller than
the text on disk, in WAL, and in the buffer cache, and small enough that no value spills to TOAST.
Reads hand the IDs back to the model as-is, no re-tokenizing. When something downstream needs
characters, `pgtoken.text()` gives them back.

**No tokenizer.** You tokenize with whatever you already use: tiktoken, HuggingFace,
SentencePiece, your own. The database needs two things from it: how many token IDs it has, and
optionally a `token_id -> bytes` table if you want text back. It never sees a merge table and
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

If you have no PostgreSQL to hand, `setup_pg.sh` installs one under `~/.local/share`, no root
needed.

Set `pgtoken.table_dir` in `postgresql.conf`. Trained rankings and token mappings live there.

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
Two vocabularies are two types, so PostgreSQL refuses to move values between them. Token IDs
mean nothing outside the tokenizer that produced them.

The type sets `STORAGE EXTERNAL` itself, so there is no `ALTER TABLE` to remember.

### Getting text back

`pgtoken.text()` needs a `token_id -> bytes` table, exported from the tokenizer that produced your
IDs. `scripts/load_mapping.py` does that for any tiktoken encoding, in about a second for o200k:

```sh
uv run python scripts/load_mapping.py --encoding o200k_base --vocabulary o200k
# o200k_base: 200000 of 200019 ids have bytes, 19 unassigned
# mapped 200000 of 200019 ids for vocabulary o200k
```

```sql
SELECT pgtoken.text(body) FROM documents;   -- 'Hello, world!'
```

The table maps ids to bytes rather than strings, because one character often spans two tokens.
pgtoken concatenates the bytes and interprets UTF-8 once, at the end. To use a different tokenizer,
replace `export_pairs` in that script; the rest is generic.

o200k_base leaves 19 of its 200019 ids unassigned, and `decode_single_token_bytes` raises on them,
which is why a one-line comprehension over `range(n_vocab)` crashes. The script skips them and says
so. Its two special tokens are fine: they decode to the literal text `<|endoftext|>` and
`<|endofprompt|>`.

The mapping is write-once, which is what lets `pgtoken.text` be `IMMUTABLE` and back an index:

```sql
CREATE INDEX ON documents USING gin (to_tsvector('english', pgtoken.text(body)));
```

`create_vocabulary` also declares a `pgtoken.text` for each domain, so a bare `text(body)` calls
this function rather than PostgreSQL's cast-to-`text` syntax. That only works with `pgtoken` on
`search_path`; without it, write `pgtoken.text(body)`. An explicit `body::text` is still a cast and
still gives you the ID list.

### Which read to use

| you want | write | cost |
| --- | --- | --- |
| the stored bytes, no server work | `SELECT body`, binary mode | none, the fast path |
| token IDs for SQL-side work, and for `train` | `body::int[]` | 4 B/token |
| text, for a human | `pgtoken.text(body)` | needs a mapping |

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

The ranking holds only the tokens your corpus contained: 28 bytes for one skewed corpus, not the
800 KB a full vocabulary would need. Tokens it never saw still encode losslessly, just wider.

A vocabulary is immutable: size, compression, ranking and mapping are fixed once set, because
stored values reference its id. Changing your mind means a new one:

```sql
SELECT pgtoken.create_vocabulary('corpus_v2', vocab_size => 200019);
ALTER TABLE documents ALTER COLUMN body TYPE tokens.corpus_v2 USING body::tokens.corpus_v2;
```

If the column backs a `pgtoken.text` index, load the new vocabulary's mapping first, then run the
`ALTER`. Rebuilding the index detokenizes every row, and it fails on a vocabulary that has no
mapping yet.

## Benchmarks

C4 English prose, 512-token chunks, gpt2 ids via HuggingFace `tokenizers` v1, PostgreSQL 14,
20000 rows. Regenerate with `benchmarks/bench_storage_wal.py` and `benchmarks/bench_readwrite.py`.

**Storage, WAL and TOAST.** The token-native column is smaller on disk, in WAL, and in the buffer
cache, by roughly the same factor each way. This is the durable win, and it holds for any
tokenizer: IDs are IDs.

| column | payload/row | total relation | WAL/row | docs/8 KB page | past TOAST line |
| --- | --: | --: | --: | --: | --: |
| `text` (pglz) | 1893 B | 45.9 MB | 2098 B | 6.32 | 7000 / 20000 |
| `text` (lz4) | 1832 B | 42.6 MB | 1955 B | 4.32 | 1500 / 20000 |
| `pgtoken` freq | **901 B** (2.10x) | **21.0 MB** (2.19x) | **995 B** (2.11x) | **8.00** | **0** / 20000 |

Every token-native value stays inline while a third of the text rows spill to TOAST, and each
spilled value costs an extra index and heap fetch on every read. Add a 4 KB embedding column and
the relation and WAL ratios fall to ~1.2x, because the embedding dominates the row and is TOASTed
in every table.

**Codec cost**, no database in the way (`cd core && cargo run --release --example codec_bench`),
one quiet core:

| codec | encode | decode | bytes/token |
| --- | --: | --: | --: |
| `raw16` | 0.57 µs | 0.26 µs | 2.02 |
| `raw24` | 1.11 µs | 0.42 µs | 3.02 |
| `freq` | 5.38 µs | **4.06 µs** | **1.89** |

**Read latency.** A fast tokenizer reshapes the read story. Re-tokenizing a 512-token chunk with
`tokenizers` v1 costs single-digit microseconds, not the hundreds a cold tiktoken table shows, so
pgtoken does not win reads by dodging an expensive tokenize any more. It wins on the payload:
~2.1x fewer bytes off disk and over the wire, no TOAST fetch, and a `raw16` unpack in ~0.26 µs.
The `freq` codec spends ~4 µs decoding to buy the smallest payload; `raw16` keeps both the bytes
and the CPU low, and is the better default when reads dominate.

## Limitations

- **Text needs a mapping.** Without `load_mapping`, reads give you token IDs and `psql` shows
  integers.
- **No `=`, `ORDER BY`, `GROUP BY` or `DISTINCT`** on the column. Byte order of a compressed value
  is meaningless, and there is no equality operator yet.
- **A column must name a vocabulary.** A bare `pgtoken.tokens` column accepts inserts and fails on
  read. PostgreSQL applies a type modifier after the input function runs, so there is nowhere
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

All in the `pgtoken` schema. Casts: `int[] → tokens` (assignment) and `tokens → int[]`
(explicit). For the stored bytes without going through binary mode, call
`pgtoken.tokens_send(body)`.

Setting: `pgtoken.table_dir`, where rankings and mappings live (`SIGHUP`). Not session-settable on
purpose: two sessions must never decode one value differently.

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
