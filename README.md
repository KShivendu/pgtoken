# pgtoken

[![CI](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml/badge.svg)](https://github.com/KShivendu/pgtoken/actions/workflows/ci.yml)

Store text in PostgreSQL as token IDs instead of UTF-8.

Agents read and write token IDs, not characters. A `text` column makes them re-tokenize on
every read; `pgtoken` stores the IDs directly, compressed, and hands them back as-is.

**No tokenizer.** You tokenize with whatever you already use — tiktoken, HuggingFace,
SentencePiece, your own — and the database stays out of it. It only needs to know how many token
IDs your tokenizer has, so it can pick a storage width; it never sees a merge table and never
spends a cycle tokenizing.

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

## Usage

```sql
CREATE EXTENSION pgtoken;

-- Declare the ID space once. The storage width follows from it: 200019 ids need 3 bytes.
SELECT pgtoken.create_vocabulary('o200k', vocab_size => 200019);

CREATE TABLE documents (id bigserial PRIMARY KEY, body tokens.o200k);

-- token IDs from your tokenizer, client-side
INSERT INTO documents (body) VALUES ('{24912,2375}');

SELECT body FROM documents;          -- binary mode: the stored bytes, no server work
SELECT body::bytea FROM documents;   -- any client: hex, decode with the reference codec
SELECT body::int[] FROM documents;   -- {24912,2375}
```

`create_vocabulary` also creates the domain `tokens.o200k`, which is what you put on the column.
Two vocabularies are two types, so PostgreSQL refuses to move values between them by assignment —
token IDs are only meaningful to the tokenizer that produced them.

The type sets `STORAGE EXTERNAL` itself, so PostgreSQL will not try to compress a payload that is
already compressed. There is no `ALTER TABLE` to remember.

**Pick your read by what the caller is.** In binary mode `SELECT body` hands over exactly what is
on disk — that is the fast path, and the reason this extension exists. `body::bytea` is the
fallback for drivers that make binary mode awkward, at hex's 2× expansion but still with no
server-side work. `body::int[]` is for SQL-side use; it costs 4 bytes per token on the wire, more
than the blob it came from.

**For a human-facing reader**, load a `token_id -> bytes` mapping once — export it from your
tokenizer, the same way you already export the vocabulary size:

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

## Compression

| method | size | decode | needs training |
| --- | --: | --: | --- |
| `raw` (default) | 1–3 B/token | 0.3–0.4 µs | no |
| `freq` | ~1.9 B/token | 4 µs | yes |

Per 512-token chunk over a 200k vocabulary. `raw` packs IDs at the fixed width `vocab_size`
implies. `freq` remaps them to frequency rank and packs with a varint, so common tokens cost one
byte.

`freq` needs a ranking, trained from any query returning `int[]`:

```sql
SELECT pgtoken.create_vocabulary('corpus', vocab_size => 200019, compression => 'freq');
SELECT pgtoken.train('corpus', 'SELECT ids FROM my_corpus');
```

The ranking holds only the tokens your corpus actually contained — for one skewed corpus that is
28 bytes, not the 800 KB a full vocabulary would need. Tokens it never saw still encode
losslessly, just a little wider.

A vocabulary is immutable: its size, compression and ranking are fixed once set, because stored
values reference its id. Changing your mind means a new vocabulary and an `ALTER TABLE`:

```sql
SELECT pgtoken.create_vocabulary('corpus_v2', vocab_size => 200019);
ALTER TABLE documents ALTER COLUMN body TYPE tokens.corpus_v2 USING body::tokens.corpus_v2;
```

## Benchmarks

C4 English, 512-token chunks, o200k, PostgreSQL 14, one pinned core. The workload is an agent: it
holds token IDs and wants token IDs back, so the `text` column pays a detokenize on write and a
tokenize on every read.

**Read**, µs per query:

| column | fan-out 1 | fan-out 10 | fan-out 100 | wire @100 |
| --- | --: | --: | --: | --: |
| `text` (pglz) | 1176 | 4066 | 19258 | 233 KB |
| `text` (lz4) | 1191 | 3864 | 19421 | 233 KB |
| `pgtoken` raw | **748** (1.6×) | **903** (4.5×) | **2588** (7.4×) | 147 KB |
| `pgtoken` freq | 938 (1.3×) | 1888 (2.2×) | 10485 (1.8×) | **88 KB** |

At fan-out 100 the `text` column spends 16.5 ms of its 19.3 ms tokenizing. That is the cost
`pgtoken` removes, and it recurs on every read. Storage is ~2.1× smaller than `text` in payload,
relation size, and WAL.

The codec alone, measured in Rust with no database in the way
(`cd core && cargo run --release --example codec_bench`):

| codec | encode | decode | bytes/token |
| --- | --: | --: | --: |
| `raw16` | 0.57 µs | 0.26 µs | 2.02 |
| `raw24` | 1.11 µs | 0.42 µs | 3.02 |
| `freq` | 5.38 µs | **4.06 µs** | **1.89** |

So `freq` costs ~4 µs to decode, against ~250 µs to tokenize the equivalent text. The end-to-end
figures above understate it, because the Python client pays numpy overhead on 512-element arrays.

> The end-to-end numbers predate the type. `benchmarks/bench_readwrite.py` and `pgcommon.py`
> still call the removed `encode`/`decode` functions and need porting to vocabularies before they
> run again. Ratios were stable across runs; absolute microseconds are not — they inflate
> several-fold under load.

## Limitations

- **Reads give you token IDs, not text**, unless you load a mapping. `pgtoken.text(body)`
  detokenizes once `pgtoken.load_mapping` has loaded a `token_id -> bytes` mapping for the
  vocabulary, and because that mapping is write-once, the function is `IMMUTABLE` and can back a
  GIN index over `to_tsvector` — full-text search and `LIKE` both work through it.
- **No `=`, `ORDER BY`, `GROUP BY` or `DISTINCT`** on the column itself. Byte order of a
  compressed value is meaningless, and there is no equality operator yet.
- **A column must name a vocabulary.** A bare `pgtoken.tokens` column accepts inserts and then
  fails on read, because PostgreSQL applies a type modifier after the input function runs, so
  there is nowhere earlier to refuse. The rows are recoverable in place with
  `ALTER TABLE ... TYPE tokens.<name>`.
- **Binary writes are trusted.** Text and `int[]` input check every id against `vocab_size`;
  `COPY BINARY` and the `bytea` cast check only the 12-byte header, because scanning the payload
  would cost the write path the speed it exists for.
- **A vocabulary's name and id are reserved forever**, even after you drop its domain, since
  stored values reference the id.
- **Rankings are corpus-specific.** Ratios drop if the ranking and your data diverge.

`benchmarks/pgtoken_client.py` is a reference client-side codec in Python, byte-compatible with
the extension in both directions (`benchmarks/test_client.py` asserts it).

## Reference

| function | returns | |
| --- | --- | --- |
| `create_vocabulary(name, vocab_size [, compression] [, id])` | `int` | also creates `tokens.<name>` |
| `train(name, query [, max_ranks])` | `text` | ranking for `freq`, write-once |
| `load_mapping(name, query)` | `text` | `token_id -> bytes` mapping for `text`, write-once |
| `vocabulary_info(name)` | record | size, compression, width, ranking, sha256 |
| `drop_vocabulary(name)` | | drops the domain; the id stays reserved |
| `token_count(tokens)` | `int` | header only, no decode |
| `describe(tokens)` | record | codec, vocabulary, sizes |
| `text(tokens)` | `text` | detokenize; needs a mapping; `IMMUTABLE` |

Casts: `int[] → tokens` (assignment), and `tokens → int[]`, `tokens → bytea`, `bytea → tokens`
(explicit).

Setting: `pgtoken.table_dir`, where rankings live (`SIGHUP`). It is not session-settable on
purpose — two sessions must never decode one value differently.

## Tests

```sh
cd core && cargo test          # codecs, no PostgreSQL needed
cd ext  && cargo pgrx test pg14
```

Clear `$pgtoken.table_dir` between runs — rankings are files, they survive the rollback that
resets everything else, and `train` refuses to overwrite one.

CI runs both across PostgreSQL 14–18, plus `cargo fmt`, `clippy -D warnings`, and an install
check. `benchmarks/test_client.py` additionally asserts the Python client is byte-compatible with
the extension, but needs a running server so it is not part of CI.

## License

Apache-2.0
