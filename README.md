# pgtoken

Store text in PostgreSQL as token IDs instead of UTF-8.

Agents read and write token IDs, not characters. A `text` column makes them re-tokenize on
every read; `pgtoken` stores the IDs directly, compressed, and hands them back as-is.

**No tokenizer.** The extension takes `int[]` and returns `int[]`. You tokenize with whatever
you already use — tiktoken, HuggingFace, SentencePiece, your own — and the database stays out
of it. So any tokenizer works, nothing needs a vocabulary size, and the server never spends a
cycle tokenizing.

<sub>Background: [blog](https://www.kshivendu.dev/blog/token-storage) ·
[paper](https://arxiv.org/abs/2608.02376)</sub>

> Early days. Tested on PostgreSQL 14 only, and the API may still change.

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

CREATE TABLE documents (id bigserial PRIMARY KEY, body bytea);
ALTER TABLE documents ALTER COLUMN body SET STORAGE EXTERNAL;

-- token IDs from your tokenizer, client-side
INSERT INTO documents (body) VALUES (pgtoken.encode('{24912,2375}'));

SELECT pgtoken.decode(body) FROM documents;   -- {24912,2375}
SELECT body FROM documents;                   -- the blob, to decode client-side
```

`STORAGE EXTERNAL` stops PostgreSQL compressing a payload that is already compressed.

On the agent path, select `body` and decode client-side. `pgtoken.decode` is for SQL-side
work — `int[]` costs 4 bytes per token on the wire, more than the blob it came from.

## Codecs

| codec | size | decode | needs a table |
| --- | --: | --: | --- |
| `raw` | 2–3 B/token | 0.3–0.4 µs | no |
| `freq` | ~1.9 B/token | 4 µs | yes |

Per 512-token chunk, on a Zipf-like stream over a 200k vocabulary. `raw` packs IDs at a fixed
width, picking 2 or 3 bytes from the data. `freq` remaps them to frequency rank and packs with a
varint, so common tokens cost one byte.

Train a table from any query returning `int[]`:

```sql
SELECT pgtoken.train(1, 'SELECT ids FROM my_corpus');
INSERT INTO documents (body) VALUES (pgtoken.encode('{24912,2375}', 'freq', 1));
```

The table stores only the tokens your corpus actually contained — for one skewed corpus that
is 28 bytes, not the 800 KB a full vocabulary would need. Tokens it never saw still encode
losslessly, just a little wider, which is why nothing has to declare a vocabulary size.

Table ids are permanent, since stored values reference them. `pgtoken.recode` changes a value's
codec without leaving the token-ID domain.

## Benchmarks

C4 English, 512-token chunks, o200k, PostgreSQL 14, one pinned core. The workload is an agent:
it holds token IDs and wants token IDs back, so the `text` column pays a detokenize on write
and a tokenize on every read.

**Write**, µs per row, batches of 50:

| column | total | = client | + insert | bytes/row | vs text |
| --- | --: | --: | --: | --: | --: |
| `text` (pglz) | 190 | 78 | 113 | 2322 | 1.00× |
| `text` (lz4) | 157 | 81 | 75 | 2322 | 1.21× |
| `pgtoken` raw | **71** | 24 | 48 | 1473 | **2.68×** |
| `pgtoken` freq | 165 | 115 | 50 | **884** | 1.16× |

**Read**, µs per query:

| column | fan-out 1 | fan-out 10 | fan-out 100 | wire @100 |
| --- | --: | --: | --: | --: |
| `text` (pglz) | 1176 | 4066 | 19258 | 233 KB |
| `text` (lz4) | 1191 | 3864 | 19421 | 233 KB |
| `pgtoken` raw | **748** (1.6×) | **903** (4.5×) | **2588** (7.4×) | 147 KB |
| `pgtoken` freq | 938 (1.3×) | 1888 (2.2×) | 10485 (1.8×) | **88 KB** |

At fan-out 100 the `text` column spends 16.5 ms of its 19.3 ms tokenizing. That is the cost
`pgtoken` removes, and it recurs on every read.

**`freq` is understated above.** Those figures use the Python client in `benchmarks/`, which
pays numpy overhead on 512-element arrays. The codec itself, measured in Rust with no database
in the way (`cd core && cargo run --release --example codec_bench`):

| codec | encode | decode | bytes/token |
| --- | --: | --: | --: |
| `raw16` | 0.57 µs | 0.26 µs | 2.02 |
| `raw24` | 1.11 µs | 0.42 µs | 3.02 |
| `freq` | 5.38 µs | **4.06 µs** | **1.89** |

Per 512-token chunk. So `freq` really costs ~4 µs to decode, not the ~90 µs the Python client
shows — against ~250 µs to tokenize the equivalent text. A Rust client gets `freq`'s size with
roughly `raw`'s speed.

Storage is ~2.1× smaller than `text` in payload, total relation size, and WAL
(`bench_storage_wal.py`).

Reproducing the end-to-end numbers needs the corpora from the
[token-storage](https://github.com/KShivendu/token-storage) repo:

```sh
TOKEN_STORAGE_REPO=/path/to/token-storage \
  taskset -c 4 uv run python benchmarks/bench_readwrite.py
```

Ratios are stable across runs; absolute microseconds are not — they inflate several-fold under
load, so check `/proc/loadavg` before quoting one.

## Limitations

- **`decode` returns `int[]`, not text.** Detokenizing is yours to do. `psql` shows integers,
  and there is no full-text search over the column — keep a separate `text` or `tsvector`
  column if you need it.
- **No `ORDER BY`, `LIKE` or `pg_trgm`** on the column: byte order of a compressed value is
  meaningless. `=`, `GROUP BY`, `DISTINCT` and hash joins do work, on shorter keys than
  `int[]`.
- **IDs are only meaningful to the tokenizer that produced them.** The extension cannot check
  that for you, so a column mixing tokenizers is your problem to avoid.
- **Coding tables are corpus-specific.** Ratios drop if the table and your data diverge.

`benchmarks/pgtoken_client.py` is a reference client-side codec in Python, byte-compatible with
the extension in both directions (`benchmarks/test_client.py` asserts it).

## Reference

| function | returns | |
| --- | --- | --- |
| `encode(int[])` | `bytea` | uses the `pgtoken.*` defaults |
| `encode(int[], codec, table_id)` | `bytea` | pinned, `IMMUTABLE` |
| `decode(bytea)` | `int[]` | token IDs |
| `token_count(bytea)` | `int` | header only, no decode |
| `describe(bytea)` | record | codec, table, sizes |
| `recode(bytea, codec, table_id)` | `bytea` | change codec, keeping the IDs |
| `train(table_id, query)` | `text` | train from a query returning `int[]` |
| `train(table_id, query, max_ranks)` | `text` | as above, capping table size |
| `table_info(table_id)` | record | ranked tokens, sha256, file size |

Settings: `pgtoken.table_dir` (where coding tables live, `SIGHUP`), `pgtoken.default_codec`,
`pgtoken.default_table_id`.

## Tests

```sh
cd core && cargo test          # codecs, no PostgreSQL needed
cd ext  && cargo pgrx test pg14
```

## License

Apache-2.0
