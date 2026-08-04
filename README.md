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

| codec | size | read | needs a table |
| --- | --: | --- | --- |
| `raw` | 2–3 B/token | fastest | no |
| `freq` | ~1.3 B/token | fast | yes |

`raw` packs IDs at a fixed width, picking 2 or 3 bytes from the data. `freq` remaps them to
frequency rank and packs with a varint, so common tokens cost one byte.

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

C4 English, 512-token chunks, o200k, PostgreSQL 14. Agent reads against a `text` column:

| fan-out | speedup |
| --: | --: |
| 10 | 5.7–7.5× |
| 100 | 9.5–14.3× |

Spread across four runs at different machine loads. Roughly 90% of the `text` path is
tokenization in every one of them.

Storage is ~2.1× smaller in payload, total relation size, and WAL.

Reproducing needs the corpora from the
[token-storage](https://github.com/KShivendu/token-storage) repo:

```sh
TOKEN_STORAGE_REPO=/path/to/token-storage uv run python benchmarks/bench_latency.py
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
