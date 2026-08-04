# pgtoken

Store text in PostgreSQL as BPE token IDs instead of UTF-8.

Agents read and write token IDs, not characters. A `text` column makes them re-tokenize on
every read; `pgtoken` hands over the IDs directly, in less than half the space.

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

INSERT INTO documents (body) VALUES (pgtoken.encode('hello world'));

SELECT pgtoken.decode(body) FROM documents;   -- text, for a human
SELECT body FROM documents;                   -- blob, decoded client-side by an agent
```

`STORAGE EXTERNAL` stops PostgreSQL compressing a payload that is already compressed.

When the writer is a model that already holds token IDs, no tokenizer runs at all:

```sql
INSERT INTO documents (body)
VALUES (pgtoken.encode_ids('{24912,2375}', 'o200k', 'raw', 0));
```

## Codecs

| codec | ratio | read speed | needs a trained table |
| --- | --: | --- | --- |
| `raw` | 1.6× | fastest | no |
| `freq` | 2.7× | fast | yes |
| `ans` | 3.4× | slower | yes |

`freq` is the recommended default. Train a table over your own corpus:

```sql
SELECT pgtoken.train_freq_table(1, 'o200k', 'SELECT body FROM my_corpus');
INSERT INTO documents (body)
VALUES (pgtoken.encode('hello world', 'o200k', 'freq', 1));
```

Table ids are permanent, since stored values reference them.

Tokenizers: `r50k`, `cl100k`, `o200k`.

## Benchmarks

C4 English, 512-token chunks, o200k, PostgreSQL 14.

Agent reads, against a `text` column:

| fan-out | `raw` | `ans` |
| --: | --: | --: |
| 10 | 7.1–7.5× faster | 2.9–3.0× |
| 100 | 11.6–14.3× faster | 3.6× |

92% of the `text` path is tokenization. Storage is 2.1× smaller in payload, total relation
size, and WAL.

To reproduce, you need the held-out corpora from the
[token-storage](https://github.com/KShivendu/token-storage) repo:

```sh
TOKEN_STORAGE_REPO=/path/to/token-storage \
  uv run python benchmarks/bench_latency.py
```

Ratios are stable across runs; absolute microseconds are not — they inflate several-fold
under load, so check `/proc/loadavg` before quoting one.

## Limitations

- **One tokenizer per value.** IDs only mean something to a consumer using the same
  tokenizer. Values record their own, so a column can hold several during a migration.
- **No `ORDER BY`, `LIKE` or `pg_trgm`** on the column — byte order of a compressed value is
  meaningless. For full-text search, index an expression:
  `CREATE INDEX ON documents USING gin (to_tsvector('english', pgtoken.decode(body)));`
- `=`, `GROUP BY`, `DISTINCT` and hash joins do work, on shorter keys than `text`.
- **Coding tables are corpus-specific.** Ratios drop if the table and your data diverge.

## Reference

| function | returns | |
| --- | --- | --- |
| `encode(text)` | `bytea` | uses the `pgtoken.*` defaults |
| `encode(text, tokenizer, codec, table_id)` | `bytea` | pinned, `IMMUTABLE` |
| `encode_ids(int[], tokenizer, codec, table_id)` | `bytea` | from a model's own IDs |
| `decode(bytea)` | `text` | back to text |
| `token_ids(bytea)` | `int[]` | token IDs |
| `token_count(bytea)` | `int` | header only, no decode |
| `describe(bytea)` | record | tokenizer, codec, table, sizes |
| `recode(bytea, codec, table_id)` | `bytea` | change codec, keeping the IDs |
| `train_freq_table(id, tokenizer, query)` | `text` | |
| `train_ans_table(id, tokenizer, query)` | `text` | |
| `table_info(id)` | record | kind, tokenizer, vocab, sha256 |

Settings: `pgtoken.table_dir` (where coding tables live, `SIGHUP`),
`pgtoken.default_tokenizer`, `pgtoken.default_codec`, `pgtoken.default_table_id`.

## Tests

```sh
cd core && cargo test          # codecs, no PostgreSQL needed
cd ext  && cargo pgrx test pg14
```

## License

Apache-2.0
