# pgtoken-rs

Python client for [pgtoken](../README.md). It reads and writes `tokens.<vocabulary>` columns as
numpy arrays of token IDs. Values are encoded and decoded on your machine, with the same Rust codec
the extension uses, so the server does no work per token.

## Install

Not on PyPI yet. Install from a checkout. You need Rust, because this builds a small native module.

```sh
uv pip install "./py[psycopg]"        # or: pip install "./py[psycopg]"
```

The `psycopg` extra pulls in psycopg 3. Leave it out if you only want the codec functions.

## Use with psycopg

```python
import psycopg
from pgtoken_rs.psycopg import register

conn = psycopg.connect("postgresql://localhost/mydb", autocommit=True)
vocab = register(conn, "o200k")          # once per vocabulary, per connection

conn.execute("INSERT INTO documents (body) VALUES (%s)", [vocab.encode([24912, 2375])])

ids = conn.execute("SELECT body FROM documents", binary=True).fetchone()[0]
print(ids)                               # [24912  2375], a numpy uint32 array
```

`register` runs one query, `pgtoken.vocabulary_codec('o200k')`. It returns the vocabulary's id, its
codec, and for a `freq` vocabulary the trained ranking as bytes. That works over any connection,
including a hosted PostgreSQL whose disk you cannot see.

After `register`, the connection knows three things:

- `vocab.encode(ids)` takes a list or numpy array and returns an encoded value. Pass it as a
  query parameter and it goes over the wire as compact bytes.
- Any pgtoken column reads back as a numpy `uint32` array. Use `binary=True` for speed. Text results
  work too, at the cost of parsing `{24912,2375}`.
- Binary `COPY` takes plain ID lists or arrays once you name the column type:

```python
docs = [[24912, 2375], [9906, 11, 1917]]   # token ids from your tokenizer

with conn.cursor() as cur, cur.copy("COPY documents (id, body) FROM STDIN (FORMAT BINARY)") as cp:
    cp.set_types(["int8", vocab.type_name])   # vocab.type_name is 'tokens.o200k'
    for i, ids in enumerate(docs, start=1):
        cp.write_row((i, ids))
```

`COPY ... TO STDOUT (FORMAT BINARY)` with the same `set_types` call reads arrays back.

You can register several vocabularies on one connection. Each value names its own vocabulary in
its header, so reads pick the right ranking by themselves.

## What it checks

`vocab.encode` rejects an ID below 0 or at or above the vocabulary's size, and anything that is not
a 1-D sequence of integers. This matters for `COPY BINARY`, because the server checks only the
header of a binary value.

The server still checks that a value belongs to the column's vocabulary. Inserting an `o200k` value
into a `tokens.cl100k` column fails.

Reading a `freq` value whose vocabulary was not registered on that connection raises a
`ValueError` that names the missing vocabulary id. `raw` values need no ranking, so they always
decode.

## Codec functions

The psycopg layer is built on three functions, which you can call yourself:

```python
import pgtoken_rs

table = pgtoken_rs.Table.from_bytes(ranking)              # freq only
value = pgtoken_rs.encode(ids, "freq", vocabulary_id, table)
ids = pgtoken_rs.decode(value, table)                     # numpy uint32 array
```

The codec is `raw8`, `raw16`, `raw24` or `freq`. `pgtoken.vocabulary_codec(name)` tells you which
one a vocabulary uses.

## Tests

```sh
uv run python py/tests/test_roundtrip.py   # codec only, no PostgreSQL
uv run python py/tests/test_psycopg.py     # needs a running server with pgtoken installed
```

`test_psycopg.py` works in its own database, `pgtoken_pytest`, which it creates and drops. It
connects with the usual `PG*` variables, or to the local server from `setup_pg.sh`.

## Building wheels

```sh
cd py && uvx maturin build --release      # wheel lands in py/target/wheels
```
