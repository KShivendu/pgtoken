# Paper benchmarks

Scripts behind the tables and figures of the pgtoken paper. Every script writes one run
directory, `paper/runs/<date>-<name>/` (smoke runs go under `paper/runs/smoke/`), holding the
config, PostgreSQL settings, git SHA and machine, the summary JSON and a per-query JSONL.

| script | what it measures |
| --- | --- |
| `check_o200k.py` | which o200k tokenizer the text arms use: ID-for-ID equality with tiktoken `o200k_base` on real chunks (English, code, Hindi), and ns per token, cold and warm |
| `rag_e2e.py` | Wave 1: pgvector HNSW top-k, fetch the k chunk bodies, produce token IDs as numpy arrays in the client, for every storage format |

## Setup

Two Docker images, built by `docker compose`:

- `pgtoken-paper-server:0.1.1`: `pgvector/pgvector:pg16` plus the released
  `postgresql-16-pgtoken_0.1.1_amd64.deb` (checksum pinned), with the settings in
  `docker/postgresql.paper.conf` applied identically to every arm.
- `pgtoken-paper-client:0.1.0`: Python 3.12, uv, Rust 1.88 (to build `pgtoken_rs` from `py/`),
  tiktoken and HF `tokenizers`, with the tokenizer files baked in at pinned revisions.

The corpora come from the [token-storage](https://github.com/KShivendu/token-storage) repo
(C4 English, codeparrot, Hindi Wikipedia, stored as r50k ids; the scripts re-tokenize with
o200k). By default compose mounts `../../token-storage`, i.e. a checkout next to this repo; set
`TOKEN_STORAGE_REPO` to point elsewhere. Only `tnbench.py` and `data/corpus/*_test.npy` are read.

## Run

```sh
cd paper
# smoke: tokenizer check, then rag_e2e at 20k rows, k=100, 1 and 4 clients, 300 queries per arm
PGTOKEN_GIT_SHA=$(git rev-parse --short HEAD) docker compose up --build --abort-on-container-exit
# the full Wave 1 matrix
RUN_MODE=wave1 docker compose up --build --abort-on-container-exit
# one script with your own flags
docker compose run --rm client python paper/rag_e2e.py --rows 50000 --k 10 100 --clients 1 2 4
# clean up the server and its data volume (images are kept)
docker compose down -v
```

Without Docker: start any PostgreSQL 16 with pgvector and pgtoken, then
`PGTOKEN_DSN='host=... user=...' TOKEN_STORAGE_REPO=... uv run python paper/rag_e2e.py ...`.

Tables are reused across runs when rows, seed and corpus match (`rag_meta`); pass `--no-reuse` to
reload. Files written into `paper/runs/` from the container are owned by root.

## rag_e2e.py in one paragraph

One table per arm, `(id int PRIMARY KEY, emb vector(384), body <type>)`, with an HNSW index on
`emb`. Every table holds the same rows: 512-token o200k chunks of C4 English (cycled if `--rows`
exceeds the 2,795 unique chunks, with a warning) and the same random unit embeddings. Arms:
text (pglz), text (lz4), text (lz4) tokenized as one batch on all cores, `int4[]`, `bytea`
raw24, pgtoken raw (raw24), pgtoken freq (freq, or freq_lz4 when lz4 makes a value smaller), and
an oracle whose IDs are already in client memory. A query is
`SELECT id, body FROM rag_<arm> ORDER BY emb <=> $1 LIMIT k` (prepared, binary results); the
client then turns the k bodies into k numpy uint32 arrays. Reported per arm: end-to-end p50, p90
and p99, the fetch and decode stages (fetch alone is the floor for any token-ID format: bytes in
hand, no decode), the ANN-only query, stored and wire bytes per row, TOAST counts, the paired
per-query speedup against text (lz4) and bytea raw24, and at more than one client the throughput
and Little's-law ratio. Every fetched row is checked ID for ID.
