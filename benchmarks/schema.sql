-- Benchmark schema: the same corpus stored three ways, so the only difference between the
-- tables is how the text column is represented.
--
--   docs_text      text, default EXTENDED storage, i.e. pglz. The realistic baseline: what
--                  a Postgres column holding RAG chunks looks like today.
--   docs_text_lz4  text with COMPRESSION lz4. Matches the paper's LZ4 baseline and what
--                  Qdrant and Elasticsearch actually do.
--   docs_tnt       token-native bytea with STORAGE EXTERNAL, which tells Postgres not to
--                  compress a payload that is already entropy-coded.
--
-- Every table carries the same surrogate embedding column so the row shape, and therefore
-- the tuple size and TOAST decisions, stay comparable. The embedding is never measured.
-- pgvector is not installed here, so a fixed-length bytea stands in for a 1024-dim float32
-- vector (4096 bytes); the byte footprint is what matters for tuple layout, not the type.

CREATE EXTENSION IF NOT EXISTS pgtoken;
CREATE EXTENSION IF NOT EXISTS pg_buffercache;
CREATE EXTENSION IF NOT EXISTS pg_prewarm;

DROP TABLE IF EXISTS docs_text, docs_text_lz4, docs_tnt;

CREATE TABLE docs_text (
    id        integer PRIMARY KEY,
    embedding bytea,
    body      text NOT NULL
);

CREATE TABLE docs_text_lz4 (
    id        integer PRIMARY KEY,
    embedding bytea,
    -- COMPRESSION precedes the constraint in PostgreSQL's column syntax.
    body      text COMPRESSION lz4 NOT NULL
);

CREATE TABLE docs_tnt (
    id        integer PRIMARY KEY,
    embedding bytea,
    body      bytea NOT NULL
);

-- Do not let Postgres compress the already-compressed token payload. Running pglz over ANS
-- or streamvbyte output costs CPU for essentially no gain.
ALTER TABLE docs_tnt ALTER COLUMN body SET STORAGE EXTERNAL;

-- Keep the surrogate embedding out of the comparison: it is identical across tables, and
-- letting Postgres compress or TOAST it differently per table would confound the
-- measurement of the text column.
ALTER TABLE docs_text     ALTER COLUMN embedding SET STORAGE EXTERNAL;
ALTER TABLE docs_text_lz4 ALTER COLUMN embedding SET STORAGE EXTERNAL;
ALTER TABLE docs_tnt      ALTER COLUMN embedding SET STORAGE EXTERNAL;
