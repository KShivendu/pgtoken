#!/usr/bin/env bash
# pg_dump a database with tokens columns, restore it into fresh databases, and check every value
# comes back. Covers a raw vocabulary, a trained freq vocabulary with a loaded mapping, and the
# bundled o200k preset; a column in `public` and one in schema `aaa`, which sorts before
# `pgtoken` (so its rows load before the vocabulary catalog's rows); and a view with a cast to a
# vocabulary type. Restores twice: a plain SQL dump through psql, and a custom-format dump through
# `pg_restore -j 4`, which loads table data in parallel with no order between tables.
#
# Trained rankings and mappings live in the database as functions (see ext/src/registry.rs), so
# they have to come through the dump too, and land before any table data needs them. Last, a
# second database reuses the same vocabulary ids with different artefacts, to show ids and
# artefacts are per database.
#
# Needs a running server with pgtoken installed, e.g. the local one:
#     eval "$(bash setup_pg.sh --env)" && bash scripts/test_dump_restore.sh
set -euo pipefail

SRC=pgtoken_dump_src
DST=pgtoken_dump_dst
DSTJ=pgtoken_dump_dst_j
OTHER=pgtoken_dump_other
WORK="$(mktemp -d)"
q() { psql -X -q -v ON_ERROR_STOP=1 -At "$@"; }

for db in "$SRC" "$DST" "$DSTJ" "$OTHER"; do q -d postgres -c "DROP DATABASE IF EXISTS $db"; done
q -d postgres -c "CREATE DATABASE $SRC"

q -d "$SRC" <<'SQL'
CREATE EXTENSION pgtoken;
SELECT pgtoken.create_vocabulary('dr_raw', 50257);
SELECT pgtoken.create_vocabulary('dr_freq', 50257, compression => 'freq');
SELECT pgtoken.train('dr_freq', $$SELECT ARRAY[464, 262, 262, 286, 11, 13, 13, 13]$$);
SELECT pgtoken.load_mapping('dr_freq',
    $$SELECT * FROM (VALUES (464, 'The'::bytea), (262, ' the'), (286, ' of'), (11, ','),
                            (13, '.'), (0, '!'), (50256, '<|endoftext|>')) m$$);
SELECT pgtoken.create_vocabulary('o200k');

CREATE TABLE public.docs (id int PRIMARY KEY, a tokens.dr_raw, b tokens.dr_freq, c tokens.o200k);
CREATE SCHEMA aaa;
CREATE TABLE aaa.docs (LIKE public.docs);
INSERT INTO public.docs VALUES
    (1, ARRAY[464, 262, 50256]::int[]::tokens.dr_raw, ARRAY[464, 262, 50256]::int[]::tokens.dr_freq,
        '{13225,11,2375,0}'),
    (2, ARRAY[]::int[]::tokens.dr_raw,               ARRAY[13, 13, 0]::int[]::tokens.dr_freq,
        '{}'),
    (3, NULL, NULL, NULL);
INSERT INTO aaa.docs SELECT * FROM public.docs;
CREATE VIEW public.v AS SELECT id, ARRAY[1, 2]::int[]::tokens.dr_freq AS c FROM public.docs;
SQL

snapshot() {
  q -d "$1" -c "SELECT 'public', id, a::int[], b::int[], c::int[], pgtoken.text(b), pgtoken.text(c)
                FROM public.docs
                UNION ALL SELECT 'aaa', id, a::int[], b::int[], c::int[], pgtoken.text(b),
                                 pgtoken.text(c) FROM aaa.docs
                UNION ALL SELECT 'view', id, NULL, c::int[], NULL, NULL, NULL FROM public.v
                ORDER BY 1, 2"
  q -d "$1" -c "SELECT v.name, i.id, i.compression, i.preset, i.rank_sha256, i.map_sha256
                FROM pgtoken.vocabulary v, pgtoken.vocabulary_info(v.name) i ORDER BY i.id"
}

check() {  # check <restored db> <how>
  # Into files first: an error inside a process substitution would not stop the script.
  snapshot "$SRC" >"$WORK/src.txt"
  snapshot "$1" >"$WORK/dst.txt"
  if diff "$WORK/src.txt" "$WORK/dst.txt"; then
    echo "ok ($2): $(wc -l <"$WORK/dst.txt") rows and vocabularies restored identically"
  else
    echo "FAIL ($2): restored rows differ" >&2
    exit 1
  fi
}

pg_dump -d "$SRC" -f "$WORK/dump.sql"
echo "domains and artefact functions in the dump:"
grep -n -E "CREATE DOMAIN|CREATE FUNCTION pgtoken\.artefact_" "$WORK/dump.sql" | cut -c1-110 | sed 's/^/  /'

q -d postgres -c "CREATE DATABASE $DST"
q -d "$DST" -f "$WORK/dump.sql" >/dev/null
check "$DST" "pg_dump | psql"

pg_dump -Fc -d "$SRC" -f "$WORK/dump.custom"
q -d postgres -c "CREATE DATABASE $DSTJ"
pg_restore -j 4 --exit-on-error -d "$DSTJ" "$WORK/dump.custom"
check "$DSTJ" "pg_dump -Fc | pg_restore -j 4"

# Same ids, different database, different artefacts: neither sees the other's.
q -d postgres -c "CREATE DATABASE $OTHER"
q -d "$OTHER" <<'SQL' >/dev/null
CREATE EXTENSION pgtoken;
SELECT pgtoken.create_vocabulary('dr_raw', 50257);
SELECT pgtoken.create_vocabulary('dr_freq', 50257, compression => 'freq');
SELECT pgtoken.train('dr_freq', $$SELECT ARRAY[7, 7, 7, 8]$$);
SQL
read -r src_id src_sha <<<"$(q -d "$SRC" -F ' ' -c "SELECT id, rank_sha256 FROM pgtoken.vocabulary_info('dr_freq')")"
read -r oth_id oth_sha <<<"$(q -d "$OTHER" -F ' ' -c "SELECT id, rank_sha256 FROM pgtoken.vocabulary_info('dr_freq')")"
oth_map="$(q -d "$OTHER" -c "SELECT coalesce(map_sha256, 'none') FROM pgtoken.vocabulary_info('dr_freq')")"
if [ "$src_id" = "$oth_id" ] && [ "$src_sha" != "$oth_sha" ] && [ "$oth_map" = none ]; then
  echo "ok (per database): vocabulary id $src_id has a different ranking in each database, and no mapping in $OTHER"
else
  echo "FAIL (per database): $src_id/$src_sha vs $oth_id/$oth_sha, mapping $oth_map" >&2
  exit 1
fi

for db in "$SRC" "$DST" "$DSTJ" "$OTHER"; do q -d postgres -c "DROP DATABASE $db"; done
/bin/rm -rf "$WORK"
