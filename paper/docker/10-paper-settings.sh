#!/bin/bash
# Runs once, at initdb time, inside the official postgres entrypoint.
set -e
echo "include '/etc/postgresql/postgresql.paper.conf'" >> "$PGDATA/postgresql.conf"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgtoken;
SQL
