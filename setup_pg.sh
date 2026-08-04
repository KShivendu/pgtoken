#!/usr/bin/env bash
# Set up a local PostgreSQL for building and testing pgtoken, without root.
#
# Ubuntu's postgresql .deb packages are downloaded with `apt-get download` (which needs no
# privileges) and extracted with `dpkg -x` into a prefix under ~/.local/share. The only
# wrinkle is that the extracted pg_config still reports its compiled-in /usr paths, so this
# installs a small shim that rewrites them to the prefix. pgrx and PGXS then find the
# headers, libs and binaries that are actually present.
#
# The alternative, `cargo pgrx init --pg18 download`, builds PostgreSQL from source and
# needs build dependencies this box lacks (libreadline-dev, libxslt1-dev, xsltproc), which
# cannot be installed without sudo. Hence this route.
#
# Caveat: this pins PostgreSQL 14, Ubuntu 22.04's default. 14 is new enough for everything
# the harness needs (per-column `COMPRESSION lz4` landed in 14, and pg_buffercache /
# pg_prewarm / pg_stat_statements all ship with it). Testing against 16/17/18 needs either
# the PGDG apt repository or a source build; see README.md.
#
# Usage:
#   bash 12_postgres/setup_pg.sh            # install and initialise
#   bash 12_postgres/setup_pg.sh --start    # start the server
#   bash 12_postgres/setup_pg.sh --stop     # stop it
#   bash 12_postgres/setup_pg.sh --psql     # open a shell against it

set -euo pipefail

PG_MAJOR=14
DEST="${PGTOKEN_PG_HOME:-$HOME/.local/share/pgtoken-pg}"
PGROOT="$DEST/pgroot"
PGBIN="$PGROOT/usr/lib/postgresql/$PG_MAJOR/bin"
PGSHARE="$PGROOT/usr/share/postgresql/$PG_MAJOR"
PGDATA="$DEST/data"
# The Unix socket path has a hard 107-byte limit, so keep the directory short rather than
# putting it next to the data directory.
SOCKDIR="${PGTOKEN_PG_SOCK:-/tmp/tnt-pg}"
PORT="${PGTOKEN_PG_PORT:-55432}"

export LD_LIBRARY_PATH="$PGROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

PACKAGES=(
  postgresql-"$PG_MAJOR"
  postgresql-client-"$PG_MAJOR"
  postgresql-server-dev-"$PG_MAJOR"
  postgresql-common
  postgresql-client-common
  libpq5
  libpq-dev
  libllvm14
)

start_server() {
  mkdir -p "$SOCKDIR"
  "$PGBIN/pg_ctl" -D "$PGDATA" -l "$DEST/server.log" \
    -o "-p $PORT -k $SOCKDIR -c listen_addresses=''" start
  echo "server up on $SOCKDIR:$PORT (log: $DEST/server.log)"
}

stop_server() {
  "$PGBIN/pg_ctl" -D "$PGDATA" stop -m fast || true
}

case "${1:-install}" in
  --start) start_server; exit 0 ;;
  --stop)  stop_server;  exit 0 ;;
  --psql)  exec "$PGBIN/psql" -h "$SOCKDIR" -p "$PORT" -U postgres "${@:2}" ;;
  --env)
    # Print the environment other scripts need, for `eval "$(setup_pg.sh --env)"`.
    # PATH and LD_LIBRARY_PATH are emitted in double quotes so the existing values expand
    # when the caller evals this; single quotes would replace rather than prepend.
    echo "export PGHOST='$SOCKDIR'"
    echo "export PGPORT='$PORT'"
    echo "export PGUSER=postgres"
    echo "export PATH=\"$PGBIN:\$PATH\""
    echo "export LD_LIBRARY_PATH=\"$PGROOT/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH\""
    exit 0 ;;
  install) ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac

echo "==> downloading PostgreSQL $PG_MAJOR packages (no root required)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
(cd "$WORK" && apt-get download "${PACKAGES[@]}")

echo "==> extracting into $PGROOT"
mkdir -p "$PGROOT"
for deb in "$WORK"/*.deb; do dpkg -x "$deb" "$PGROOT"; done

echo "==> installing pg_config shim"
mkdir -p "$DEST/bin"
cat > "$DEST/bin/pg_config" <<EOF
#!/bin/sh
# Rewrites the extracted pg_config's compiled-in /usr paths to the local prefix.
exec "$PGBIN/pg_config" "\$@" | sed \\
  -e "s#^/usr#$PGROOT/usr#g" \\
  -e "s# /usr# $PGROOT/usr#g" \\
  -e "s#-I/usr#-I$PGROOT/usr#g" \\
  -e "s#-L/usr#-L$PGROOT/usr#g"
EOF
chmod +x "$DEST/bin/pg_config"
"$DEST/bin/pg_config" --version

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "==> initdb at $PGDATA"
  "$PGBIN/initdb" -D "$PGDATA" -L "$PGSHARE" -U postgres --no-locale --encoding=UTF8
else
  echo "==> reusing existing cluster at $PGDATA"
fi

echo "==> registering with pgrx"
command -v cargo-pgrx >/dev/null 2>&1 || cargo install cargo-pgrx --locked
cargo pgrx init --pg"$PG_MAJOR" "$DEST/bin/pg_config"

cat <<EOF

Done. PostgreSQL $PG_MAJOR is installed at $DEST and registered with pgrx.

  start:  bash 12_postgres/setup_pg.sh --start
  psql:   bash 12_postgres/setup_pg.sh --psql
  env:    eval "\$(bash 12_postgres/setup_pg.sh --env)"
EOF
