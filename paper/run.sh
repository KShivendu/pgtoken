#!/usr/bin/env bash
# Run a paper experiment inside the client container (docker compose sets PGTOKEN_DSN etc.).
#   bash paper/run.sh smoke    # check_o200k + rag_e2e at 20k rows, k=100, 1 and 4 clients
#   bash paper/run.sh wave1    # the full Wave 1 matrix (k = 10, 50, 100; 1, 2, 4, 8 clients)
# Extra arguments go to rag_e2e.py, e.g. `bash paper/run.sh wave1 --rows 2000000`.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-smoke}"; shift || true

case "$mode" in
  smoke)
    python paper/check_o200k.py --smoke --chunks 600
    python paper/rag_e2e.py --smoke --rows 20000 --k 100 --clients 1 4 --queries 300 "$@"
    ;;
  wave1)
    python paper/check_o200k.py --chunks 2000
    python paper/rag_e2e.py --rows 1000000 --k 10 50 100 --clients 1 2 4 8 --queries 2000 "$@"
    ;;
  *)
    echo "unknown mode $mode (smoke | wave1)" >&2; exit 2 ;;
esac
