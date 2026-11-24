#!/usr/bin/env bash
# The Wave 1 campaign on the dedicated box, in decision order. Runs on the host (not in a
# container); each step is one rag_e2e / consumer_share / lz4_threshold run with its own run dir.
#   bash paper/bench_campaign.sh dbpedia-warm | dbpedia-cold | consumer | c4-load | c4-warm | c4-cold | lz4 | check | storage
set -euo pipefail
cd "$(dirname "$0")"
DC="sudo -E docker compose -f docker-compose.yml -f docker-compose.bench.yml"
run() { $DC run --rm --no-deps client "$@"; }
server() {  # (re)start the server with a given shared_buffers and wait until it answers
  PG_SHARED_BUFFERS=$1 $DC up -d --force-recreate server
  until sudo docker exec pgtoken-paper-server-1 psql -h 127.0.0.1 -U postgres -tAc 'select 1' >/dev/null 2>&1; do sleep 2; done
  sudo docker exec pgtoken-paper-server-1 psql -h 127.0.0.1 -U postgres -tAc 'show shared_buffers'
}
DBP="--data /data/dbpedia --rows 999000 --layout split --tag dbp"
C4="--data /data/c4 --rows 1000000 --layout split --tag c4"
case "$1" in
  dbpedia-warm) server 24GB; run python paper/rag_e2e.py $DBP --name dbpedia-warm \
                  --k 100 10 50 --clients 1 4 8 16 --queries 600 --warmup 20 ;;
  dbpedia-cold) server 128MB; run python paper/rag_e2e.py $DBP --name dbpedia-cold --cold \
                  --k 100 --clients 1 16 --queries 100 --warmup 3 --sweep auto; server 24GB ;;
  dbpedia-cold16) server 128MB; run python paper/rag_e2e.py $DBP --name dbpedia-cold16 --cold \
                  --k 100 --clients 16 --queries 160 --warmup 3 --sweep auto; server 24GB ;;
  consumer)     sudo docker run --rm --network pgtoken-paper_default --cpuset-cpus 8-15 \
                  -v /data:/data -v "$PWD":/pgtoken/paper -v "$PWD/../benchmarks":/pgtoken/benchmarks \
                  -v "$(realpath ../../token-storage)":/token-storage:ro \
                  -e PGTOKEN_HOST_NAME="${PGTOKEN_HOST_NAME:-pgtoken-bench}" \
                  pgtoken-paper-client-ml:0.1.0 python paper/consumer_share.py ;;
  c4-load)      server 24GB; run python paper/rag_e2e.py $C4 --name c4-load --load-only ;;
  c4-warm)      server 24GB; run python paper/rag_e2e.py $C4 --name c4-warm \
                  --k 100 10 50 --clients 1 4 8 16 --queries 600 --warmup 20 ;;
  c4-cold)      server 128MB; run python paper/rag_e2e.py $C4 --name c4-cold --cold \
                  --k 100 --clients 1 --queries 100 --warmup 3 --sweep auto; server 24GB ;;
  lz4)          run python paper/lz4_threshold.py --data /data/dbpedia /data/c4 --sample 100000 ;;
  check)        run python paper/check_o200k.py --chunks 2000 ;;
  storage)      run python paper/storage.py --rows 1000 ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
