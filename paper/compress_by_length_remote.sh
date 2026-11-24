#!/bin/bash
# Run paper/compress_by_length.py on ssh tpuf-bench (no Rust there, so pgtoken_rs ships as a
# wheel built on the laptop). Run from the laptop, in the pgtoken repo root:
#
#   bash paper/compress_by_length_remote.sh setup
#   bash paper/compress_by_length_remote.sh run wildchat --datasets wildchat --tokenizers o200k
#   bash paper/compress_by_length_remote.sh run rest --resume <remote run dir> \
#        --datasets wildchat,dbpedia,c4,code --tokenizers o200k,qwen3
#   bash paper/compress_by_length_remote.sh fetch <run dir name>
#
# Each run goes into its own tmux session (cl-<name>), pinned to core 2, logs in ~/pgtoken-cl/.
set -euo pipefail
HOST=tpuf-bench
case "$1" in
setup)
  W=$(mktemp -d)
  uv build --wheel py --python 3.12 --out-dir "$W"
  ssh $HOST 'mkdir -p ~/pgtoken-cl/pgtoken ~/pgtoken-cl/token-storage/data/corpus ~/pgtoken-cl/wheel'
  rsync -az --include='*.py' --include='*.cpp' --include='benchmarks/' --include='paper/' --exclude='*' ./ $HOST:pgtoken-cl/pgtoken/
  rsync -az "${TOKEN_STORAGE_REPO:-$HOME/projects/token-storage}"/tnbench.py $HOST:pgtoken-cl/token-storage/
  rsync -az "${TOKEN_STORAGE_REPO:-$HOME/projects/token-storage}"/data/corpus/code_t*.npy $HOST:pgtoken-cl/token-storage/data/corpus/
  rsync -az "$W"/*.whl $HOST:pgtoken-cl/wheel/
  ssh $HOST 'cd ~/pgtoken-cl && ~/.local/bin/uv venv -q --python 3.12 .venv &&
    VIRTUAL_ENV=.venv ~/.local/bin/uv pip install -q wheel/*.whl numpy tiktoken "tokenizers==1.0.0rc2" zstandard lz4 pyarrow huggingface_hub "psycopg[binary]"'
  ;;
run)
  shift; name=$1; shift
  sha=$(git rev-parse --short HEAD)$( [ -n "$(git status --porcelain)" ] && echo -dirty)
  ssh $HOST "tmux new -d -s cl-$name 'cd ~/pgtoken-cl/pgtoken && TOKEN_STORAGE_REPO=~/pgtoken-cl/token-storage PGTOKEN_GIT_SHA=$sha RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 /usr/bin/time -v taskset -c 2 ~/pgtoken-cl/.venv/bin/python paper/compress_by_length.py $* > ~/pgtoken-cl/$name.log 2> ~/pgtoken-cl/$name.err'"
  ;;
fetch)
  rsync -az --partial $HOST:pgtoken-cl/pgtoken/paper/runs/"$2"/ paper/runs/"$2"/
  uv run --with pyarrow --with lz4 python paper/compress_by_length.py --summarize paper/runs/"$2"
  ;;
esac
