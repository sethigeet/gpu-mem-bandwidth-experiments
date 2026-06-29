#!/usr/bin/env bash
set -euo pipefail

# Sync, profile a single prefix-sharing config under nsys remotely, render the DRAM
# bandwidth timeline, and copy the plot back.
# Usage: run_prefix_nsys_remote.sh EXPERIMENT [extra prefix_main.py args...]
#   e.g. run_prefix_nsys_remote.sh homogeneity --values 1.0   (fully homogeneous)
#        run_prefix_nsys_remote.sh homogeneity --values 0.5   (mixed prefixes)

EXPERIMENT=${1:?usage: run_prefix_nsys_remote.sh EXPERIMENT [args...]}
shift

EXTRA_ARGS=$(printf '%q ' "$@")
set --

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/sync_remote.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-results/prefix_bw_nsys_${EXPERIMENT}_${TIMESTAMP}}

ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_prefix_nsys.sh '$OUT' '$EXPERIMENT' $EXTRA_ARGS"
ssh "$HOST" "cd $REMOTE_DIR && uv run python prefix_main.py visualize ${OUT}.sqlite -o ${OUT}.png"

mkdir -p "$(dirname "$OUT")"
scp "$HOST:$REMOTE_DIR/${OUT}.png" "${OUT}.png"
echo "Copied visualization to ${OUT}.png"
