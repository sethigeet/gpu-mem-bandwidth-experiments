#!/usr/bin/env bash
set -euo pipefail

# Sync the repo, run a prefix-sharing sweep remotely, and copy back the CSV + plot.
# Usage: run_prefix_remote.sh EXPERIMENT [extra prefix_main.py args...]
#   EXPERIMENT: homogeneity | prefix-length | num-groups | batch-size
# Host / remote dir come from scripts/sync_remote.sh defaults.

EXPERIMENT=${1:?usage: run_prefix_remote.sh EXPERIMENT [args...]}
shift

# Save remaining args before sourcing sync_remote.sh (which uses $1/$2 for HOST/REMOTE_DIR).
EXTRA_ARGS=
if (($# > 0)); then
  EXTRA_ARGS=$(printf '%q ' "$@")
fi
set --

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/sync_remote.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-results/prefix_bw_${EXPERIMENT}_${TIMESTAMP}}

ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_prefix.sh '$EXPERIMENT' '$OUT' $EXTRA_ARGS"

mkdir -p "$(dirname "$OUT")"
scp "$HOST:$REMOTE_DIR/${OUT}.csv" "${OUT}.csv"
scp "$HOST:$REMOTE_DIR/${OUT}.png" "${OUT}.png"
echo "Copied ${OUT}.csv and ${OUT}.png"
