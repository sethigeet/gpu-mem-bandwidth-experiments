#!/usr/bin/env bash
set -euo pipefail

# sync the local project to the remote host (also sets the HOST and REMOTE_DIR vars)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/sync_remote.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-results/attention_bw_ncu_${TIMESTAMP}}
if [[ $# -gt 0 ]]; then
  shift
fi

# run ncu remotely
ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_ncu.sh ${OUT}.csv $@"

# uncomment to copy if needed
# scp "$HOST:$REMOTE_DIR/${OUT}.csv" "${OUT}.csv"
# echo "copied remote results from $HOST:$REMOTE_DIR/${OUT}.csv to ${OUT}.csv"

# Generate visualization remotely and copy image
ssh "$HOST" "cd $REMOTE_DIR && uv run python main.py visualize ${OUT}.csv -o ${OUT}.png"
mkdir -p "$(dirname "$OUT")"
scp "$HOST:$REMOTE_DIR/${OUT}.png" "${OUT}.png"
echo "Copied visualization to ${OUT}.png"
