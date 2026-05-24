#!/usr/bin/env bash
set -euo pipefail

# Save args before sourcing sync_remote.sh (which uses $1/$2 for HOST/REMOTE_DIR)
EXTRA_ARGS=$(printf '%q ' "$@")
set --

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/sync_remote.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-results/llm_bw_ncu_${TIMESTAMP}}

ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_llm_ncu.sh '${OUT}.csv' $EXTRA_ARGS"

ssh "$HOST" "cd $REMOTE_DIR && uv run python llm_main.py visualize ${OUT}.csv -o ${OUT}.png $EXTRA_ARGS"
mkdir -p "$(dirname "$OUT")"
scp "$HOST:$REMOTE_DIR/${OUT}.png" "${OUT}.png"
echo "Copied visualization to ${OUT}.png"
