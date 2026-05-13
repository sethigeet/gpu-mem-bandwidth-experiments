#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/sync_remote.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-results/attention_bw_nsys_${TIMESTAMP}}
if [[ $# -gt 0 ]]; then
  shift
fi

ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_nsys.sh $OUT $@"

# Generate visualization remotely and copy image
ssh "$HOST" "cd $REMOTE_DIR && uv run python main.py visualize ${OUT}.sqlite -o ${OUT}.png"
mkdir -p "$(dirname "$OUT")"
scp "$HOST:$REMOTE_DIR/${OUT}.png" "${OUT}.png"
echo "Copied visualization to ${OUT}.png"

# Raw results are large, uncomment to copy if needed
# scp "$HOST:$REMOTE_DIR/${OUT}.nsys-rep" "${OUT}.nsys-rep"
# scp "$HOST:$REMOTE_DIR/${OUT}.sqlite" "${OUT}.sqlite"
