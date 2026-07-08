#!/usr/bin/env bash

set -euo pipefail

# Usage: run_component_nsys_remote.sh [HOST] [REMOTE_DIR] [extra component_main.py matrix args...]

HOST_ARG=${REMOTE_HOST:-hinton-01}
REMOTE_DIR_ARG=${REMOTE_DIR:-~/code/attention-bw}

if [[ $# -gt 0 && "$1" != --* ]]; then
  HOST_ARG=$1
  shift
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  REMOTE_DIR_ARG=$1
  shift
fi

EXTRA_ARGS=$(printf '%q ' "$@")
set -- "$HOST_ARG" "$REMOTE_DIR_ARG"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/sync_remote.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-results/component_bw_nsys_${TIMESTAMP}}

ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_component_nsys.sh '$OUT' $EXTRA_ARGS"

mkdir -p "$(dirname "$OUT")"
scp "$HOST:$REMOTE_DIR/${OUT}.csv" "${OUT}.csv"
scp "$HOST:$REMOTE_DIR/${OUT}_summary.png" "${OUT}_summary.png"
scp "$HOST:$REMOTE_DIR/${OUT}.png" "${OUT}.png"
scp "$HOST:$REMOTE_DIR/${OUT}_nsys_summary.csv" "${OUT}_nsys_summary.csv"
scp "$HOST:$REMOTE_DIR/${OUT}.config.json" "${OUT}.config.json"
echo "Copied component artifacts to $(dirname "$OUT")"

