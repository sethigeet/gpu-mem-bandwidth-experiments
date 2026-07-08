#!/usr/bin/env bash

set -euo pipefail

# Usage: run_component_ncu_remote.sh [HOST] [REMOTE_DIR] [extra component_main.py run args...]

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
OUT=${OUT:-results/component_bw_ncu_${TIMESTAMP}}

ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_component_ncu.sh '${OUT}.csv' $EXTRA_ARGS"

mkdir -p "$(dirname "$OUT")"
scp "$HOST:$REMOTE_DIR/${OUT}.csv" "${OUT}.csv"
echo "Copied NCU results to ${OUT}.csv"

