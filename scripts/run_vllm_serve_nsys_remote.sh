#!/usr/bin/env bash

set -euo pipefail

# Sync and start vLLM serving profiling under request load in a detached tmux
# session on the GPU host. Fetch artifacts with scripts/fetch_vllm_serve_nsys_remote.sh.
# Usage: run_vllm_serve_nsys_remote.sh [HOST] [REMOTE_DIR] [extra vllm_main.py serve args...]

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
OUT=${OUT:-results/vllm_bw_serve_nsys_${TIMESTAMP}}
SESSION=${TMUX_SESSION:-vllm_bw_serve_${TIMESTAMP}}
REMOTE_LOG=${OUT}.remote.log
REMOTE_DIR_ABS=$(ssh "$HOST" "cd $REMOTE_DIR && pwd")

REMOTE_CMD=$(printf \
  'set -o pipefail; cd %q || exit 1; mkdir -p %q; ./scripts/run_vllm_serve_nsys.sh %q %s 2>&1 | tee -a %q; code=${PIPESTATUS[0]}; echo "__VLLM_BW_DONE_EXIT_${code}__" | tee -a %q; exit $code' \
  "$REMOTE_DIR_ABS" "$(dirname "$OUT")" "$OUT" "$EXTRA_ARGS" "$REMOTE_LOG" "$REMOTE_LOG")

ssh "$HOST" "tmux new-session -d -s '$SESSION' $(printf '%q' "$REMOTE_CMD")"

echo "Started remote tmux session: ${SESSION}"
echo "Remote output prefix: ${OUT}"
echo "Remote log: ${REMOTE_LOG}"
echo "Watch with: ssh ${HOST} \"tmux attach -t ${SESSION}\""
echo "Fetch when done: scripts/fetch_vllm_serve_nsys_remote.sh ${HOST} ${REMOTE_DIR_ABS} ${OUT}"

