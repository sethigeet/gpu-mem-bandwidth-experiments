#!/usr/bin/env bash

set -euo pipefail

# Fetch artifacts from a completed remote tmux vLLM serving profile.
# Usage: fetch_vllm_serve_nsys_remote.sh [HOST] [REMOTE_DIR] OUT

HOST_ARG=${REMOTE_HOST:-hinton-01}
REMOTE_DIR_ARG=${REMOTE_DIR:-~/code/attention-bw}

if [[ $# -gt 1 && "$1" != --* ]]; then
  HOST_ARG=$1
  shift
fi

if [[ $# -gt 1 && "$1" != --* ]]; then
  REMOTE_DIR_ARG=$1
  shift
fi

OUT=${1:?usage: fetch_vllm_serve_nsys_remote.sh [HOST] [REMOTE_DIR] OUT}

REMOTE_DIR_ABS=$(ssh "$HOST_ARG" "cd $REMOTE_DIR_ARG && pwd")

mkdir -p "$(dirname "$OUT")"
scp "$HOST_ARG:$REMOTE_DIR_ABS/${OUT}.png" "${OUT}.png"
scp "$HOST_ARG:$REMOTE_DIR_ABS/${OUT}_summary.csv" "${OUT}_summary.csv"
scp "$HOST_ARG:$REMOTE_DIR_ABS/${OUT}.remote.log" "${OUT}.remote.log"
scp -r "$HOST_ARG:$REMOTE_DIR_ABS/${OUT}_logs" "${OUT}_logs"

echo "Copied visualization to ${OUT}.png"
echo "Copied summary to ${OUT}_summary.csv"
echo "Copied remote log to ${OUT}.remote.log"
echo "Copied logs to ${OUT}_logs"

