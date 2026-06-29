#!/usr/bin/env bash

set -euo pipefail

# Run one prefix-sharing sweep on the GPU host and render the throughput plot.
# Usage: run_prefix.sh EXPERIMENT [OUT] [extra prefix_main.py args...]
#   EXPERIMENT: homogeneity | prefix-length | num-groups | batch-size

EXPERIMENT=${1:?usage: run_prefix.sh EXPERIMENT [OUT] [args...]}
shift

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/prefix_bw_${EXPERIMENT}_${TIMESTAMP}}
shift || true

mkdir -p "$(dirname "$OUT")"

export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

uv run prefix_main.py "$EXPERIMENT" -o "${OUT}.csv" "$@"
uv run prefix_main.py visualize "${OUT}.csv" -o "${OUT}.png"
echo "Wrote ${OUT}.csv and ${OUT}.png"
