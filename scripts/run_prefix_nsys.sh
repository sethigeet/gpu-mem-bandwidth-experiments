#!/usr/bin/env bash

set -euo pipefail

# Profile a single prefix-sharing config under nsys to capture the DRAM bandwidth
# timeline (the paper's Fig 4b claim). Pass a single sweep value so the timeline is clean.
# Usage: run_prefix_nsys.sh OUT EXPERIMENT [extra prefix_main.py args...]

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/prefix_bw_nsys_${TIMESTAMP}}
shift || true
EXPERIMENT=${1:?usage: run_prefix_nsys.sh OUT EXPERIMENT [args...]}
shift

mkdir -p "$(dirname "$OUT")"

nsys profile \
  --trace=cuda,nvtx \
  --gpu-metrics-devices=all \
  --gpu-metrics-frequency=200000 \
  --duration=0 \
  --output="$OUT" \
  --force-overwrite=true \
  uv run prefix_main.py "$EXPERIMENT" -o "${OUT}.csv" "$@"

nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
echo "Exported to ${OUT}.sqlite"
