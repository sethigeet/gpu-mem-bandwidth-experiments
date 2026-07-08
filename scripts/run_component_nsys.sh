#!/usr/bin/env bash

set -euo pipefail

# Profile the staged component matrix with Nsight Systems and render a throughput
# waterfall plus DRAM/SM utilization timeline.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/component_bw_nsys_${TIMESTAMP}}
if [[ $# -gt 0 ]]; then
  shift
fi

mkdir -p "$(dirname "$OUT")"

NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
NSYS_GPU_METRICS_DEVICES=${NSYS_GPU_METRICS_DEVICES:-all}
NSYS_GPU_METRICS_FREQUENCY=${NSYS_GPU_METRICS_FREQUENCY:-50000}

nsys profile \
  --trace="$NSYS_TRACE" \
  --gpu-metrics-devices="$NSYS_GPU_METRICS_DEVICES" \
  --gpu-metrics-frequency="$NSYS_GPU_METRICS_FREQUENCY" \
  --duration=0 \
  --output="$OUT" \
  --force-overwrite=true \
  uv run component_main.py matrix -o "${OUT}.csv" "$@"

nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
uv run component_main.py visualize "${OUT}.csv" -o "${OUT}_summary.png"
uv run component_main.py visualize "${OUT}.sqlite" -o "${OUT}.png" --summary-output "${OUT}_nsys_summary.csv"

echo "Exported ${OUT}.sqlite"
echo "Wrote ${OUT}.csv, ${OUT}_summary.png, ${OUT}.png, and ${OUT}_nsys_summary.csv"

