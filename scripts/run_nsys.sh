#!/usr/bin/env bash

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/attention_bw_nsys_${TIMESTAMP}}
shift || true

mkdir -p "$(dirname "$OUT")"

nsys profile \
  --trace=cuda,nvtx \
  --gpu-metrics-devices=all \
  --gpu-metrics-frequency=10000 \
  --output="$OUT" \
  --force-overwrite=true \
  uv run main.py run \
  --iters 1 \
  --warmup 0 "$@"

nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
echo "Exported to ${OUT}.sqlite"
