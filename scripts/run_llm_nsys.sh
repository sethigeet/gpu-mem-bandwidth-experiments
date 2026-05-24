#!/usr/bin/env bash

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/llm_bw_nsys_${TIMESTAMP}}
shift || true

mkdir -p "$(dirname "$OUT")"

nsys profile \
  --trace=cuda,nvtx \
  --gpu-metrics-devices=all \
  --gpu-metrics-frequency=200000 \
  --duration=0 \
  --output="$OUT" \
  --force-overwrite=true \
  uv run llm_main.py run \
  --decode-tokens 5 \
  --warmup-tokens 2 "$@"

nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
echo "Exported to ${OUT}.sqlite"
