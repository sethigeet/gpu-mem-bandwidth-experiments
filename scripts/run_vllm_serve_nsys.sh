#!/usr/bin/env bash

set -euo pipefail

# Profile a standard vLLM OpenAI server while `vllm bench serve` sends request load.
# Usage: run_vllm_serve_nsys.sh [OUT] [extra vllm_main.py serve args...]

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/vllm_bw_serve_nsys_${TIMESTAMP}}
shift || true

mkdir -p "$(dirname "$OUT")"

export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
NSYS_GPU_METRICS_FREQUENCY=${NSYS_GPU_METRICS_FREQUENCY:-50000}

nsys profile \
  --trace="$NSYS_TRACE" \
  --gpu-metrics-devices=all \
  --gpu-metrics-frequency="$NSYS_GPU_METRICS_FREQUENCY" \
  --duration=0 \
  --output="$OUT" \
  --force-overwrite=true \
  uv run vllm_main.py serve \
  --log-dir "${OUT}_logs" "$@"

nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
uv run vllm_main.py visualize "${OUT}.sqlite" -o "${OUT}.png" --summary-output "${OUT}_summary.csv"

echo "Exported ${OUT}.sqlite"
echo "Wrote ${OUT}.png and ${OUT}_summary.csv"

