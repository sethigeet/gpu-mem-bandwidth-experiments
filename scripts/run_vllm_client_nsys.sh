#!/usr/bin/env bash

set -euo pipefail

# Start vLLM serve normally, then profile only `vllm bench serve` with nsys.
# Nsight GPU metrics are device-wide, so this captures the serving GPU bandwidth
# during request load without tracing server startup/model loading.
# Usage: run_vllm_client_nsys.sh [OUT] [extra vllm_main.py client-nsys args...]

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/vllm_bw_client_nsys_${TIMESTAMP}}
shift || true

mkdir -p "$(dirname "$OUT")"

export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

uv run vllm_main.py client-nsys \
  --output-prefix "$OUT" \
  --log-dir "${OUT}_logs" "$@"

