#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_remote.sh"

COMMAND=${1:-}
[[ "$COMMAND" == "sweep" || "$COMMAND" == "nsys" ]] || {
  echo "Usage: $0 {sweep|nsys} EXPERIMENT [--remote] [--host HOST] [--remote-dir DIR] [--out PREFIX] -- [benchmark args]" >&2
  exit 2
}
EXPERIMENT=${2:?An experiment name is required}
shift 2
parse_common_args "$@"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${COMMON_OUT:-results/prefix_cache_${COMMAND}_${EXPERIMENT}_${TIMESTAMP}}

if [[ "$COMMON_REMOTE" == true ]]; then
  sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" \
    ./scripts/prefix-cache.sh "$COMMAND" "$EXPERIMENT" --out "$OUT" -- "${COMMON_EXTRA[@]}"
  if [[ "$COMMAND" == "sweep" ]]; then
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.csv"
  fi
  copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.png"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

if [[ "$COMMAND" == "sweep" ]]; then
  uv run gpu-memory-benchmarks prefix-cache "$EXPERIMENT" -o "${OUT}.csv" "${COMMON_EXTRA[@]}"
  uv run gpu-memory-benchmarks prefix-cache visualize "${OUT}.csv" -o "${OUT}.png"
else
  nsys profile \
    --trace="${NSYS_TRACE:-cuda,nvtx}" \
    --gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES:-all}" \
    --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY:-200000}" \
    --duration=0 --output="$OUT" --force-overwrite=true \
    uv run gpu-memory-benchmarks prefix-cache "$EXPERIMENT" -o "${OUT}.csv" "${COMMON_EXTRA[@]}"
  nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
  uv run gpu-memory-benchmarks prefix-cache visualize "${OUT}.sqlite" -o "${OUT}.png"
fi
