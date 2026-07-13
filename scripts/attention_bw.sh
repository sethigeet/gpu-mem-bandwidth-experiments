#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_remote_lib.sh"

COMMAND=${1:-}
[[ "$COMMAND" == "ncu" || "$COMMAND" == "nsys" ]] || {
  echo "Usage: $0 {ncu|nsys} [--remote] [--host HOST] [--remote-dir DIR] [--out PREFIX] -- [benchmark args]" >&2
  exit 2
}
shift
parse_common_args "$@"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${COMMON_OUT:-results/attention_bw_${COMMAND}_${TIMESTAMP}}

if [[ "$COMMON_REMOTE" == true ]]; then
  sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" \
    ./scripts/attention_bw.sh "$COMMAND" --out "$OUT" -- "${COMMON_EXTRA[@]}"
  copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.png"
  echo "Copied visualization to ${OUT}.png"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
if [[ "$COMMAND" == "ncu" ]]; then
  ncu \
    --target-processes all \
    --nvtx --nvtx-include "regex:attention_bw:.*:iter]" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
    --csv --log-file "${OUT}.csv" \
    uv run main.py run --kernels all --iters 5 --warmup 3 "${COMMON_EXTRA[@]}"
  uv run main.py visualize "${OUT}.csv" -o "${OUT}.png"
else
  nsys profile \
    --trace="${NSYS_TRACE:-cuda,nvtx}" \
    --gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES:-all}" \
    --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY:-200000}" \
    --duration=0 --output="$OUT" --force-overwrite=true \
    uv run main.py run --iters 5 --warmup 2 "${COMMON_EXTRA[@]}"
  nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
  uv run main.py visualize "${OUT}.sqlite" -o "${OUT}.png"
fi
