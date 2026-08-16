#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_remote.sh"

SUITE=${1:-}
COMMAND=${2:-}
case "$SUITE:$COMMAND" in
  attention:ncu|attention:nsys|model:ncu|model:nsys) ;;
  *)
    echo "Usage: $0 {attention|model} {ncu|nsys} [--remote] [--host HOST] [--remote-dir DIR] [--out PREFIX] -- [benchmark args]" >&2
    exit 2
    ;;
esac
shift 2
parse_common_args "$@"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${COMMON_OUT:-results/${SUITE}_${COMMAND}_${TIMESTAMP}}

if [[ "$COMMON_REMOTE" == true ]]; then
  sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" \
    ./scripts/profile.sh "$SUITE" "$COMMAND" --out "$OUT" -- "${COMMON_EXTRA[@]}"
  copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.png"
  echo "Copied visualization to ${OUT}.png"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
if [[ "$SUITE" == attention ]]; then
  RANGE_PATTERN='regex:gpu_memory:attention:.*:iter]'
  NCU_DEFAULTS=(--kernels all --iters 5 --warmup 3)
  NSYS_DEFAULTS=(--iters 5 --warmup 2)
else
  RANGE_PATTERN='regex:gpu_memory:model:.*:iter]'
  NCU_DEFAULTS=(--decode-tokens 1 --warmup-tokens 2)
  NSYS_DEFAULTS=(--decode-tokens 5 --warmup-tokens 2)
fi

if [[ "$COMMAND" == ncu ]]; then
  ncu \
    --target-processes all \
    --nvtx --nvtx-include "$RANGE_PATTERN" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
    --csv --log-file "${OUT}.csv" \
    uv run gpu-memory-benchmarks "$SUITE" run "${NCU_DEFAULTS[@]}" "${COMMON_EXTRA[@]}"
  uv run gpu-memory-benchmarks "$SUITE" visualize "${OUT}.csv" -o "${OUT}.png"
else
  nsys profile \
    --trace="${NSYS_TRACE:-cuda,nvtx}" \
    --gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES:-all}" \
    --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY:-200000}" \
    --duration=0 --output="$OUT" --force-overwrite=true \
    uv run gpu-memory-benchmarks "$SUITE" run "${NSYS_DEFAULTS[@]}" "${COMMON_EXTRA[@]}"
  nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
  uv run gpu-memory-benchmarks "$SUITE" visualize "${OUT}.sqlite" -o "${OUT}.png"
fi
