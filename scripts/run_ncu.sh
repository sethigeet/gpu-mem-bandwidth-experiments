#!/usr/bin/env bash

set -euo pipefail;

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/attention_bw_ncu_${TIMESTAMP}.csv}
shift

mkdir -p "$(dirname "$OUT")"

ncu \
  --target-processes all \
  --nvtx --nvtx-include "regex:attention_bw:.*:iter]" \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
  --csv --log-file "$OUT" \
  uv run main.py run --kernels all \
  --iters 5 \
  --warmup 3 "$@"