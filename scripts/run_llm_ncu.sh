#!/usr/bin/env bash

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/llm_bw_ncu_${TIMESTAMP}.csv}
shift || true

mkdir -p "$(dirname "$OUT")"

# NOTE: keep the decode tokens small as ncu takes a long time for each pass
ncu \
  --target-processes all \
  --nvtx --nvtx-include "regex:llm_bw:.*:iter]" \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
  --csv --log-file "$OUT" \
  uv run llm_main.py run \
  --decode-tokens 1 \
  --warmup-tokens 2 "$@"
