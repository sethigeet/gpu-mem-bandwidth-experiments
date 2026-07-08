#!/usr/bin/env bash

set -euo pipefail

# Profile one component stage with Nsight Compute. Keep decode token counts low:
# ncu replays kernels several times to collect counters.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/component_bw_ncu_${TIMESTAMP}.csv}
if [[ $# -gt 0 ]]; then
  shift
fi

mkdir -p "$(dirname "$OUT")"

ncu \
  --target-processes all \
  --nvtx --nvtx-include "regex:component_bw:.*:iter]" \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
  --csv --log-file "$OUT" \
  uv run component_main.py run \
  --decode-tokens 1 \
  --warmup-tokens 2 "$@"

