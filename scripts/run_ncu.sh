#!/usr/bin/env bash

set -euo pipefail;

OUT=${1:-results/attention_bw_ncu.csv}
shift

mkdir -p "$(dirname "$OUT")"

ncu \
  --target-processes all \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv --log-file "$OUT" \
  uv run main.py \
  --iters 5 \
  --warmup 3 "$@"