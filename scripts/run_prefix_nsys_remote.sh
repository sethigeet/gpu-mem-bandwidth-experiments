#!/usr/bin/env bash
set -euo pipefail

# Sync, profile a single prefix-sharing config under nsys remotely, render the DRAM
# bandwidth timeline, and copy the plot back.
# Usage: run_prefix_nsys_remote.sh EXPERIMENT [extra prefix_main.py args...]
#   e.g. run_prefix_nsys_remote.sh homogeneity --values 1.0   (fully homogeneous)
#        run_prefix_nsys_remote.sh homogeneity --values 0.5   (mixed prefixes)

EXPERIMENT=${1:?usage: run_prefix_nsys_remote.sh EXPERIMENT [args...]}
shift

EXTRA_ARGS=$(printf '%q ' "$@")
set --

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/sync_remote.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-results/prefix_bw_nsys_${EXPERIMENT}_${TIMESTAMP}}

ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_prefix_nsys.sh '$OUT' '$EXPERIMENT' $EXTRA_ARGS"
ssh "$HOST" "cd $REMOTE_DIR && uv run python prefix_main.py visualize ${OUT}.sqlite -o ${OUT}.png"
ssh "$HOST" "cd $REMOTE_DIR && uv run python - '${OUT}.sqlite' '${OUT}.metrics.csv' <<'PY'
import sys
from pathlib import Path

from llm_bw.visualize.nsys import load_nsys_metrics

sqlite_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

metrics, _ = load_nsys_metrics(sqlite_path, exclude_warmup=True)
bw = metrics[metrics['metric_name'].str.contains('DRAM', na=False)].copy()
sm = metrics[metrics['metric_name'].str.contains('SMs Active', na=False)].copy()

rows = []
for label, df in (('dram', bw), ('sm_active', sm)):
    for metric_name, mdf in df.groupby('metric_name'):
        values = mdf['value']
        rows.append({
            'metric_group': label,
            'metric_name': metric_name,
            'samples': int(values.count()),
            'mean_pct': float(values.mean()) if not values.empty else 0.0,
            'max_pct': float(values.max()) if not values.empty else 0.0,
            'p95_pct': float(values.quantile(0.95)) if not values.empty else 0.0,
            'sum_pct_samples': float(values.sum()) if not values.empty else 0.0,
        })

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open('w') as f:
    f.write('metric_group,metric_name,samples,mean_pct,max_pct,p95_pct,sum_pct_samples\n')
    for row in rows:
        f.write(
            f"{row['metric_group']},{row['metric_name']},{row['samples']},"
            f"{row['mean_pct']:.6f},{row['max_pct']:.6f},{row['p95_pct']:.6f},"
            f"{row['sum_pct_samples']:.6f}\n"
        )
print(f'Wrote {output_path}')
PY"

mkdir -p "$(dirname "$OUT")"
scp "$HOST:$REMOTE_DIR/${OUT}.csv" "${OUT}.csv"
scp "$HOST:$REMOTE_DIR/${OUT}.metrics.csv" "${OUT}.metrics.csv"
scp "$HOST:$REMOTE_DIR/${OUT}.png" "${OUT}.png"
echo "Copied ${OUT}.csv, ${OUT}.metrics.csv, and ${OUT}.png"
