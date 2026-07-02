#!/usr/bin/env bash

set -euo pipefail

# Profile a single prefix-sharing config under nsys to capture the DRAM bandwidth
# timeline (the paper's Fig 4b claim). Pass a single sweep value so the timeline is clean.
# Usage: run_prefix_nsys.sh OUT EXPERIMENT [extra prefix_main.py args...]

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${1:-results/prefix_bw_nsys_${TIMESTAMP}}
shift || true
EXPERIMENT=${1:?usage: run_prefix_nsys.sh OUT EXPERIMENT [args...]}
shift

mkdir -p "$(dirname "$OUT")"

export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  if [[ -z "${TRITON_PTXAS_PATH:-}" && -x "$CUDA_HOME/bin/ptxas" ]]; then
    export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
  fi
fi
if [[ -n "${PREFIX_PYTHON:-}" ]]; then
  RUNNER=("$PREFIX_PYTHON")
else
  RUNNER=(uv run python)
fi
NSYS_BIN=${NSYS_BIN:-nsys}
GPU_METRICS_DEVICE_OPT=--gpu-metrics-devices
if ! "$NSYS_BIN" profile --help 2>/dev/null | python3 -c "import sys; sys.exit(0 if '--gpu-metrics-devices' in sys.stdin.read() else 1)"; then
  GPU_METRICS_DEVICE_OPT=--gpu-metrics-device
fi
if [[ -z "${GPU_METRICS_DEVICES:-}" ]]; then
  if [[ "$GPU_METRICS_DEVICE_OPT" == "--gpu-metrics-devices" ]]; then
    GPU_METRICS_DEVICES=cuda-visible
  else
    GPU_METRICS_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
    GPU_METRICS_DEVICES=${GPU_METRICS_DEVICES:-0}
  fi
fi
GPU_METRICS_FREQUENCY=${GPU_METRICS_FREQUENCY:-200000}
NSYS_DURATION=${NSYS_DURATION:-}
NSYS_DURATION_ARGS=()
if [[ -n "$NSYS_DURATION" ]]; then
  NSYS_DURATION_ARGS=(--duration="$NSYS_DURATION")
fi

"$NSYS_BIN" profile \
  --trace=cuda,nvtx \
  "$GPU_METRICS_DEVICE_OPT=$GPU_METRICS_DEVICES" \
  --gpu-metrics-frequency="$GPU_METRICS_FREQUENCY" \
  "${NSYS_DURATION_ARGS[@]}" \
  --output="$OUT" \
  --force-overwrite=true \
  "${RUNNER[@]}" prefix_main.py "$EXPERIMENT" -o "${OUT}.csv" "$@"

"$NSYS_BIN" export --force-overwrite=true --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
echo "Exported to ${OUT}.sqlite"
