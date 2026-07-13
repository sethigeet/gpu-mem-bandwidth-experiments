#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_remote_lib.sh"

COMMAND=${1:-}
case "$COMMAND" in
  profile|compare|fetch|install) ;;
  *)
    echo "Usage: $0 {profile|compare|fetch|install} [serve|client] [--remote] [--detach] [--host HOST] [--remote-dir DIR] [--out PREFIX] -- [vLLM args]" >&2
    exit 2
    ;;
esac
shift

PROFILE_SCOPE=
if [[ "$COMMAND" == "profile" ]]; then
  PROFILE_SCOPE=${1:-}
  [[ "$PROFILE_SCOPE" == "serve" || "$PROFILE_SCOPE" == "client" ]] || {
    echo "profile requires a scope: serve or client" >&2
    exit 2
  }
  shift
fi
parse_common_args "$@"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ "$COMMAND" == "compare" ]]; then
  OUT=${COMMON_OUT:-results/vllm_scheduling_compare_${TIMESTAMP}}
elif [[ "$COMMAND" == "profile" ]]; then
  OUT=${COMMON_OUT:-results/vllm_bw_${PROFILE_SCOPE}_nsys_${TIMESTAMP}}
else
  OUT=$COMMON_OUT
fi

if [[ "$COMMAND" == "install" ]]; then
  [[ "$COMMON_REMOTE" == true ]] || {
    echo "vLLM is GPU-host-only; use install --remote" >&2
    exit 2
  }
  PACKAGES=("${COMMON_EXTRA[@]}")
  ((${#PACKAGES[@]})) || PACKAGES=(vllm)
  sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" uv pip install "${PACKAGES[@]}"
  exit 0
fi

if [[ "$COMMAND" == "fetch" ]]; then
  [[ -n "$OUT" ]] || { echo "fetch requires --out PREFIX" >&2; exit 2; }
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  if ssh "$COMMON_HOST" "test -d $(printf '%q' "$REMOTE_DIR_ABS/$OUT")"; then
    mkdir -p "$OUT"
    rsync -az \
      --exclude '*.sqlite' \
      --exclude '*.nsys-rep' \
      "$COMMON_HOST:$REMOTE_DIR_ABS/$OUT/" "$OUT/"
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.remote.log"
    echo "Copied scheduling comparison results to $OUT (raw NSYS files remain remote)"
  else
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.png"
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}_summary.csv"
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.remote.log"
    copy_remote_dir "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}_logs"
    echo "Copied vLLM profile artifacts for $OUT"
  fi
  exit 0
fi

if [[ "$COMMON_REMOTE" == true ]]; then
  sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  if [[ "$COMMAND" == "compare" ]]; then
    REMOTE_ARGS=(./scripts/vllm_bw.sh compare --out "$OUT" -- "${COMMON_EXTRA[@]}")
  else
    REMOTE_ARGS=(./scripts/vllm_bw.sh profile "$PROFILE_SCOPE" --out "$OUT" -- "${COMMON_EXTRA[@]}")
  fi
  if [[ "$COMMON_DETACH" == true ]]; then
    SESSION=${TMUX_SESSION:-vllm_bw_${COMMAND}_${TIMESTAMP}}
    start_remote_tmux \
      "$COMMON_HOST" "$REMOTE_DIR_ABS" "$SESSION" "${OUT}.remote.log" "${REMOTE_ARGS[@]}"
    echo "Started remote tmux session: $SESSION"
    echo "Remote output: $OUT"
    echo "Fetch with: scripts/vllm_bw.sh fetch --host $COMMON_HOST --remote-dir $REMOTE_DIR_ABS --out $OUT"
    exit 0
  fi
  run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" "${REMOTE_ARGS[@]}"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

if [[ "$COMMAND" == "compare" ]]; then
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    echo "GPU busy; waiting ${GPU_WAIT_INTERVAL_S:-60} seconds before vLLM comparison"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    sleep "${GPU_WAIT_INTERVAL_S:-60}"
  done
  uv run vllm_main.py scheduling-compare --output-dir "$OUT" "${COMMON_EXTRA[@]}"
elif [[ "$PROFILE_SCOPE" == "client" ]]; then
  uv run vllm_main.py client-nsys \
    --output-prefix "$OUT" --log-dir "${OUT}_logs" "${COMMON_EXTRA[@]}"
else
  nsys profile \
    --trace="${NSYS_TRACE:-cuda,nvtx}" \
    --gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES:-all}" \
    --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY:-50000}" \
    --duration=0 --output="$OUT" --force-overwrite=true \
    uv run vllm_main.py serve --log-dir "${OUT}_logs" "${COMMON_EXTRA[@]}"
  nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
  uv run vllm_main.py visualize \
    "${OUT}.sqlite" -o "${OUT}.png" --summary-output "${OUT}_summary.csv"
fi
