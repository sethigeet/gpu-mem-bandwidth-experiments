#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_remote_lib.sh"

COMMAND=${1:-}
case "$COMMAND" in
  ncu|nsys|ncu-matrix|bundle|sweep|ncu-sweep|sweep-bundle|fetch) ;;
  *)
    echo "Usage: $0 {ncu|nsys|ncu-matrix|bundle|sweep|ncu-sweep|sweep-bundle|fetch} [--remote] [--detach] [--host HOST] [--remote-dir DIR] [--out PREFIX] -- [benchmark args]" >&2
    exit 2
    ;;
esac
shift
parse_common_args "$@"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=${COMMON_OUT:-results/component_bw_${COMMAND}_${TIMESTAMP}}

if [[ "$COMMAND" == "fetch" ]]; then
  [[ -n "$COMMON_OUT" ]] || { echo "fetch requires --out PREFIX" >&2; exit 2; }
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  mkdir -p "$(dirname "$OUT")"
  scp "$COMMON_HOST:$REMOTE_DIR_ABS/${OUT}"'*' "$(dirname "$OUT")/"
  echo "Copied component artifacts for ${OUT}"
  exit 0
fi

if [[ "$COMMON_REMOTE" == true ]]; then
  sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  REMOTE_ARGS=(./scripts/component_bw.sh "$COMMAND" --out "$OUT" -- "${COMMON_EXTRA[@]}")
  if [[ "$COMMON_DETACH" == true ]]; then
    SESSION=${TMUX_SESSION:-component_bw_${COMMAND}_${TIMESTAMP}}
    start_remote_tmux \
      "$COMMON_HOST" "$REMOTE_DIR_ABS" "$SESSION" "${OUT}.remote.log" "${REMOTE_ARGS[@]}"
    echo "Started remote tmux session: $SESSION"
    echo "Remote output prefix: $OUT"
    echo "Fetch with: scripts/component_bw.sh fetch --host $COMMON_HOST --remote-dir $REMOTE_DIR_ABS --out $OUT"
    exit 0
  fi
  run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" "${REMOTE_ARGS[@]}"
  if [[ "$COMMAND" == "ncu" ]]; then
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.csv"
  elif [[ "$COMMAND" == "nsys" ]]; then
    for suffix in .csv _summary.png .png _nsys_summary.csv .config.json; do
      copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}${suffix}"
    done
  fi
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
case "$COMMAND" in
  ncu)
    ncu \
      --target-processes all \
      --replay-mode "${NCU_REPLAY_MODE:-kernel}" \
      --nvtx --nvtx-include "regex:component_bw:.*:iter]" \
      --metrics "${NCU_METRICS:-dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum}" \
      --csv --log-file "${OUT}.csv" \
      uv run component_main.py run "${COMMON_EXTRA[@]}" --decode-tokens 1 --warmup-tokens 2
    ;;
  nsys)
    nsys profile \
      --trace="${NSYS_TRACE:-cuda,nvtx}" \
      --gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES:-all}" \
      --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY:-50000}" \
      --duration=0 --output="$OUT" --force-overwrite=true \
      uv run component_main.py matrix -o "${OUT}.csv" "${COMMON_EXTRA[@]}"
    nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
    uv run component_main.py visualize "${OUT}.csv" -o "${OUT}_summary.png"
    uv run component_main.py visualize \
      "${OUT}.sqlite" -o "${OUT}.png" --summary-output "${OUT}_nsys_summary.csv"
    ;;
  ncu-matrix)
    while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
      echo "GPU busy; waiting ${GPU_WAIT_INTERVAL_S:-60} seconds"
      sleep "${GPU_WAIT_INTERVAL_S:-60}"
    done
    for stage in attention_kernel attention_layer mlp block blocks model paged_attention paged_model; do
      echo "__COMPONENT_NCU_STAGE_START_${stage}__"
      "$0" ncu --out "${OUT}_${stage}" -- --stage "$stage" "${COMMON_EXTRA[@]}"
      echo "__COMPONENT_NCU_STAGE_DONE_${stage}__"
    done
    ;;
  bundle)
    while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
      echo "GPU busy; waiting ${GPU_WAIT_INTERVAL_S:-60} seconds"
      sleep "${GPU_WAIT_INTERVAL_S:-60}"
    done
    uv run component_main.py matrix -o "${OUT}_throughput.csv" "${COMMON_EXTRA[@]}"
    uv run component_main.py visualize "${OUT}_throughput.csv" -o "${OUT}_throughput.png"
    "$0" ncu-matrix --out "${OUT}_ncu" -- "${COMMON_EXTRA[@]}"
    ;;
  sweep)
    uv run component_main.py sweep -o "${OUT}.csv" "${COMMON_EXTRA[@]}"
    ;;
  ncu-sweep)
    while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
      echo "GPU busy; waiting ${GPU_WAIT_INTERVAL_S:-60} seconds"
      sleep "${GPU_WAIT_INTERVAL_S:-60}"
    done
    BATCH_SIZES=${COMPONENT_BATCH_SIZES:-"1 2 4 8 16 32 40 48 64 128 256 512 1024"}
    for batch_size in $BATCH_SIZES; do
      for stage in attention_kernel attention_layer mlp block blocks model paged_attention paged_model; do
        stage_output="${OUT}_b${batch_size}_${stage}"
        if [[ -f "${stage_output}.done" || -f "${stage_output}.skipped" ]]; then
          echo "__COMPONENT_NCU_BATCH_${batch_size}_STAGE_SKIPPED_${stage}__"
          continue
        fi
        echo "__COMPONENT_NCU_BATCH_${batch_size}_STAGE_START_${stage}__"
        if NCU_REPLAY_MODE=${NCU_REPLAY_MODE:-kernel} \
          NCU_METRICS=${NCU_METRICS:-dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum} \
          "$0" ncu --out "$stage_output" -- \
          "${COMMON_EXTRA[@]}" --stage "$stage" --batch-size "$batch_size"; then
          touch "${stage_output}.done"
          echo "__COMPONENT_NCU_BATCH_${batch_size}_STAGE_DONE_${stage}__"
        else
          [[ ! -f "${stage_output}.csv" ]] || mv "${stage_output}.csv" "${stage_output}.failed.log"
          touch "${stage_output}.skipped"
          echo "__COMPONENT_NCU_BATCH_${batch_size}_STAGE_INFEASIBLE_${stage}__"
        fi
      done
    done
    ;;
  sweep-bundle)
    while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
      echo "GPU busy; waiting ${GPU_WAIT_INTERVAL_S:-60} seconds"
      sleep "${GPU_WAIT_INTERVAL_S:-60}"
    done
    uv run component_main.py sweep -o "${OUT}_throughput.csv" "${COMMON_EXTRA[@]}"
    COMPONENT_BATCH_SIZES=${COMPONENT_BATCH_SIZES:-"1 2 4 8 16 32 40 48 64 128 256 512 1024"} \
      "$0" ncu-sweep --out "${OUT}_ncu" -- "${COMMON_EXTRA[@]}"
    ;;
esac
