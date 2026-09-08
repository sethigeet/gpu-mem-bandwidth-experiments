#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_remote.sh"

COMMAND=${1:-}
case "$COMMAND" in
  profile|compare|fetch|install|install-policies|install-timing) ;;
  *)
    echo "Usage: $0 {profile|compare|fetch|install|install-policies|install-timing} [serve|client|overhead|model-gpu] [--remote] [--detach] [--host HOST] [--remote-dir DIR] [--out PREFIX] -- [vLLM args]" >&2
    exit 2
    ;;
esac
shift

PROFILE_SCOPE=
if [[ "$COMMAND" == "profile" ]]; then
  PROFILE_SCOPE=${1:-}
  [[ "$PROFILE_SCOPE" == "serve" || "$PROFILE_SCOPE" == "client" || "$PROFILE_SCOPE" == "overhead" || "$PROFILE_SCOPE" == "model-gpu" ]] || {
    echo "profile requires a scope: serve, client, overhead, or model-gpu" >&2
    exit 2
  }
  shift
fi
parse_common_args "$@"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ "$COMMAND" == "install-timing" ]]; then
  [[ "$COMMON_REMOTE" == true ]] || {
    echo "The instrumented vLLM installation is GPU-host-only; use install-timing --remote" >&2
    exit 2
  }
  sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  TIMING_ARGS=(python3 -m gpu_benchmarks.serving.timing_installer)
  if ((${#COMMON_EXTRA[@]})); then
    TIMING_ARGS+=("${COMMON_EXTRA[@]}")
  else
    TIMING_ARGS+=(--venv-python .venv/bin/python)
  fi
  run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" "${TIMING_ARGS[@]}"
  exit 0
fi

if [[ "$COMMAND" == "install-policies" ]]; then
  [[ "$COMMON_REMOTE" == true ]] || {
    echo "The policy-enabled vLLM fork is GPU-host-only; use install-policies --remote" >&2
    exit 2
  }
  sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
  REMOTE_DIR_ABS=$(resolve_remote_dir "$COMMON_HOST" "$COMMON_REMOTE_DIR")
  INSTALL_ARGS=(
    python3 -m gpu_benchmarks.serving.policy_installer
    "${COMMON_EXTRA[@]}"
  )
  if [[ -n ${WAIT_FOR_TMUX_SESSION:-} ]]; then
    INSTALL_ARGS+=(--wait-for-tmux-session "$WAIT_FOR_TMUX_SESSION")
  fi
  if [[ "$COMMON_DETACH" == true ]]; then
    SESSION=${TMUX_SESSION:-vllm_install_policies_${TIMESTAMP}}
    LOG="results/vllm_policy_install_${TIMESTAMP}.remote.log"
    start_remote_tmux \
      "$COMMON_HOST" "$REMOTE_DIR_ABS" "$SESSION" "$LOG" "${INSTALL_ARGS[@]}"
    echo "Started remote policy installation in tmux session: $SESSION"
    echo "Remote installation log: $LOG"
  else
    run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" "${INSTALL_ARGS[@]}"
  fi
  exit 0
fi

if [[ "$COMMAND" == "compare" ]]; then
  OUT=${COMMON_OUT:-results/vllm_scheduling_compare_${TIMESTAMP}}
elif [[ "$COMMAND" == "profile" ]]; then
  OUT=${COMMON_OUT:-results/vllm_${PROFILE_SCOPE}_nsys_${TIMESTAMP}}
else
  OUT=$COMMON_OUT
fi

if [[ "$COMMAND" == "install" ]]; then
  [[ "$COMMON_REMOTE" == true ]] || {
    echo "vLLM is GPU-host-only; use install --remote" >&2
    exit 2
  }
  PACKAGES=("${COMMON_EXTRA[@]}")
  ((${#PACKAGES[@]})) || PACKAGES=(vllm==0.22.1)
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
  elif ssh "$COMMON_HOST" "test -f $(printf '%q' "$REMOTE_DIR_ABS/${OUT}_timing.csv")"; then
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}_timing.csv"
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}_timing_raw.csv"
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.remote.log"
    copy_remote_dir "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}_logs"
    echo "Copied detailed CPU overhead artifacts for $OUT"
  elif ssh "$COMMON_HOST" "test -f $(printf '%q' "$REMOTE_DIR_ABS/${OUT}_model_gpu.csv")"; then
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}_model_gpu.csv"
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}_kernel_summary.csv"
    copy_remote_file "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}.remote.log"
    copy_remote_dir "$COMMON_HOST" "$REMOTE_DIR_ABS" "${OUT}_logs"
    echo "Copied model GPU timing artifacts for $OUT (raw NSYS files remain remote)"
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
  REMOTE_ENV=(env)
  for variable in \
    VLLM_ATTENTION_BACKEND \
    VLLM_BENCH_PROFILE_GPU \
    VLLM_BENCH_PROFILE_INTERVAL \
    VLLM_BENCH_PROFILE_SCHEDULER \
    VLLM_BENCH_PROFILE_TIMELINE \
    HF_HUB_OFFLINE \
    HF_HOME \
    WAIT_FOR_TMUX_SESSION; do
    if [[ -v "$variable" ]]; then
      REMOTE_ENV+=("$variable=${!variable}")
    fi
  done
  if [[ "$COMMAND" == "compare" ]]; then
    REMOTE_ARGS=(
      "${REMOTE_ENV[@]}"
      ./scripts/vllm-server.sh compare --out "$OUT" -- "${COMMON_EXTRA[@]}"
    )
  else
    REMOTE_ARGS=(
      "${REMOTE_ENV[@]}"
      ./scripts/vllm-server.sh profile "$PROFILE_SCOPE" --out "$OUT" -- "${COMMON_EXTRA[@]}"
    )
  fi
  if [[ "$COMMON_DETACH" == true ]]; then
    SESSION=${TMUX_SESSION:-vllm_${COMMAND}_${TIMESTAMP}}
    start_remote_tmux \
      "$COMMON_HOST" "$REMOTE_DIR_ABS" "$SESSION" "${OUT}.remote.log" "${REMOTE_ARGS[@]}"
    echo "Started remote tmux session: $SESSION"
    echo "Remote output: $OUT"
    echo "Fetch with: scripts/vllm-server.sh fetch --host $COMMON_HOST --remote-dir $REMOTE_DIR_ABS --out $OUT"
    exit 0
  fi
  run_remote "$COMMON_HOST" "$REMOTE_DIR_ABS" "${REMOTE_ARGS[@]}"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

if [[ "$COMMAND" == "compare" ]]; then
  HAS_CUSTOM_POLICY=false
  HAS_VLLM_EXECUTABLE=false
  for arg in "${COMMON_EXTRA[@]}"; do
    case "$arg" in
      radix_cost|chunked_hash_tree_bandit|chunked_hash_tree_python|chunked_hash_tree_cpp)
        HAS_CUSTOM_POLICY=true
        ;;
      --vllm-executable) HAS_VLLM_EXECUTABLE=true ;;
    esac
  done
  if [[ "$HAS_CUSTOM_POLICY" == true && "$HAS_VLLM_EXECUTABLE" == false ]]; then
    POLICY_VENV=${VLLM_POLICY_VENV:-$HOME/.cache/gpu-memory-benchmarks/vllm/policy_venv}
    COMMON_EXTRA+=(--vllm-executable "$POLICY_VENV/bin/vllm")
  fi
  if [[ -n ${WAIT_FOR_TMUX_SESSION:-} ]]; then
    while tmux has-session -t "$WAIT_FOR_TMUX_SESSION" 2>/dev/null; do
      echo "Waiting for tmux session $WAIT_FOR_TMUX_SESSION to finish"
      sleep "${GPU_WAIT_INTERVAL_S:-60}"
    done
  fi
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    echo "GPU busy; waiting ${GPU_WAIT_INTERVAL_S:-60} seconds before vLLM comparison"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    sleep "${GPU_WAIT_INTERVAL_S:-60}"
  done
  uv run gpu-memory-benchmarks vllm scheduling-compare --output-dir "$OUT" "${COMMON_EXTRA[@]}"
elif [[ "$PROFILE_SCOPE" == "client" ]]; then
  uv run gpu-memory-benchmarks vllm client-nsys \
    --output-prefix "$OUT" --log-dir "${OUT}_logs" "${COMMON_EXTRA[@]}"
elif [[ "$PROFILE_SCOPE" == "overhead" ]]; then
  export VLLM_BENCH_PROFILE_SCHEDULER=1
  uv run gpu-memory-benchmarks vllm serve --log-dir "${OUT}_logs" "${COMMON_EXTRA[@]}"
  uv run gpu-memory-benchmarks vllm timing-summary \
    --log "${OUT}_logs/server.log" --output "${OUT}_timing.csv" \
    --raw-output "${OUT}_timing_raw.csv"
elif [[ "$PROFILE_SCOPE" == "model-gpu" ]]; then
  export VLLM_BENCH_PROFILE_GPU=1
  nsys profile \
    --trace="${NSYS_TRACE:-cuda,nvtx}" \
    --duration=0 --output="$OUT" --force-overwrite=true \
    uv run gpu-memory-benchmarks vllm serve --log-dir "${OUT}_logs" "${COMMON_EXTRA[@]}"
  nsys export --type=sqlite --force-overwrite=true \
    --output="${OUT}.sqlite" "${OUT}.nsys-rep"
  nsys stats --report nvtx_gpu_proj_sum --format=csv --force-export=true \
    "${OUT}.nsys-rep" \
    > "${OUT}_model_gpu.csv"
  nsys stats --report cuda_gpu_kern_sum --format=csv --force-export=true \
    "${OUT}.nsys-rep" \
    > "${OUT}_kernel_summary.csv"
else
  nsys profile \
    --trace="${NSYS_TRACE:-cuda,nvtx}" \
    --gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES:-all}" \
    --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY:-50000}" \
    --duration=0 --output="$OUT" --force-overwrite=true \
    uv run gpu-memory-benchmarks vllm serve --log-dir "${OUT}_logs" "${COMMON_EXTRA[@]}"
  nsys export --type=sqlite --output="${OUT}.sqlite" "${OUT}.nsys-rep"
  uv run gpu-memory-benchmarks vllm visualize \
    "${OUT}.sqlite" -o "${OUT}.png" --summary-output "${OUT}_summary.csv"
fi
