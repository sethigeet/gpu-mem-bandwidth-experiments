#!/usr/bin/env bash

set -euo pipefail

# Run Nsight Compute once per component stage. This is intended to be launched
# inside tmux on the remote GPU host because NCU can run for a long time.

OUT_PREFIX=${1:-results/component_bw_ncu_10k}
shift || true

STAGES=(
  attention_kernel
  attention_layer
  mlp
  block
  blocks
  model
  paged_attention
  paged_model
)

mkdir -p "$(dirname "$OUT_PREFIX")"

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
  echo "GPU busy; waiting 60 seconds before component NCU run"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  sleep 60
done

for stage in "${STAGES[@]}"; do
  echo "__COMPONENT_NCU_STAGE_START_${stage}__"
  ./scripts/run_component_ncu.sh "${OUT_PREFIX}_${stage}.csv" --stage "$stage" "$@"
  echo "__COMPONENT_NCU_STAGE_DONE_${stage}__"
done

echo "__COMPONENT_NCU_ALL_DONE__"

