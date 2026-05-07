#!/usr/bin/env bash
set -euo pipefail

# sync the local project to the remote host (also sets the HOST and REMOTE_DIR vars)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/sync_remote.sh"

OUT=${OUT:-results/attention_bw_ncu.csv}
if [[ $# -gt 0 ]]; then
  shift
fi

ssh "$HOST" "cd $REMOTE_DIR && ./scripts/run_ncu.sh $OUT $@"
scp "$HOST:$REMOTE_DIR/$OUT" "$OUT"
echo "copied remote results from $HOST:$REMOTE_DIR/$OUT to $OUT"