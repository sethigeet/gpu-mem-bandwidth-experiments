#!/usr/bin/env bash
set -euo pipefail

sync_remote() {
  local host=$1
  local root=$2
  local dest=$3

  ssh "$host" "mkdir -p $dest"
  rsync -az --delete \
    --exclude '.git/' \
    --filter=':- .gitignore' \
    "$root/" "$host:$dest/"
}

HOST=${1:-hinton-01}
REMOTE_DIR=${2:-~/code/attention-bw}
sync_remote "$HOST" "$(pwd)" "$REMOTE_DIR"

echo "synced $(pwd) to $HOST:$REMOTE_DIR"
