#!/usr/bin/env bash
set -euo pipefail

sync_remote() {
  local host=$1
  local root=$2
  local dest=$3

  ssh "$host" "mkdir -p $dest"
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv*/' \
    --filter=':- .gitignore' \
    "$root/" "$host:$dest/"
}

HOST=${1:-${HOST:-hinton-01}}
REMOTE_DIR=${2:-${REMOTE_DIR:-~/code/attention-bw}}
sync_remote "$HOST" "$(pwd)" "$REMOTE_DIR"

echo "synced $(pwd) to $HOST:$REMOTE_DIR"
