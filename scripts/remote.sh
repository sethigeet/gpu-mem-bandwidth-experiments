#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_remote_lib.sh"

COMMAND=${1:-}
[[ "$COMMAND" == "sync" ]] || {
  echo "Usage: $0 sync [--host HOST] [--remote-dir DIR]" >&2
  exit 2
}
shift
parse_common_args "$@"
sync_project "$COMMON_HOST" "$COMMON_REMOTE_DIR"
echo "Synced $PROJECT_ROOT to $COMMON_HOST:$COMMON_REMOTE_DIR"
