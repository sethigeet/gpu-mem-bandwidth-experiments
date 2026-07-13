#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

REMOTE_HOST_DEFAULT=${REMOTE_HOST:-hinton-01}
REMOTE_DIR_DEFAULT=${REMOTE_DIR:-~/code/attention-bw}

parse_common_args() {
  COMMON_REMOTE=false
  COMMON_DETACH=false
  COMMON_HOST=$REMOTE_HOST_DEFAULT
  COMMON_REMOTE_DIR=$REMOTE_DIR_DEFAULT
  COMMON_OUT=${OUT:-}
  COMMON_EXTRA=()
  while (($#)); do
    case "$1" in
      --remote)
        COMMON_REMOTE=true
        shift
        ;;
      --detach)
        COMMON_DETACH=true
        shift
        ;;
      --host)
        COMMON_HOST=${2:?--host requires a value}
        shift 2
        ;;
      --remote-dir)
        COMMON_REMOTE_DIR=${2:?--remote-dir requires a value}
        shift 2
        ;;
      --out)
        COMMON_OUT=${2:?--out requires a value}
        shift 2
        ;;
      --)
        shift
        COMMON_EXTRA=("$@")
        return
        ;;
      *)
        echo "Unknown wrapper option: $1 (put benchmark arguments after --)" >&2
        return 2
        ;;
    esac
  done
}

sync_project() {
  local host=$1
  local remote_dir=$2
  ssh "$host" "mkdir -p $remote_dir"
  rsync -az --delete \
    --exclude '.git/' \
    --filter=':- .gitignore' \
    "$PROJECT_ROOT/" "$host:$remote_dir/"
}

resolve_remote_dir() {
  local host=$1
  local remote_dir=$2
  ssh "$host" "cd $remote_dir && pwd"
}

shell_join() {
  printf '%q ' "$@"
}

run_remote() {
  local host=$1
  local remote_dir=$2
  shift 2
  local command
  command=$(shell_join "$@")
  ssh "$host" "cd $(printf '%q' "$remote_dir") && $command"
}

start_remote_tmux() {
  local host=$1
  local remote_dir=$2
  local session=$3
  local remote_log=$4
  shift 4
  local command remote_command
  command=$(shell_join "$@")
  printf -v remote_command \
    'set -o pipefail; cd %q || exit 1; mkdir -p %q; %s 2>&1 | tee -a %q; code=${PIPESTATUS[0]}; echo "__BW_DONE_EXIT_${code}__" | tee -a %q; exit $code' \
    "$remote_dir" "$(dirname "$remote_log")" "$command" "$remote_log" "$remote_log"
  ssh "$host" "tmux new-session -d -s $(printf '%q' "$session") $(printf '%q' "$remote_command")"
}

copy_remote_file() {
  local host=$1
  local remote_dir=$2
  local remote_path=$3
  local local_path=${4:-$remote_path}
  mkdir -p "$(dirname "$local_path")"
  scp "$host:$remote_dir/$remote_path" "$local_path"
}

copy_remote_dir() {
  local host=$1
  local remote_dir=$2
  local path=$3
  mkdir -p "$(dirname "$path")"
  scp -r "$host:$remote_dir/$path" "$path"
}
