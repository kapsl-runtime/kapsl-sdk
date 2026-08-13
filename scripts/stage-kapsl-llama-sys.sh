#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$REPO_ROOT/third_party/llama.cpp-kapsl"
STAGING_DIR="$REPO_ROOT/patches/llama-cpp-sys-2/llama.cpp"

if [[ ! -f "$SOURCE_DIR/CMakeLists.txt" ]]; then
  echo "llama.cpp submodule is missing; run: git submodule update --init --recursive" >&2
  exit 1
fi

mkdir -p "$STAGING_DIR"
rsync -a --delete --exclude='.git/' "$SOURCE_DIR/" "$STAGING_DIR/"

echo "Staged the Kapsl llama.cpp fork for cargo package/publish: $STAGING_DIR"
