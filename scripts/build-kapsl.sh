#!/bin/bash
# Build kapsl on the cluster.
#
# Usage:
#   ./scripts/build-kapsl.sh [feature]
#
# feature defaults to gguf-cuda-shared-kv.  Supported values:
#   gguf-cuda-shared-kv   llama.cpp fork + Kapsl shared KV pool (default)
#   gguf-cuda             llama.cpp with static CUDA KV
#   gguf-native           GGUF with Kapsl's native CUDA kernels
#   native                safetensors with Kapsl's native CUDA kernels
#   cuda                  alias for gguf-cuda
#
# Override any path via env var before calling, e.g.:
#   KAPSL_DIR=/other/path ./scripts/build-kapsl.sh
set -euo pipefail

# Default layout: this script lives in <kapsl-sdk>/scripts, with kapsl-engine
# checked out next to kapsl-sdk (kapsl-runtime's Cargo path deps require the
# side-by-side layout). Matches both the HPC checkout and a fresh cloud VM.
SCRIPT_SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAPSL_SDK_DIR="${KAPSL_SDK_DIR:-$SCRIPT_SDK_DIR}"
KAPSL_DIR="${KAPSL_DIR:-$(dirname "$KAPSL_SDK_DIR")/kapsl-engine/kapsl-runtime}"
LLVM_ENV="${LLVM_ENV:-/fs04/vf38/tngu0610/.conda/envs/llvm}"
KAPSL_LLAMA_CPP_DIR="${KAPSL_LLAMA_CPP_DIR:-$KAPSL_SDK_DIR/third_party/llama.cpp-kapsl}"
KAPSL_FEATURE="${1:-${KAPSL_FEATURE:-gguf-cuda-shared-kv}}"

# Normalise aliases.
case "$KAPSL_FEATURE" in
  gguf-cuda-shared-kv|gguf-cuda|gguf-native|native) ;;
  cuda) KAPSL_FEATURE="gguf-cuda" ;;
  *)
    echo "error: unsupported feature '$KAPSL_FEATURE'" >&2
    echo "       valid: gguf-cuda-shared-kv | gguf-cuda | gguf-native | native | cuda" >&2
    exit 2
    ;;
esac

echo "==> Building kapsl feature: $KAPSL_FEATURE"

# Validate directories.
if [[ ! -f "$KAPSL_DIR/Cargo.toml" ]]; then
  echo "error: KAPSL_DIR does not look like a Cargo workspace: $KAPSL_DIR" >&2
  exit 2
fi

if [[ "$KAPSL_FEATURE" == "gguf-cuda-shared-kv" ]]; then
  if [[ ! -f "$KAPSL_SDK_DIR/crates/kapsl-llm/Cargo.toml" ]]; then
    echo "error: KAPSL_SDK_DIR is wrong or not checked out: $KAPSL_SDK_DIR" >&2
    exit 2
  fi
  if [[ ! -d "$KAPSL_LLAMA_CPP_DIR" ]]; then
    echo "error: KAPSL_LLAMA_CPP_DIR missing: $KAPSL_LLAMA_CPP_DIR" >&2
    echo "       run: git submodule update --init  (inside KAPSL_SDK_DIR)" >&2
    exit 2
  fi
fi

echo "==> Cleaning feature build artifacts"
rm -rf "$KAPSL_DIR/target/release/build/llama-cpp-sys-2"*
rm -rf "$KAPSL_DIR/target/release/build/cudarc"*
rm -f  "$KAPSL_DIR/target/release/deps/libllama_cpp_sys_2-"*.rlib
rm -f  "$KAPSL_DIR/target/release/deps/libllama_cpp_2-"*.rlib

echo "==> Setting build environment"
conda deactivate 2>/dev/null || true

# HPC: libclang comes from a conda env and CUDA from a module. On cloud VMs
# (e.g. vast.ai) neither exists — rely on system libclang/nvcc instead.
if [[ -d "$LLVM_ENV/lib" ]]; then
  export LIBCLANG_PATH="$LLVM_ENV/lib"
  export BINDGEN_EXTRA_CLANG_ARGS="-I$LLVM_ENV/lib/clang/22/include"
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.2.0
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found; install the CUDA toolkit or load the cuda module" >&2
  exit 2
fi

export LLAMA_CUDA=1
export KAPSL_SDK_DIR
export KAPSL_LLAMA_CPP_DIR
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-native}"
export CMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS:--Xcompiler=-fPIC}"
export ORT_STRATEGY="${ORT_STRATEGY:-compile}"

cd "$KAPSL_DIR"

LOG="build-${KAPSL_FEATURE}.log"
echo "==> Building (log: $LOG)"
cargo build -p kapsl --release --no-default-features --features "$KAPSL_FEATURE" \
  2>&1 | tee "$LOG"

echo ""
echo "Binary: $KAPSL_DIR/target/release/kapsl"
echo "Log:    $KAPSL_DIR/$LOG"
