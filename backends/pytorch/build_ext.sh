#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-$(command -v python3)}"

if [ -z "$PYTHON_BIN" ]; then
  echo "Python not found. Set PYTHON or ensure python3 is on PATH."
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake not found. Install CMake and re-run ./backends/pytorch/build_ext.sh."
  exit 1
fi

BUILD_DIR="$REPO_ROOT/backends/pytorch/torch_ext/build"
OUT_DIR="$REPO_ROOT/backends/pytorch/torch_ext/sad_ops"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cmake -S "$REPO_ROOT/backends/pytorch/torch_ext" -B "$BUILD_DIR" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$OUT_DIR" \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -j

echo "Built PyTorch extension into: $OUT_DIR"
