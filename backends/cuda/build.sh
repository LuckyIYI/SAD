#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"

if ! command -v cmake >/dev/null 2>&1; then
    echo "CMake not found. Install CMake and re-run ./backends/cuda/build.sh."
    exit 1
fi

mkdir -p "$BUILD_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j
