#!/bin/bash
set -euo pipefail

if [ "$(uname)" != "Darwin" ]; then
    echo "The Metal backend can only be built on macOS."
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_DIR="$ROOT_DIR/backends/metal/sources"
SHADER_DIR="$ROOT_DIR/backends/metal/shaders"
BUILD_DIR="$ROOT_DIR/build/metal"

mkdir -p "$BUILD_DIR"

xcrun metal -O3 -I "$SHADER_DIR" \
    -c "$SHADER_DIR/sad.metal" \
    -o "$BUILD_DIR/sad.air"

xcrun metal -O3 -I "$SHADER_DIR" \
    -c "$SHADER_DIR/radix_sort.metal" \
    -o "$BUILD_DIR/radix_sort.air"

xcrun metallib \
    "$BUILD_DIR/sad.air" \
    "$BUILD_DIR/radix_sort.air" \
    -o "$BUILD_DIR/sad.metallib"

swift_sources=()
while IFS= read -r source_file; do
    swift_sources+=("$source_file")
done < <(find "$SOURCE_DIR" -type f -name '*.swift' | sort)

swiftc -O \
    -o "$BUILD_DIR/sad" \
    "${swift_sources[@]}"

echo "Built Metal backend into $BUILD_DIR"
