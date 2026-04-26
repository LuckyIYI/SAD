#!/usr/bin/env bash
set -euo pipefail

# Main entrypoint for SAD training and render backends.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

METAL_BUILD_DIR="$ROOT_DIR/build/metal"
METAL_BIN="$METAL_BUILD_DIR/sad"
METAL_LIB="$METAL_BUILD_DIR/sad.metallib"
CUDA_BIN="$ROOT_DIR/backends/cuda/bin/sad_cuda"
WGPU_SCRIPT="$ROOT_DIR/backends/webgpu_py/train_wgpu.py"
PYTORCH_SCRIPT="$ROOT_DIR/backends/pytorch/train.py"
DEFAULT_IMAGE="$ROOT_DIR/test.png"

export CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/training_config.json}"

backend="auto"
backend_set=false
args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --backend=*)
            backend="${1#*=}"
            backend_set=true
            shift
            ;;
        --backend)
            if [ $# -lt 2 ]; then
                echo "Missing value for --backend"
                exit 1
            fi
            backend="$2"
            backend_set=true
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

case "$backend" in
    webgpu) backend="wgpu" ;;
esac

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

print_usage() {
    cat <<'EOF'
Usage:
  ./run.sh <image.png> [--backend auto|metal|cuda|wgpu|pytorch] [backend options]
  ./run.sh --render <sites.txt> --backend <metal|cuda|wgpu|pytorch> [render options]

Defaults:
  Uses training_config.json from the repo root.
  If no image is passed, uses test.png only when it exists at the repo root.
  Native backends are built automatically when source files changed.
  Missing Python backend dependencies are installed automatically unless SAD_NO_AUTO_INSTALL=1.
EOF
}

if [ ${#args[@]} -eq 1 ] && { [ "${args[0]}" = "--help" ] || [ "${args[0]}" = "-h" ]; } && [ "$backend_set" = false ]; then
    print_usage
    exit 0
fi

help_requested=false
for arg in ${args[@]+"${args[@]}"}; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        help_requested=true
        break
    fi
done

cuda_available() {
    if [ -x "$CUDA_BIN" ]; then
        return 0
    fi
    have_cmd nvcc && have_cmd cmake
}

metal_available() {
    if [ -x "$METAL_BIN" ] && [ -f "$METAL_LIB" ]; then
        return 0
    fi
    [ "$(uname)" = "Darwin" ] && have_cmd xcrun && have_cmd swiftc
}

wgpu_available() {
    have_cmd python3 || return 1
    python3 - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
from PIL import Image  # noqa: F401
import wgpu  # noqa: F401
PY
}

pytorch_available() {
    have_cmd python3 || return 1
    python3 - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
from PIL import Image  # noqa: F401
import torch
if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
    raise SystemExit(1)
PY
}

bootstrap_backend() {
    local name="$1"
    if [ "${SAD_NO_AUTO_INSTALL:-0}" = "1" ]; then
        return 1
    fi
    echo "Bootstrapping $name backend..."
    "$ROOT_DIR/install.sh" --backend "$name"
}

select_auto_backend() {
    if [ "$(uname)" = "Darwin" ]; then
        if metal_available; then echo "metal"; return 0; fi
        if wgpu_available; then echo "wgpu"; return 0; fi
        if pytorch_available; then echo "pytorch"; return 0; fi
        if cuda_available; then echo "cuda"; return 0; fi
    else
        if cuda_available; then echo "cuda"; return 0; fi
        if wgpu_available; then echo "wgpu"; return 0; fi
        if pytorch_available; then echo "pytorch"; return 0; fi
        if metal_available; then echo "metal"; return 0; fi
    fi
    return 1
}

ensure_metal_built() {
    local need_build=false
    if [ "$(uname)" != "Darwin" ]; then
        echo "Metal backend requires macOS."
        exit 1
    fi
    if [ ! -x "$METAL_BIN" ] || [ ! -f "$METAL_LIB" ]; then
        need_build=true
    elif find "$ROOT_DIR/backends/metal/sources" -type f -name '*.swift' -newer "$METAL_BIN" | grep -q .; then
        need_build=true
    elif find "$ROOT_DIR/backends/metal/shaders" -type f -name '*.metal' -newer "$METAL_LIB" | grep -q .; then
        need_build=true
    fi
    if [ "$need_build" = true ]; then
        "$ROOT_DIR/backends/metal/build.sh"
    fi
}

ensure_cuda_built() {
    local need_build=false
    if [ ! -x "$CUDA_BIN" ]; then
        need_build=true
    elif find "$ROOT_DIR/backends/cuda/src" -type f -newer "$CUDA_BIN" | grep -q .; then
        need_build=true
    fi
    if [ "$need_build" = true ]; then
        if ! have_cmd nvcc; then
            echo "CUDA toolkit not found (nvcc missing)."
            exit 1
        fi
        if ! have_cmd cmake; then
            echo "CMake not found. Install CMake 3.20+ for the CUDA backend."
            exit 1
        fi
        "$ROOT_DIR/backends/cuda/build.sh"
    fi
}

if [ "$backend" = "auto" ]; then
    if [ "$help_requested" = true ]; then
        print_usage
        exit 0
    fi
    if ! selected_backend="$(select_auto_backend)"; then
        bootstrap_backend auto || {
            echo "No usable backend found."
            echo "Install CUDA, Xcode command line tools, or Python3+pip; or run ./install.sh --backend <backend>."
            exit 1
        }
        selected_backend="$(select_auto_backend)" || {
            echo "No usable backend found after bootstrap."
            exit 1
        }
    fi
    backend="$selected_backend"
    echo "Auto-selected backend: $backend"
fi

run_backend() {
    case "$backend" in
        metal) "$METAL_BIN" "$@" ;;
        cuda) "$CUDA_BIN" "$@" ;;
        wgpu) python3 "$WGPU_SCRIPT" "$@" ;;
        pytorch) python3 "$PYTORCH_SCRIPT" "$@" ;;
    esac
}

if [ "$help_requested" = true ]; then
    case "$backend" in
        metal) ensure_metal_built ;;
        cuda) ensure_cuda_built ;;
        wgpu|pytorch) ;;
        *)
            echo "Unknown backend: $backend (expected auto, metal, cuda, wgpu, or pytorch; webgpu is also accepted)"
            exit 1
            ;;
    esac
    run_backend "${args[@]}"
    exit 0
fi

case "$backend" in
    metal)
        ensure_metal_built
        ;;
    cuda)
        ensure_cuda_built
        ;;
    wgpu)
        if ! wgpu_available; then
            bootstrap_backend wgpu || exit 1
        fi
        ;;
    pytorch)
        if ! pytorch_available; then
            bootstrap_backend pytorch || exit 1
        fi
        ;;
    *)
        echo "Unknown backend: $backend (expected auto, metal, cuda, wgpu, or pytorch; webgpu is also accepted)"
        exit 1
        ;;
esac

if [ "${args[0]:-}" = "--render" ]; then
    run_backend "${args[@]}"
    exit 0
fi

if [ ${#args[@]} -eq 0 ] || [[ "${args[0]}" == --* ]]; then
    if [ ! -f "$DEFAULT_IMAGE" ]; then
        echo "Default image not found: $DEFAULT_IMAGE"
        echo "Run: ./run.sh /path/to/image.png [flags...]"
        exit 1
    fi
    input_path="$DEFAULT_IMAGE"
    extra=(${args[@]+"${args[@]}"})
else
    input_path="${args[0]}"
    extra=("${args[@]:1}")
fi

batch_log=""
if [ -d "$input_path" ]; then
    images=()
    for ext in png jpg jpeg PNG JPG JPEG; do
        while IFS= read -r -d '' file; do
            images+=("$file")
        done < <(find "$input_path" -maxdepth 1 -type f -name "*.$ext" -print0 2>/dev/null)
    done

    if [ ${#images[@]} -eq 0 ]; then
        echo "No image files found in directory: $input_path"
        echo "Supported formats: png, jpg, jpeg"
        exit 1
    fi

    out_dir="results"
    for ((i=0; i<${#extra[@]}; i++)); do
        if [ "${extra[$i]}" = "--out-dir" ] && [ $((i + 1)) -lt ${#extra[@]} ]; then
            out_dir="${extra[$((i + 1))]}"
            break
        elif [[ "${extra[$i]}" == --out-dir=* ]]; then
            out_dir="${extra[$i]#*=}"
            break
        fi
    done

    mkdir -p "$out_dir"
    batch_log="$(cd "$out_dir" && pwd)/batch_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "Batch processing log - $(date)"
        echo "Input directory: $input_path"
        echo "Output directory: $out_dir"
        echo "Backend: $backend"
        echo "Total images: ${#images[@]}"
        echo "==========================================="
    } > "$batch_log"
    echo "Found ${#images[@]} image(s) in $input_path"
    echo "Batch log: $batch_log"
else
    images=("$input_path")
fi

run_single() {
    local img="$1"
    local img_basename
    img_basename="$(basename "$img")"
    local start_time
    start_time="$(date +%s)"
    local temp_output=""
    local exit_code=0

    echo "Processing: $img"

    set +e
    if [ -n "$batch_log" ]; then
        temp_output="$(mktemp)"
        run_backend "$img" ${extra[@]+"${extra[@]}"} 2>&1 | tee "$temp_output"
        exit_code=${PIPESTATUS[0]}
    else
        run_backend "$img" ${extra[@]+"${extra[@]}"}
        exit_code=$?
    fi
    set -e

    local end_time
    end_time="$(date +%s)"
    local duration=$((end_time - start_time))

    if [ -n "$batch_log" ]; then
        {
            echo ""
            echo "Image: $img_basename"
            echo "Duration: ${duration}s"
        } >> "$batch_log"

        local psnr sites
        psnr="$(grep -i "psnr" "$temp_output" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
        sites="$(grep -i "active" "$temp_output" | tail -1 | grep -oE '[0-9]+/[0-9]+' | head -1 || true)"
        [ -n "$psnr" ] && echo "PSNR: $psnr dB" >> "$batch_log"
        [ -n "$sites" ] && echo "Active sites: $sites" >> "$batch_log"

        if [ "$exit_code" -eq 0 ]; then
            echo "Status: SUCCESS" >> "$batch_log"
        else
            echo "Status: FAILED (exit code: $exit_code)" >> "$batch_log"
        fi
        echo "-------------------------------------------" >> "$batch_log"
        rm -f "$temp_output"
    fi

    return "$exit_code"
}

for img in "${images[@]}"; do
    if ! run_single "$img"; then
        echo "Error processing $img"
        if [ -n "$batch_log" ]; then
            {
                echo ""
                echo "BATCH PROCESSING FAILED at image: $(basename "$img")"
                echo "Batch log saved to: $batch_log"
            } >> "$batch_log"
        fi
        exit 1
    fi
    echo ""
done

if [ -n "$batch_log" ]; then
    echo "Batch processing complete."
    echo "Processed ${#images[@]} image(s) successfully."
    echo "Batch log: $batch_log"
fi
