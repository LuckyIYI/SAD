#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

backend="auto"
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --backend=*)
            backend="${1#*=}"
            shift
            ;;
        --backend)
            backend="$2"
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done
if [ ${#args[@]} -gt 0 ]; then
    set -- "${args[@]}"
else
    set --
fi

case "$backend" in
    webgpu)
        backend="wgpu"
        ;;
esac

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

print_usage() {
    cat <<'EOF'
Usage:
  ./install.sh --backend auto
  ./install.sh --backend metal
  ./install.sh --backend cuda
  ./install.sh --backend wgpu
  ./install.sh --backend pytorch

Notes:
  auto prefers Metal on macOS, CUDA elsewhere, then WGPU, then PyTorch.
  wgpu/pytorch install Python packages with python3 -m pip install --user.
EOF
}

for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        print_usage
        exit 0
    fi
done

install_cuda() {
    if ! have_cmd nvcc; then
        echo "CUDA toolkit not found (nvcc missing)."
        echo "Install CUDA and ensure nvcc is on PATH, then re-run ./install.sh --backend cuda"
        return 1
    fi
    if ! have_cmd cmake; then
        echo "CMake not found. Install CMake 3.20+ and re-run ./install.sh --backend cuda"
        return 1
    fi
    "$ROOT_DIR/backends/cuda/build.sh" || return 1
}

install_metal() {
    if [ "$(uname)" != "Darwin" ]; then
        echo "Metal backend requires macOS."
        return 1
    fi
    "$ROOT_DIR/backends/metal/build.sh" || return 1
}

ensure_python_pip() {
    if ! have_cmd python3; then
        echo "Python3 not found. Install Python3 and re-run."
        return 1
    fi
    if ! python3 -m pip --version >/dev/null 2>&1; then
        echo "pip not found for Python3. Install pip and re-run."
        return 1
    fi
}

install_wgpu() {
    ensure_python_pip || return 1
    python3 -m pip install --user -U -r "$ROOT_DIR/backends/webgpu_py/requirements.txt" || return 1
    python3 - <<'PY' >/dev/null 2>&1 || return 1
import numpy  # noqa: F401
from PIL import Image  # noqa: F401
import wgpu  # noqa: F401
PY
}

install_pytorch() {
    ensure_python_pip || return 1
    if ! have_cmd cmake; then
        echo "CMake not found. Install CMake 3.26+ and re-run ./install.sh --backend pytorch"
        return 1
    fi
    python3 -m pip install --user -U -r "$ROOT_DIR/backends/pytorch/requirements.txt" || return 1
    PYTHON=python3 "$ROOT_DIR/backends/pytorch/build_ext.sh" || return 1
}

if [ "$backend" = "auto" ]; then
    if [ "$(uname)" = "Darwin" ]; then
        if install_metal; then
            echo "Installed Metal backend."
            exit 0
        fi
        if install_wgpu; then
            echo "Installed WGPU backend."
            exit 0
        fi
    else
        if install_cuda; then
            echo "Installed CUDA backend."
            exit 0
        fi
        if install_wgpu; then
            echo "Installed WGPU backend."
            exit 0
        fi
    fi
    if install_pytorch; then
        echo "Installed PyTorch backend."
        exit 0
    fi
    echo "No backend could be installed."
    exit 1
fi

if [ "$backend" = "cuda" ]; then
    install_cuda
    exit 0
fi

if [ "$backend" = "metal" ]; then
    install_metal
    exit 0
fi

if [ "$backend" = "wgpu" ]; then
    install_wgpu
    exit 0
fi

if [ "$backend" = "pytorch" ]; then
    install_pytorch
    exit 0
fi

echo "Unknown backend: $backend (expected 'auto', 'cuda', 'metal', 'wgpu', or 'pytorch'; 'webgpu' is also accepted)"
exit 1
