# SAD: Soft Anisotropic Diagrams

Code for the paper _Soft Anisotropic Diagrams for Differentiable Image Representation_, with Metal, WGPU, CUDA, PyTorch, and browser backends.

Project page: https://luckyiyi.github.io/SAD/

## Run

```bash
./run.sh /path/to/image.png
./run.sh
./run.sh /path/to/image.png --backend auto
./run.sh /path/to/image.png --backend metal
./run.sh /path/to/image.png --backend cuda
./run.sh /path/to/image.png --backend wgpu
./run.sh /path/to/image.png --backend pytorch --target-bpp 2.0
```

Backend options: `auto`, `metal`, `wgpu`, `cuda`, `pytorch`.
Compatibility note: `--backend webgpu` is still accepted and maps to `wgpu`.
Default training targets `DEFAULT_TARGET_BPP=1.0` from `training_config.json`; pass `--target-bpp` to override it.
If no image path is passed, `run.sh` uses the included `test.png` smoke-test image.

## Install

`run.sh` is the preferred entry point on a new machine. It builds native
backends automatically when sources changed and, if no usable backend is found,
bootstraps Python backend dependencies through `install.sh`. Set
`SAD_NO_AUTO_INSTALL=1` to disable automatic Python package installs.

Manual install is still available:

```bash
./install.sh --backend auto
```

Or install a specific backend:

```bash
./install.sh --backend metal
./install.sh --backend cuda
./install.sh --backend wgpu
./install.sh --backend pytorch
```

Notes:
- Metal requires macOS + Xcode command line tools (`xcode-select --install`).
- `./build.sh` and `./install.sh --backend metal` build the Metal executable into `build/metal/`.
- CUDA requires a CUDA toolkit with `nvcc` on PATH and CMake 3.20+.
- WGPU (wgpu-py) installs Python deps from `backends/webgpu_py/requirements.txt`.
- PyTorch requires Python 3.9+, PyTorch, CMake 3.26+, and Metal or CUDA.

## Render From Saved Sites

All backends support render-only mode from a saved `*_sites.txt` or `*.json`:

```bash
./run.sh --render /path/to/sites.txt --backend metal
./run.sh --render /path/to/sites.txt --backend cuda --width 1024 --height 1024
./run.sh --render /path/to/sites.txt --backend wgpu --width 1024 --height 1024
./run.sh --render /path/to/sites.txt --backend pytorch --width 1024 --height 1024
```

Metal also has a convenience script:

```bash
./render.sh results/foo_sites.json
./render.sh results/foo_sites.txt --width 1024 --height 1024
```

## Viewer

Open `backends/webgpu_js/index.html` in a WebGPU-capable browser, load a `*_sites.txt`,
set width/height, and inspect the forward pass with debug overlays.

## Browser Demo

The hosted demo is available from the project page: https://luckyiyi.github.io/SAD/

## Backends

Each backend has its own README with setup, build, and usage details:

- `backends/metal/README.md` (Swift + Metal)
- `backends/cuda/README.md` (C++/CUDA)
- `backends/pytorch/README.md` (PyTorch MPS/CUDA extension)
- `backends/webgpu_py/README.md` (wgpu-py training)
- `backends/webgpu_js/README.md` (browser viewer, trainer, and demo)

See `backends/README.md` for an overview.
