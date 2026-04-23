# WGPU Python Backend (wgpu-py)

Python WebGPU training and evaluation using the shared WGSL kernels.

## Requirements

- Python 3.9+ (3.10+ recommended)
- WebGPU-capable GPU/driver (Metal on macOS, DX12/Vulkan on Windows/Linux)

Install:

```bash
./install.sh --backend wgpu
# or manually:
python -m pip install --user -r backends/webgpu_py/requirements.txt
```

Notes:
- `training_config.json` at the repo root is required and must define `INIT_LOG_TAU` and `INIT_RADIUS`.

## Train

```bash
python backends/webgpu_py/train_wgpu.py /path/to/image.png --out-dir results
# or
./run.sh /path/to/image.png --backend wgpu
```

Use Hilbert probe candidates to hybridize VPT:

```bash
python backends/webgpu_py/train_wgpu.py /path/to/image.png --cand-hilbert-window 32 --cand-hilbert-probes 4
```

Target a bits-per-pixel budget (16 bytes/site):

```bash
python backends/webgpu_py/train_wgpu.py /path/to/image.png --target-bpp 0.3 --out-dir results
```

Override candidate refinement or learning-rate defaults:

```bash
python backends/webgpu_py/train_wgpu.py /path/to/image.png --cand-passes 2 --cand-radius-scale 64 --lr 0.5
```

Initialize from an existing sites file:

```bash
python backends/webgpu_py/train_wgpu.py /path/to/image.png --init-from-sites /path/to/sites.txt
```

## Render From Sites

```bash
python backends/webgpu_py/train_wgpu.py --render /path/to/sites.txt --width 1024 --height 1024
# or
./run.sh --render /path/to/sites.txt --backend wgpu --width 1024 --height 1024
```

Optional PSNR against a target image:

```bash
python backends/webgpu_py/train_wgpu.py --render /path/to/sites.txt --render-target /path/to/target.png
```
