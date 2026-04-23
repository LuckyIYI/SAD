# SAD Backends

SAD ships multiple backends with shared WGSL kernels.

- `metal/` — Swift + Metal training and rendering (macOS).
- `cuda/` — C++/CUDA training and rendering (NVIDIA GPUs).
- `pytorch/` — PyTorch MPS/CUDA training and render (GPU extension).
- `webgpu_js/` — Browser viewer and interactive trainer demo (WebGPU).
- `webgpu_py/` — Python training and rendering via wgpu-py (`--backend wgpu`).
- `shared/` — WGSL sources shared by WebGPU backends.

Use `./run.sh --backend metal`, `./run.sh --backend cuda`, `./run.sh --backend wgpu`, or `./run.sh --backend pytorch` from the repo root.

Render-only mode is available for all backends via:
```bash
./run.sh --render /path/to/sites.txt --backend <metal|cuda|wgpu|pytorch> --width 1024 --height 1024
```
