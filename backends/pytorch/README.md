# PyTorch Backend

High-performance GPU training via the Metal (MPS) or CUDA extension.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CMake 3.26+
- Metal (macOS) or CUDA (Linux/Windows)

## Usage

Install the GPU extension:
```bash
./install.sh --backend pytorch
```

Or install/build it manually:
```bash
python -m pip install -e backends/pytorch/torch_ext --no-build-isolation
```

If pip editable builds fail (PEP517/build isolation), use the repo CMake build helper:
```bash
./backends/pytorch/build_ext.sh
```

Then run with:
```bash
PYTHONPATH=$PWD/backends/pytorch/torch_ext:$PYTHONPATH \
  ./run.sh <image.png> --backend pytorch --target-bpp 2.0
```

Train directly from the repo root:
```bash
python backends/pytorch/train.py <image.png>
```

Or via `run.sh`:
```bash
./run.sh <image.png> --backend pytorch --target-bpp 2.0
```

## Render From Sites

```bash
python backends/pytorch/train.py --render /path/to/sites.txt --width 1024 --height 1024
```

Or via `run.sh`:
```bash
./run.sh --render /path/to/sites.txt --backend pytorch --width 1024 --height 1024
```

Optional PSNR against a target image:
```bash
python backends/pytorch/train.py --render /path/to/sites.txt --render-target /path/to/target.png
```

## Notes

- This backend uses GPU kernels (Metal or CUDA).
- The extension auto-detects Metal/CUDA during build.
- `run.sh --backend pytorch` attempts to build the extension automatically if the compiled ops are missing or stale.
- RTX 50xx (sm_120) requires a nightly CUDA torch wheel (stable torch does not yet support sm_120).
