# CUDA Backend

C++/CUDA implementation of SAD training and rendering.

## Requirements

- CUDA Toolkit (nvcc)
- CMake 3.20+
- A CUDA-capable GPU

## Build

```bash
./backends/cuda/build.sh
```

Binary is written to `backends/cuda/bin/sad_cuda`.

## Train

```bash
./backends/cuda/bin/sad_cuda /path/to/image.png --out-dir results
```

Target a bits-per-pixel budget (16 bytes/site):

```bash
./backends/cuda/bin/sad_cuda /path/to/image.png --target-bpp 0.3 --out-dir results
```

## Render From Sites

```bash
./backends/cuda/bin/sad_cuda --render /path/to/sites.txt --width 1024 --height 1024 --out-dir results
```

## Notes

- Uses `training_config.json` from the repo root (required).
- Exports trained sites as TXT. Render-only mode accepts TXT or JSON site files.
