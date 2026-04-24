# WebGPU JS

Browser tools backed by the shared WGSL kernels:

- `index.html` - **viewer** for `*_sites.txt` files (renderer-only)
- `train.html` - **trainer** (direct JS port of `backends/webgpu_py/train_wgpu.py`)
- `bench.html` - automated browser benchmark harness for the trainer modules

Both pages use the shared SAD WGSL kernels under `backends/shared/` and hyperparameters from `training_config.json`.

## Requirements

- WebGPU-capable browser (Chrome/Edge, Safari Tech Preview).
- HTTP(S) hosting. ES modules + `fetch` won't run from `file://`; GitHub Pages
  works because it serves the repository over HTTPS:

  ```bash
  python -m http.server 8000 -d /path/to/SAD
  # then open http://localhost:8000/backends/webgpu_js/train.html
  ```

## Viewer (`index.html`)

- Load a `*_sites.txt` file and set width/height to match the target image.
- Pan/zoom, render-mode switcher, interpolation between two sites files, PNG export.

## Trainer (`train.html`)

- Pick a target image (and optional mask / init-sites file).
- Edit hyperparameters in the left panel (they mirror the CLI flags in `train_wgpu.py`; values are pre-filled from `training_config.json`).
- Click **Train**. `viewer-freq` controls visual refreshes, while `log-freq` controls PSNR log lines (`Iter N | PSNR: X dB | Active: a/t | it/s | seconds`).
- **Export PNG** / **Export sites.txt** produce the same formats the Python and Metal trainers write.

## Benchmark (`bench.html`)

Run the browser trainer without manual file picking:

```bash
python -m http.server 8000 -d /path/to/SAD
# then open:
# http://localhost:8000/backends/webgpu_js/bench.html?image=../../test.png
```

The page logs `BENCH_RESULT` with final PSNR, active sites, training time, and total elapsed time.

### Storage-buffer limits

`gradients_tiled.wgsl` and `adam.wgsl` each bind 12 storage buffers per shader stage. The WebGPU default is often lower (8-10), so the trainer requests the adapter's maximum. On Chrome/Mac this is normally 12+ on Metal; if your browser reports fewer, training will fail at pipeline creation. Upgrade the browser or switch backends.

### Module layout

- `train.js` - DOM wiring + entry point.
- `src/shader_loader.js` - shared WGSL loader.
- `src/params.js` - parameter buffer writers, `pack_jump_step`, and `hilbert_bits_for_size`.
- `src/radix_sort.js` - radix-sort GPU pass orchestration.
- `src/encoders.js` - one class per GPU kernel.
- `src/buffers.js` - site/Adam/grad/stats/pairs buffer factories.
- `src/textures.js` - target / mask / candidate texture helpers + image upload.
- `src/io.js` - sites `.txt` / `.json` parsing + serialization + PNG download.
- `src/trainer.js` - training loop, site simulation, target-bpp planning, and epilogue.

## Shader Source

The viewer has `shaders.js` as a fallback with embedded WGSL. After editing any WGSL under `backends/shared/`, regenerate it with:

```bash
python backends/shared/gen_shaders_js.py
```

The trainer fetches WGSL at runtime and has no embedded fallback.
