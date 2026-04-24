# SAD Metal Backend

Swift + Metal implementation for training and rendering.

## Build

From the repo root:

```bash
./build.sh
```

This writes the Metal executable and shader artifacts into `build/metal/`.

## Train

```bash
./run.sh --backend metal /path/to/image.png
```

Default training output is `<image>.png` and `<image>_sites.txt`. Add
`--include-debug-mask` to also write `<image>_cells.png` and
`<image>_tau_heatmap.png`.

Target a bits-per-pixel budget (16 bytes/site):

```bash
./run.sh --backend metal /path/to/image.png --target-bpp 0.3
```

## Render from Sites

```bash
./run.sh --backend metal --render /path/to/sites.txt
```
