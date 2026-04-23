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

Target a bits-per-pixel budget (16 bytes/site):

```bash
./run.sh --backend metal /path/to/image.png --target-bpp 0.3
```

## Render from Sites

```bash
./run.sh --backend metal --render /path/to/sites.json
```
