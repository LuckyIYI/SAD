# Shared WGSL

This folder holds WGSL sources shared by the WebGPU backends:

- The top-level shared WGSL source contains common structs/helpers and the JS viewer kernels for SAD's soft anisotropic site scoring.
- `wgsl/` contains per-kernel snippets used by the Python WebGPU trainer.

## Regenerate JS Shader Bundle

The browser viewer falls back to `backends/webgpu_js/shaders.js` when it cannot fetch WGSL.
Regenerate that file after WGSL changes:

```bash
python backends/shared/gen_shaders_js.py
```
