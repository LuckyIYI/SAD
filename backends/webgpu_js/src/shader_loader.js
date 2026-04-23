// Direct port of backends/webgpu_py/sad_shaders.py.
// Fetches sad_shared.wgsl (the COMMON block between // BEGIN COMMON / // END COMMON)
// and concatenates it with each per-kernel snippet from backends/shared/wgsl/*.wgsl.
// Some kernels (gradients_tiled, adam, tau_extract, tau_writeback) are replaced
// by JS-specific overrides that consolidate the 10 per-channel gradient buffers
// into a single interleaved buffer — required because the WebGPU per-stage
// storage-buffer limit is 10 on common adapters and the originals use 12.

import {
  GRADIENTS_TILED_OVERRIDE,
  ADAM_OVERRIDE,
  TAU_EXTRACT_OVERRIDE,
  TAU_WRITEBACK_OVERRIDE,
  RENDER_HASHED_OVERRIDE,
  RENDER_TAU_HEATMAP_OVERRIDE,
} from "./wgsl_overrides.js";

const COMMON_URL = new URL("../../shared/sad_shared.wgsl", import.meta.url);
const WGSL_DIR_URL = new URL("../../shared/wgsl/", import.meta.url);

const SHADER_FILES = [
  "init_gradient",
  "init_cand",
  "seed_cand",
  "vpt",
  "candidate_pack",
  "jfa_clear",
  "jfa_flood",
  "render_compute",
  "gradients_tiled",
  "adam",
  "stats",
  "split",
  "prune",
  "clear_buffer_u32",
  "clear_buffer_i32",
  "tau_extract",
  "tau_diffuse",
  "tau_writeback",
  "score_pairs_densify",
  "score_pairs_prune",
  "write_indices",
  "hilbert",
  "radix_sort",
];

let _cache = null;

async function fetchText(url) {
  // Cache-bust so local shader edits always take effect on reload.
  const bust = new URL(url);
  bust.searchParams.set("v", Date.now().toString());
  const response = await fetch(bust, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return await response.text();
}

function extractCommon(source) {
  const beginMarker = "// BEGIN COMMON";
  const endMarker = "// END COMMON";
  const start = source.indexOf(beginMarker);
  const end = source.indexOf(endMarker);
  if (start === -1 || end === -1) {
    throw new Error("Missing COMMON markers in sad_shared.wgsl");
  }
  return source.slice(start + beginMarker.length, end).trim() + "\n";
}

export async function loadShaders() {
  if (_cache) return _cache;

  const commonSource = await fetchText(COMMON_URL);
  const common = extractCommon(commonSource);

  const snippets = await Promise.all(
    SHADER_FILES.map((name) =>
      fetchText(new URL(`${name}.wgsl`, WGSL_DIR_URL)).then((text) => [name, text])
    )
  );

  const out = { COMMON: common };
  for (const [name, body] of snippets) {
    out[name] = common + body;
  }

  _cache = {
    INIT_GRADIENT: out.init_gradient,
    INIT_CAND: out.init_cand,
    SEED_CAND: out.seed_cand,
    VPT: out.vpt,
    CANDIDATE_PACK: out.candidate_pack,
    JFA_CLEAR: out.jfa_clear,
    JFA_FLOOD: out.jfa_flood,
    RENDER: out.render_compute,
    GRADIENTS_TILED: common + GRADIENTS_TILED_OVERRIDE,
    ADAM: common + ADAM_OVERRIDE,
    STATS: out.stats,
    SPLIT: out.split,
    PRUNE: out.prune,
    CLEAR_BUFFER_U32: out.clear_buffer_u32,
    CLEAR_BUFFER_I32: out.clear_buffer_i32,
    TAU_EXTRACT: common + TAU_EXTRACT_OVERRIDE,
    TAU_DIFFUSE: out.tau_diffuse,
    TAU_WRITEBACK: common + TAU_WRITEBACK_OVERRIDE,
    SCORE_PAIRS_DENSIFY: out.score_pairs_densify,
    SCORE_PAIRS_PRUNE: out.score_pairs_prune,
    WRITE_INDICES: out.write_indices,
    HILBERT: out.hilbert,
    RADIX_SORT: out.radix_sort,
    RENDER_HASHED: common + RENDER_HASHED_OVERRIDE,
    RENDER_TAU_HEATMAP: common + RENDER_TAU_HEATMAP_OVERRIDE,
  };
  return _cache;
}
