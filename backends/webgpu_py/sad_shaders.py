"""WGSL shader loader for SAD WebGPU training.

Shader sources live in `backends/shared/sad_shared.wgsl` (common) and
`backends/shared/wgsl/*.wgsl` (per-kernel bodies). This keeps shader code
shared across Python tools and the web viewer, with a single source of truth.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARED_ROOT = ROOT.parent / "shared"
COMMON_PATH = SHARED_ROOT / "sad_shared.wgsl"
WGSL_DIR = SHARED_ROOT / "wgsl"


def _load_common_wgsl() -> str:
    data = COMMON_PATH.read_text()
    start = data.find("// BEGIN COMMON")
    end = data.find("// END COMMON")
    if start == -1 or end == -1:
        raise RuntimeError("Missing COMMON markers in sad_shared.wgsl")
    start += len("// BEGIN COMMON")
    return data[start:end].strip() + "\n"


WGSL_COMMON = _load_common_wgsl()


def _load_shader(snippet_name: str) -> str:
    path = WGSL_DIR / snippet_name
    if not path.is_file():
        raise RuntimeError(f"Missing WGSL snippet: {path}")
    return WGSL_COMMON + path.read_text()


INIT_GRADIENT_SHADER = _load_shader("init_gradient.wgsl")
INIT_CAND_SHADER = _load_shader("init_cand.wgsl")
SEED_CAND_SHADER = _load_shader("seed_cand.wgsl")
VPT_SHADER = _load_shader("vpt.wgsl")
CANDIDATE_PACK_SHADER = _load_shader("candidate_pack.wgsl")
JFA_CLEAR_SHADER = _load_shader("jfa_clear.wgsl")
JFA_FLOOD_SHADER = _load_shader("jfa_flood.wgsl")
RENDER_SHADER = _load_shader("render_compute.wgsl")
GRADIENTS_TILED_SHADER = _load_shader("gradients_tiled.wgsl")
ADAM_SHADER = _load_shader("adam.wgsl")
STATS_SHADER = _load_shader("stats.wgsl")
SPLIT_SHADER = _load_shader("split.wgsl")
PRUNE_SHADER = _load_shader("prune.wgsl")
CLEAR_BUFFER_U32_SHADER = _load_shader("clear_buffer_u32.wgsl")
CLEAR_BUFFER_I32_SHADER = _load_shader("clear_buffer_i32.wgsl")
TAU_EXTRACT_SHADER = _load_shader("tau_extract.wgsl")
TAU_DIFFUSE_SHADER = _load_shader("tau_diffuse.wgsl")
TAU_WRITEBACK_SHADER = _load_shader("tau_writeback.wgsl")
RADIX_SORT_SHADER = _load_shader("radix_sort.wgsl")
SCORE_PAIRS_DENSIFY_SHADER = _load_shader("score_pairs_densify.wgsl")
SCORE_PAIRS_PRUNE_SHADER = _load_shader("score_pairs_prune.wgsl")
WRITE_INDICES_SHADER = _load_shader("write_indices.wgsl")
HILBERT_SHADER = _load_shader("hilbert.wgsl")
