#!/usr/bin/env python3
"""Train SAD sites with wgpu (tiled gradient path)."""

import argparse
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import wgpu
from wgpu.utils import get_default_device

from encoders import (
    InitGradientEncoder,
    CandidatesEncoder,
    SeedCandidatesEncoder,
    JFAClearEncoder,
    JFAFloodEncoder,
    CandidatePackEncoder,
    RenderEncoder,
    GradientsEncoder,
    AdamEncoder,
    TauEncoder,
    StatsEncoder,
    SplitEncoder,
    PruneEncoder,
    ClearBufferEncoder,
    ScorePairsEncoder,
    WriteIndicesEncoder,
    HilbertEncoder,
)
from radix_sort import RadixSortUInt2
from sad_shared import (
    pack_jump_step,
    write_texture_rgba32float,
    create_params_buffer,
    update_params_buffer,
    create_clear_params_buffer,
    read_texture_rgba32float,
    create_hilbert_params_buffer,
    update_hilbert_params_buffer,
)
SITE_FLOATS = 10
ADAM_FLOATS = 24
BITS_PER_SITE = 16.0 * 8.0
PACKED_CAND_BYTES = 16

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", Path(__file__).resolve().parents[2] / "training_config.json"))


def _load_training_config(path: Path) -> dict:
    defaults = {
        "DEFAULT_SITES": 65536,
        "DEFAULT_MAX_SITES": 70000,
        "DEFAULT_ITERS": 2000,
        "DEFAULT_TARGET_BPP": -1.0,
        "PRUNE_PERCENTILE": 0.01,
        "PRUNE_DURING_DENSIFY": True,
        "PRUNE_START": 100,
        "PRUNE_FREQ": 20,
        "PRUNE_END": 3600,
        "DENSIFY_START": 20,
        "DENSIFY_FREQ": 20,
        "DENSIFY_END": 3500,
        "DENSIFY_PERCENTILE": 0.01,
        "DENSIFY_SCORE_ALPHA": 0.7,
        "CAND_UPDATE_FREQ": 1,
        "CAND_UPDATE_PASSES": 1,
        "CAND_RADIUS_SCALE": 64.0,
        "CAND_RADIUS_PROBES": 0,
        "CAND_DOWNSCALE": 1,
        "CAND_INJECT_COUNT": 16,
        "CAND_HILBERT_WINDOW": 0,
        "CAND_HILBERT_PROBES": 0,
        "LR_POS_BASE": 0.05,
        "LR_TAU_BASE": 0.01,
        "LR_RADIUS_BASE": 0.02,
        "LR_COLOR_BASE": 0.02,
        "LR_DIR_BASE": 0.02,
        "LR_ANISO_BASE": 0.02,
        "BETA1": 0.9,
        "BETA2": 0.999,
        "EPS": 1e-8,
        "MAX_DIM": 2048,
    }
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                defaults[key] = value
        except (OSError, json.JSONDecodeError):
            pass
    return defaults


def _adapter_summary(device) -> str:
    info = getattr(getattr(device, "adapter", None), "info", {}) or {}
    if isinstance(info, dict):
        for key in ("device", "adapter", "description"):
            value = info.get(key)
            if value:
                return str(value)
        vendor = info.get("vendor")
        if vendor:
            return str(vendor)
    return str(info) if info else "unknown"


def _init_summary(args) -> str:
    if args.init_from_sites:
        return "sites"
    return "gradient"


def _print_training_overview(args, device, width, height, actual_sites, active_sites, plan, cand_downscale):
    bpp = f" | target-bpp={args.target_bpp:.3f}" if args.target_bpp > 0.0 else ""
    print(
        f"Training | backend=wgpu | device={_adapter_summary(device)} | image={width}x{height} | "
        f"sites={active_sites}/{actual_sites} | iters={args.iters} | "
        f"log-freq={max(1, args.log_freq)} | mask={'yes' if args.mask else 'no'} | "
        f"out={args.out_dir}{bpp}"
    )
    prune_end = args.prune_end if args.prune_end > 0 else args.iters - 1
    prune = (
        f"on {args.prune_percentile:.3f} @{args.prune_start}-{prune_end}/{max(1, args.prune_freq)}"
        if args.prune_percentile > 0.0
        else "off"
    )
    densify = f"on cap={plan.max_sites_capacity}" if args.densify else "off"
    hilbert = (
        f" hilbert={args.cand_hilbert_probes}x{args.cand_hilbert_window}"
        if args.cand_hilbert_probes > 0 and args.cand_hilbert_window > 0
        else ""
    )
    print(
        f"Schedule | init={_init_summary(args)} | densify={densify} | prune={prune} | "
        f"cand=freq {args.cand_freq}, passes {args.cand_passes}, downscale {cand_downscale}x{hilbert}"
    )


CONFIG = _load_training_config(CONFIG_PATH)

DEFAULT_SITES = int(CONFIG["DEFAULT_SITES"])
DEFAULT_MAX_SITES = int(CONFIG["DEFAULT_MAX_SITES"])
DEFAULT_ITERS = int(CONFIG["DEFAULT_ITERS"])
DEFAULT_TARGET_BPP = float(CONFIG["DEFAULT_TARGET_BPP"])

PRUNE_PERCENTILE = float(CONFIG["PRUNE_PERCENTILE"])
PRUNE_START = int(CONFIG["PRUNE_START"])
PRUNE_FREQ = int(CONFIG["PRUNE_FREQ"])
PRUNE_END = int(CONFIG["PRUNE_END"])

DENSIFY_START = int(CONFIG["DENSIFY_START"])
DENSIFY_FREQ = int(CONFIG["DENSIFY_FREQ"])
DENSIFY_END = int(CONFIG["DENSIFY_END"])
DENSIFY_PERCENTILE = float(CONFIG["DENSIFY_PERCENTILE"])
DENSIFY_SCORE_ALPHA = float(CONFIG["DENSIFY_SCORE_ALPHA"])

CAND_UPDATE_FREQ = int(CONFIG["CAND_UPDATE_FREQ"])
CAND_UPDATE_PASSES = int(CONFIG["CAND_UPDATE_PASSES"])
CAND_RADIUS_SCALE = float(CONFIG["CAND_RADIUS_SCALE"])
CAND_RADIUS_PROBES = int(CONFIG["CAND_RADIUS_PROBES"])
CAND_DOWNSCALE = int(CONFIG["CAND_DOWNSCALE"])
CAND_INJECT_COUNT = int(CONFIG["CAND_INJECT_COUNT"])
CAND_HILBERT_WINDOW = int(CONFIG["CAND_HILBERT_WINDOW"])
CAND_HILBERT_PROBES = int(CONFIG["CAND_HILBERT_PROBES"])

LR_POS_BASE = float(CONFIG["LR_POS_BASE"])
LR_TAU_BASE = float(CONFIG["LR_TAU_BASE"])
LR_RADIUS_BASE = float(CONFIG["LR_RADIUS_BASE"])
LR_COLOR_BASE = float(CONFIG["LR_COLOR_BASE"])
LR_DIR_BASE = float(CONFIG["LR_DIR_BASE"])
LR_ANISO_BASE = float(CONFIG["LR_ANISO_BASE"])
if "INIT_LOG_TAU" not in CONFIG or "INIT_RADIUS" not in CONFIG:
    raise RuntimeError("training_config.json missing INIT_LOG_TAU/INIT_RADIUS (required).")
INIT_LOG_TAU = float(CONFIG["INIT_LOG_TAU"])
INIT_RADIUS = float(CONFIG["INIT_RADIUS"])
BETA1 = float(CONFIG["BETA1"])
BETA2 = float(CONFIG["BETA2"])
EPS = float(CONFIG["EPS"])

MAX_DIM = int(CONFIG["MAX_DIM"])
TAU_DIFFUSE_PASSES = 4
TAU_DIFFUSE_LAMBDA = 0.05




def load_image(path: str, max_dim: int = MAX_DIM):
    img = Image.open(path).convert("RGBA")
    width, height = img.size
    max_side = max(width, height)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        width, height = img.size
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, width, height


def load_mask(path: str, width: int, height: int):
    img = Image.open(path).convert("L")
    if img.size != (width, height):
        img = img.resize((width, height), Image.NEAREST)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mask = (arr > 0.0).astype(np.float32)
    mask_sum = float(np.count_nonzero(mask))
    return mask, mask_sum


def create_grad_params_buffer(device, site_count, inv_scale_sq, compute_removal):
    data = struct.pack(
        "<4I4f",
        int(site_count),
        int(compute_removal),
        0,
        0,
        float(inv_scale_sq),
        0.0,
        0.0,
        0.0,
    )
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def update_grad_params_buffer(queue, buffer, site_count, inv_scale_sq, compute_removal):
    data = struct.pack(
        "<4I4f",
        int(site_count),
        int(compute_removal),
        0,
        0,
        float(inv_scale_sq),
        0.0,
        0.0,
        0.0,
    )
    queue.write_buffer(buffer, 0, data)


def create_adam_params_buffer(device, lr_pos, lr_tau, lr_radius, lr_color, lr_dir,
                              lr_aniso, beta1, beta2, eps, t, width, height):
    data = struct.pack(
        "<9f4I",
        float(lr_pos),
        float(lr_tau),
        float(lr_radius),
        float(lr_color),
        float(lr_dir),
        float(lr_aniso),
        float(beta1),
        float(beta2),
        float(eps),
        int(t),
        int(width),
        int(height),
        0,
    )
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def update_adam_params_buffer(queue, buffer, lr_pos, lr_tau, lr_radius, lr_color, lr_dir,
                              lr_aniso, beta1, beta2, eps, t, width, height):
    data = struct.pack(
        "<9f4I",
        float(lr_pos),
        float(lr_tau),
        float(lr_radius),
        float(lr_color),
        float(lr_dir),
        float(lr_aniso),
        float(beta1),
        float(beta2),
        float(eps),
        int(t),
        int(width),
        int(height),
        0,
    )
    queue.write_buffer(buffer, 0, data)



def create_split_params_buffer(device, num_to_split, current_site_count):
    data = struct.pack("<IIII", int(num_to_split), int(current_site_count), 0, 0)
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def create_prune_params_buffer(device, count):
    data = struct.pack("<IIII", int(count), 0, 0, 0)
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def create_tau_params_buffer(device, site_count, lambda_value, cand_downscale):
    data = struct.pack("<4I4f", int(site_count), int(cand_downscale), 0, 0, float(lambda_value), 0.0, 0.0, 0.0)
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def update_tau_params_buffer(queue, buffer, site_count, lambda_value, cand_downscale):
    data = struct.pack("<4I4f", int(site_count), int(cand_downscale), 0, 0, float(lambda_value), 0.0, 0.0, 0.0)
    queue.write_buffer(buffer, 0, data)


def update_clear_params_buffer(queue, buffer, count):
    data = struct.pack("<IIII", int(count), 0, 0, 0)
    queue.write_buffer(buffer, 0, data)


def update_split_params_buffer(queue, buffer, num_to_split, current_site_count):
    data = struct.pack("<IIII", int(num_to_split), int(current_site_count), 0, 0)
    queue.write_buffer(buffer, 0, data)


def update_prune_params_buffer(queue, buffer, count):
    data = struct.pack("<IIII", int(count), 0, 0, 0)
    queue.write_buffer(buffer, 0, data)


def create_densify_score_params_buffer(device, site_count, pair_count, min_mass, score_alpha):
    data = struct.pack("<4I4f", int(site_count), int(pair_count), 0, 0,
                       float(min_mass), float(score_alpha), 0.0, 0.0)
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def update_densify_score_params_buffer(queue, buffer, site_count, pair_count, min_mass, score_alpha):
    data = struct.pack("<4I4f", int(site_count), int(pair_count), 0, 0,
                       float(min_mass), float(score_alpha), 0.0, 0.0)
    queue.write_buffer(buffer, 0, data)


def create_prune_score_params_buffer(device, site_count, pair_count, delta_norm):
    data = struct.pack("<4I4f", int(site_count), int(pair_count), 0, 0,
                       float(delta_norm), 0.0, 0.0, 0.0)
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def update_prune_score_params_buffer(queue, buffer, site_count, pair_count, delta_norm):
    data = struct.pack("<4I4f", int(site_count), int(pair_count), 0, 0,
                       float(delta_norm), 0.0, 0.0, 0.0)
    queue.write_buffer(buffer, 0, data)


def read_sites(device, buffer, count):
    raw = device.queue.read_buffer(buffer, 0, count * SITE_FLOATS * 4)
    data = np.frombuffer(raw, dtype=np.float32).reshape(count, SITE_FLOATS)
    return data


def write_sites_txt(path, sites, width, height):
    active = sites[sites[:, 0] >= 0.0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SAD Sites (position_x, position_y, color_r, color_g, color_b, log_tau, radius, aniso_dir_x, aniso_dir_y, log_aniso)\n")
        f.write(f"# Image size: {width} {height}\n")
        f.write(f"# Total sites: {active.shape[0]}\n")
        f.write(f"# Active sites: {active.shape[0]}\n")
        for s in active:
            f.write("{:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}\n".format(
                s[0], s[1], s[4], s[5], s[6], s[2], s[3], s[7], s[8], s[9]
            ))


def _sites_array(count: int) -> np.ndarray:
    return np.zeros((count, SITE_FLOATS), dtype=np.float32)


def _fill_site_row(dst: np.ndarray, idx: int, pos_x: float, pos_y: float,
                   color_r: float, color_g: float, color_b: float,
                   log_tau: float, radius: float,
                   aniso_dir_x: float, aniso_dir_y: float, log_aniso: float) -> None:
    dst[idx, 0] = pos_x
    dst[idx, 1] = pos_y
    dst[idx, 2] = log_tau
    dst[idx, 3] = radius
    dst[idx, 4] = color_r
    dst[idx, 5] = color_g
    dst[idx, 6] = color_b
    dst[idx, 7] = aniso_dir_x
    dst[idx, 8] = aniso_dir_y
    dst[idx, 9] = log_aniso


def load_sites_txt(path: str):
    sites = []
    width = None
    height = None
    size_re = re.compile(r"image size\s*:?\s*(\d+)\s+(\d+)", re.IGNORECASE)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = size_re.search(line)
                if match:
                    width = int(match.group(1))
                    height = int(match.group(2))
                continue
            parts = line.split()
            if len(parts) not in (7, 10):
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) == 7:
                vals.extend([1.0, 0.0, 0.0])
            sites.append(vals)

    if not sites:
        raise RuntimeError(f"No sites found in TXT: {path}")

    data = _sites_array(len(sites))
    for i, s in enumerate(sites):
        _fill_site_row(
            data,
            i,
            s[0], s[1],
            s[2], s[3], s[4],
            s[5], s[6],
            s[7], s[8], s[9],
        )
    return data, width, height


def load_sites_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = data.get("image_width")
    height = data.get("image_height")
    sites = data.get("sites", [])
    if not sites:
        raise RuntimeError(f"No sites found in JSON: {path}")

    out = _sites_array(len(sites))
    for i, s in enumerate(sites):
        pos = s.get("pos", [0.0, 0.0])
        color = s.get("color", [0.0, 0.0, 0.0])
        log_tau = float(s.get("log_tau", 0.0))
        if "radius" in s:
            radius = float(s.get("radius", 0.0))
        else:
            radius = float(s.get("radius_sq", 0.0))
        aniso_dir = s.get("aniso_dir", [1.0, 0.0])
        log_aniso = float(s.get("log_aniso", 0.0))
        _fill_site_row(
            out,
            i,
            float(pos[0]), float(pos[1]),
            float(color[0]), float(color[1]), float(color[2]),
            log_tau, radius,
            float(aniso_dir[0]), float(aniso_dir[1]),
            log_aniso,
        )
    return out, width, height


def load_sites(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return load_sites_json(path)
    return load_sites_txt(path)


def simulate_final_sites(init_sites, max_sites, iters, densify_enabled, densify_start, densify_end,
                         densify_freq, densify_percentile, prune_during_densify, prune_start,
                         prune_end, prune_freq, prune_percentile, max_split_indices):
    actual_sites = init_sites
    active_estimate = init_sites
    max_sites = max(max_sites, init_sites)

    effective_prune_start = prune_start
    if densify_enabled and not prune_during_densify and prune_start < densify_end:
        effective_prune_start = densify_end
    for it in range(iters):
        if (densify_enabled and densify_percentile > 0.0 and it >= densify_start and
                it <= densify_end and (it % densify_freq == 0) and actual_sites < max_sites):
            desired = int(active_estimate * densify_percentile)
            available = max_sites - actual_sites
            num_to_split = min(desired, available, max_split_indices)
            if num_to_split > 0:
                actual_sites += num_to_split
                active_estimate += num_to_split

        if (prune_percentile > 0.0 and it >= effective_prune_start and it < prune_end and
                (it % prune_freq == 0)):
            desired = int(active_estimate * prune_percentile)
            num_to_prune = min(desired, max_split_indices)
            if num_to_prune > 0:
                active_estimate = max(0, active_estimate - num_to_prune)

    return active_estimate


def solve_target_bpp(target_bpp, width, height, init_sites, max_sites, iters, densify_enabled,
                     densify_start, densify_end, densify_freq, base_densify, prune_during_densify,
                     prune_start, prune_end, prune_freq, base_prune, max_split_indices):
    target_sites = max(1, int(round(target_bpp * width * height / BITS_PER_SITE)))
    max_base = max(base_densify, base_prune)
    if max_base <= 0.0:
        final_sites = simulate_final_sites(
            init_sites, max_sites, iters, densify_enabled,
            densify_start, densify_end, densify_freq, 0.0,
            prune_during_densify, prune_start, prune_end, prune_freq, 0.0,
            max_split_indices,
        )
        achieved_bpp = final_sites * BITS_PER_SITE / float(width * height)
        return 0.0, 0.0, final_sites, achieved_bpp

    max_pct = 0.95
    s_max = max_pct / max_base
    if s_max > 50.0:
        s_max = 50.0

    def eval_sites(scale):
        densify = min(max_pct, base_densify * scale) if densify_enabled else 0.0
        prune = min(max_pct, base_prune * scale)
        return simulate_final_sites(
            init_sites, max_sites, iters, densify_enabled,
            densify_start, densify_end, densify_freq, densify,
            prune_during_densify, prune_start, prune_end, prune_freq, prune,
            max_split_indices,
        )

    best_scale = 0.0
    best_sites = eval_sites(0.0)
    best_err = abs(best_sites - target_sites)
    samples = 80
    for i in range(samples + 1):
        s = s_max * float(i) / float(samples)
        sites = eval_sites(s)
        err = abs(sites - target_sites)
        if err < best_err:
            best_err = err
            best_scale = s
            best_sites = sites

    step = s_max / float(samples)
    for _ in range(20):
        improved = False
        s0 = best_scale - step
        s1 = best_scale + step
        if s0 >= 0.0:
            sites = eval_sites(s0)
            err = abs(sites - target_sites)
            if err < best_err:
                best_err = err
                best_scale = s0
                best_sites = sites
                improved = True
        if s1 <= s_max:
            sites = eval_sites(s1)
            err = abs(sites - target_sites)
            if err < best_err:
                best_err = err
                best_scale = s1
                best_sites = sites
                improved = True
        if not improved:
            step *= 0.5

    densify = min(max_pct, base_densify * best_scale) if densify_enabled else 0.0
    prune = min(max_pct, base_prune * best_scale)
    achieved_bpp = best_sites * BITS_PER_SITE / float(width * height)
    return densify, prune, best_sites, achieved_bpp


@dataclass
class HilbertResources:
    order: wgpu.GPUBuffer
    pos: wgpu.GPUBuffer
    pairs: wgpu.GPUBuffer
    sort: Optional[RadixSortUInt2]
    padded_count: int
    site_count: int
    ready: bool


def hilbert_bits_for_size(width: int, height: int) -> int:
    max_dim = max(width, height)
    n = 1
    bits = 0
    while n < max_dim:
        n <<= 1
        bits += 1
    return max(bits, 1)


def make_hilbert_resources(device, site_capacity: int) -> HilbertResources:
    site_cap = max(1, site_capacity)
    hilbert_order = device.create_buffer(
        size=site_cap * 4,
        usage=wgpu.BufferUsage.STORAGE,
    )
    hilbert_pos = device.create_buffer(
        size=site_cap * 4,
        usage=wgpu.BufferUsage.STORAGE,
    )
    padded_count = ((site_cap + 1023) // 1024) * 1024
    hilbert_pairs = device.create_buffer(
        size=padded_count * 8,
        usage=wgpu.BufferUsage.STORAGE,
    )
    hilbert_sort = RadixSortUInt2(device, padded_count) if padded_count > 0 else None
    return HilbertResources(
        order=hilbert_order,
        pos=hilbert_pos,
        pairs=hilbert_pairs,
        sort=hilbert_sort,
        padded_count=padded_count,
        site_count=0,
        ready=False,
    )


def update_hilbert_buffers(device, encoder, hilbert_encoder: HilbertEncoder,
                           resources: HilbertResources, hilbert_params_buf: wgpu.GPUBuffer,
                           sites_buffer: wgpu.GPUBuffer, site_count: int,
                           width: int, height: int) -> None:
    if site_count <= 0:
        return
    sorter = resources.sort
    if sorter is None:
        return
    bits = hilbert_bits_for_size(width, height)
    update_hilbert_params_buffer(
        device.queue,
        hilbert_params_buf,
        site_count=site_count,
        padded_count=resources.padded_count,
        width=width,
        height=height,
        bits=bits,
    )
    hilbert_encoder.encode_pairs(
        encoder,
        sites_buffer,
        resources.pairs,
        hilbert_params_buf,
        resources.padded_count,
    )
    max_key_exclusive = 0xffffffff if bits >= 16 else (1 << (bits * 2))
    sorter.encode(encoder, resources.pairs, max_key_exclusive=max_key_exclusive)
    hilbert_encoder.encode_order(
        encoder,
        resources.pairs,
        resources.order,
        resources.pos,
        hilbert_params_buf,
        site_count,
    )


@dataclass
class SiteCapacityPlan:
    densify_enabled: bool
    needs_pairs: bool
    needs_prune: bool
    max_sites_capacity: int
    requested_capacity: int
    buffer_capacity: int
    score_pairs_count: int
    max_split_indices: int


def plan_site_capacity(args, initial_site_count: int, num_pixels: int) -> SiteCapacityPlan:
    needs_pairs = bool(args.densify) or args.prune_percentile > 0.0
    needs_prune = args.prune_percentile > 0.0

    if args.max_sites > 0:
        max_sites_capacity = args.max_sites
    elif args.densify:
        max_sites_capacity = min(num_pixels * 2, max(initial_site_count * 8, 8192))
    else:
        max_sites_capacity = num_pixels * 2

    requested_capacity = max_sites_capacity if args.densify else initial_site_count
    buffer_capacity = max(initial_site_count, requested_capacity)
    score_pairs_count = max(1, buffer_capacity) if needs_pairs else 0
    return SiteCapacityPlan(
        densify_enabled=bool(args.densify),
        needs_pairs=needs_pairs,
        needs_prune=needs_prune,
        max_sites_capacity=max_sites_capacity,
        requested_capacity=requested_capacity,
        buffer_capacity=buffer_capacity,
        score_pairs_count=score_pairs_count,
        max_split_indices=65536,
    )



def save_texture_rgba32float(device, texture, width, height, out_path):
    img_f = read_texture_rgba32float(device, texture, width, height)
    img_f = np.nan_to_num(img_f, nan=0.0, posinf=1.0, neginf=0.0)
    img_u8 = np.clip(img_f * 255.0, 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(img_u8)
    img.save(out_path)


def render_from_sites(args) -> None:
    sites_np, header_w, header_h = load_sites(args.render_sites)
    site_count = int(sites_np.shape[0])
    width = int(args.width) if args.width else 0
    height = int(args.height) if args.height else 0

    if width <= 0 or height <= 0:
        if header_w and header_h:
            width, height = int(header_w), int(header_h)
        elif args.render_target:
            _, width, height = load_image(args.render_target, max_dim=MAX_DIM)
        else:
            raise RuntimeError("Render requires --width/--height, a header line, or --render-target.")

    if header_w and header_h and (width != header_w or height != header_h):
        print(
            f"Warning: sites header size {header_w}x{header_h} "
            f"differs from render size {width}x{height}."
        )

    cand_downscale = max(1, int(args.cand_downscale))
    cand_width = max(1, (width + cand_downscale - 1) // cand_downscale)
    cand_height = max(1, (height + cand_downscale - 1) // cand_downscale)
    if cand_downscale > 1:
        print(f"Candidate downscale: {cand_downscale}x -> {cand_width}x{cand_height}")

    cand_radius_scale = float(args.cand_radius_scale)
    cand_radius_probes = max(0, int(args.cand_radius_probes))
    cand_inject_count = max(0, int(args.cand_inject_count))
    cand_hilbert_window = max(0, args.cand_hilbert_window)
    cand_hilbert_probes = max(0, args.cand_hilbert_probes)
    uses_hilbert = cand_hilbert_window > 0 and cand_hilbert_probes > 0

    device = get_default_device()
    print(f"Device: {device.adapter.info}")

    sites_buffer = device.create_buffer_with_data(
        data=sites_np.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    tex_desc = {
        "size": (cand_width, cand_height, 1),
        "format": wgpu.TextureFormat.rgba32uint,
        "usage": (wgpu.TextureUsage.STORAGE_BINDING
                  | wgpu.TextureUsage.TEXTURE_BINDING
                  | wgpu.TextureUsage.COPY_SRC
                  | wgpu.TextureUsage.COPY_DST),
    }
    cand0_a = device.create_texture(**tex_desc)
    cand0_b = device.create_texture(**tex_desc)
    cand1_a = device.create_texture(**tex_desc)
    cand1_b = device.create_texture(**tex_desc)

    render_tex = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba32float,
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING,
    )

    params_buf = create_params_buffer(
        device,
        width,
        height,
        site_count,
        step=0,
        seed=0,
        radius_scale=cand_radius_scale,
        radius_probes=cand_radius_probes,
        inject_count=cand_inject_count,
        hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
        hilbert_window=cand_hilbert_window if uses_hilbert else 0,
        cand_downscale=cand_downscale,
        cand_width=cand_width,
        cand_height=cand_height,
    )
    pack_params_buf = create_clear_params_buffer(device, site_count)

    candidates_encoder = CandidatesEncoder(device)
    seed_encoder = SeedCandidatesEncoder(device)
    pack_encoder = CandidatePackEncoder(device)
    render_encoder = RenderEncoder(device)
    jfa_clear_encoder = JFAClearEncoder(device)
    jfa_flood_encoder = JFAFloodEncoder(device)
    hilbert_encoder = HilbertEncoder(device) if uses_hilbert else None

    packed_candidates_buffer = device.create_buffer(
        size=site_count * PACKED_CAND_BYTES,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    hilbert_resources = make_hilbert_resources(device, site_count) if uses_hilbert else None
    hilbert_params_buf = None
    if uses_hilbert:
        if hilbert_resources is None:
            raise RuntimeError("Hilbert resources are not initialized.")
        hilbert_params_buf = create_hilbert_params_buffer(
            device,
            site_count=1,
            padded_count=hilbert_resources.padded_count,
            width=width,
            height=height,
            bits=hilbert_bits_for_size(width, height),
        )
    dummy_hilbert_buf = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE)

    update_params_buffer(
        device.queue,
        params_buf,
        width,
        height,
        site_count,
        0,
        0,
        radius_scale=cand_radius_scale,
        radius_probes=cand_radius_probes,
        inject_count=cand_inject_count,
        hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
        hilbert_window=cand_hilbert_window if uses_hilbert else 0,
        cand_downscale=cand_downscale,
        cand_width=cand_width,
        cand_height=cand_height,
    )

    encoder = device.create_command_encoder()
    candidates_encoder.encode_init(encoder, params_buf, cand0_a, cand1_a, cand_width, cand_height)
    encoder.copy_texture_to_texture(
        {"texture": cand0_a},
        {"texture": cand0_b},
        (cand_width, cand_height, 1),
    )
    seed_encoder.encode(encoder, sites_buffer, params_buf, cand0_a, cand0_b, site_count)
    device.queue.submit([encoder.finish()])

    cand0A, cand0B = cand0_b, cand0_a
    cand1A, cand1B = cand1_a, cand1_b

    update_clear_params_buffer(device.queue, pack_params_buf, site_count)
    encoder = device.create_command_encoder()
    pack_encoder.encode(encoder, sites_buffer, packed_candidates_buffer, pack_params_buf, site_count)
    device.queue.submit([encoder.finish()])

    if uses_hilbert:
        if hilbert_encoder is None or hilbert_resources is None or hilbert_params_buf is None:
            raise RuntimeError("Hilbert resources are not initialized.")
        encoder = device.create_command_encoder()
        update_hilbert_buffers(
            device,
            encoder,
            hilbert_encoder,
            hilbert_resources,
            hilbert_params_buf,
            sites_buffer,
            site_count,
            width,
            height,
        )
        device.queue.submit([encoder.finish()])
        hilbert_resources.ready = True
        hilbert_resources.site_count = site_count

    use_jfa = not args.render_no_jfa
    rounds = max(1, args.render_jfa_rounds) if use_jfa else 1
    passes = max(0, args.render_cand_passes)
    base_passes = passes // rounds if rounds > 0 else passes
    remainder = passes % rounds if rounds > 0 else 0
    jump_pass_index = 0

    max_dim = max(cand_width, cand_height)
    step = 1
    num_passes = 0
    while step < max_dim:
        step <<= 1
        num_passes += 1

    for round_idx in range(rounds):
        if use_jfa:
            update_params_buffer(
                device.queue,
                params_buf,
                width,
                height,
                site_count,
                0,
                0,
                radius_scale=cand_radius_scale,
                radius_probes=cand_radius_probes,
                inject_count=cand_inject_count,
                hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
                hilbert_window=cand_hilbert_window if uses_hilbert else 0,
                cand_downscale=cand_downscale,
                cand_width=cand_width,
                cand_height=cand_height,
            )

            encoder = device.create_command_encoder()
            jfa_clear_encoder.encode(encoder, params_buf, cand0A, cand0B, cand_width, cand_height)
            seed_encoder.encode(encoder, sites_buffer, params_buf, cand0A, cand0B, site_count)
            device.queue.submit([encoder.finish()])

            step_size = step // 2
            read_tex = cand0B
            write_tex = cand0A
            while step_size >= 1:
                update_params_buffer(
                    device.queue,
                    params_buf,
                    width,
                    height,
                    site_count,
                    step_size,
                    0,
                    radius_scale=cand_radius_scale,
                    radius_probes=cand_radius_probes,
                    inject_count=cand_inject_count,
                    hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
                    hilbert_window=cand_hilbert_window if uses_hilbert else 0,
                    cand_downscale=cand_downscale,
                    cand_width=cand_width,
                    cand_height=cand_height,
                )
                encoder = device.create_command_encoder()
                jfa_flood_encoder.encode(encoder, sites_buffer, params_buf, read_tex, write_tex, cand_width, cand_height)
                device.queue.submit([encoder.finish()])
                read_tex, write_tex = write_tex, read_tex
                step_size //= 2

            if num_passes % 2 == 0:
                encoder = device.create_command_encoder()
                encoder.copy_texture_to_texture(
                    {"texture": cand0B},
                    {"texture": cand0A},
                    (cand_width, cand_height, 1),
                )
                device.queue.submit([encoder.finish()])

        passes_this = base_passes + (1 if round_idx < remainder else 0)
        if passes_this <= 0:
            continue

        for pass_idx in range(passes_this):
            step_index = jump_pass_index + pass_idx
            step_val = pack_jump_step(step_index, cand_width, cand_height)
            step_high = (step_index >> 16) & 0xFFFF
            update_params_buffer(
                device.queue,
                params_buf,
                width,
                height,
                site_count,
                step_val,
                step_high,
                radius_scale=cand_radius_scale,
                radius_probes=cand_radius_probes,
                inject_count=cand_inject_count,
                hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
                hilbert_window=cand_hilbert_window if uses_hilbert else 0,
                cand_downscale=cand_downscale,
                cand_width=cand_width,
                cand_height=cand_height,
            )

            hilbert_order_buf = hilbert_resources.order if uses_hilbert and hilbert_resources is not None else dummy_hilbert_buf
            hilbert_pos_buf = hilbert_resources.pos if uses_hilbert and hilbert_resources is not None else dummy_hilbert_buf
            encoder = device.create_command_encoder()
            candidates_encoder.encode_update(
                encoder,
                packed_candidates_buffer,
                params_buf,
                cand0A,
                cand1A,
                cand0B,
                cand1B,
                hilbert_order_buf,
                hilbert_pos_buf,
                cand_width,
                cand_height,
            )
            device.queue.submit([encoder.finish()])
            cand0A, cand0B = cand0B, cand0A
            cand1A, cand1B = cand1B, cand1A

        jump_pass_index += passes_this

    update_params_buffer(
        device.queue,
        params_buf,
        width,
        height,
        site_count,
        0,
        0,
        radius_scale=cand_radius_scale,
        radius_probes=cand_radius_probes,
        inject_count=cand_inject_count,
        hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
        hilbert_window=cand_hilbert_window if uses_hilbert else 0,
        cand_downscale=cand_downscale,
        cand_width=cand_width,
        cand_height=cand_height,
    )
    encoder = device.create_command_encoder()
    render_encoder.encode(
        encoder,
        sites_buffer,
        params_buf,
        cand0A,
        cand1A,
        render_tex,
        width,
        height,
    )
    device.queue.submit([encoder.finish()])

    out_path = args.render_out
    if not out_path:
        stem = Path(args.render_sites).with_suffix("")
        out_path = f"{stem}_render.png"

    save_texture_rgba32float(device, render_tex, width, height, out_path)
    print(f"Saved: {out_path}")

    if args.render_target:
        target, tw, th = load_image(args.render_target, max_dim=max(width, height))
        if tw != width or th != height:
            print("Render target size mismatch; skipping PSNR.")
        else:
            render_img = read_texture_rgba32float(device, render_tex, width, height)
            render_img = np.nan_to_num(render_img, nan=0.0, posinf=1.0, neginf=0.0)
            diff = render_img[:, :, :3] - target[:, :, :3]
            mse = float(np.mean(diff * diff)) if diff.size else 0.0
            psnr = 20.0 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100.0
            print(f"Render PSNR: {psnr:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Train SAD sites with wgpu.")
    parser.add_argument("image", nargs="?", help="Target image path")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--sites", type=int, default=DEFAULT_SITES)
    parser.add_argument("--max-sites", type=int, default=DEFAULT_MAX_SITES)
    parser.add_argument("--densify", dest="densify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--densify-start", type=int, default=DENSIFY_START)
    parser.add_argument("--densify-end", type=int, default=DENSIFY_END)
    parser.add_argument("--densify-freq", type=int, default=DENSIFY_FREQ)
    parser.add_argument("--densify-percentile", type=float, default=DENSIFY_PERCENTILE)
    parser.add_argument("--densify-score-alpha", type=float, default=DENSIFY_SCORE_ALPHA)
    parser.add_argument("--prune-during-densify", dest="prune_during_densify",
                        action=argparse.BooleanOptionalAction,
                        default=bool(CONFIG.get("PRUNE_DURING_DENSIFY", True)))
    parser.add_argument("--prune-start", type=int, default=PRUNE_START)
    parser.add_argument("--prune-end", type=int, default=PRUNE_END)
    parser.add_argument("--prune-freq", type=int, default=PRUNE_FREQ)
    parser.add_argument("--prune-percentile", type=float, default=PRUNE_PERCENTILE)
    parser.add_argument("--target-bpp", type=float, default=DEFAULT_TARGET_BPP)
    parser.add_argument("--cand-freq", type=int, default=CAND_UPDATE_FREQ,
                        help="Candidate update frequency (iterations per update)")
    parser.add_argument("--cand-passes", type=int, default=CAND_UPDATE_PASSES,
                        help="Candidate update passes per update")
    parser.add_argument("--cand-radius-scale", type=float, default=CAND_RADIUS_SCALE)
    parser.add_argument("--cand-radius-probes", type=int, default=CAND_RADIUS_PROBES)
    parser.add_argument("--cand-inject", "--cand-inject-count", dest="cand_inject_count",
                        type=int, default=CAND_INJECT_COUNT)
    parser.add_argument("--cand-hilbert-window", type=int, default=CAND_HILBERT_WINDOW)
    parser.add_argument("--cand-hilbert-probes", type=int, default=CAND_HILBERT_PROBES)
    parser.add_argument("--cand-downscale", type=int, default=CAND_DOWNSCALE)
    parser.add_argument("--max-dim", type=int, default=MAX_DIM)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--lr-pos", type=float, default=LR_POS_BASE)
    parser.add_argument("--lr-tau", type=float, default=LR_TAU_BASE)
    parser.add_argument("--lr-radius", type=float, default=LR_RADIUS_BASE)
    parser.add_argument("--lr-color", type=float, default=LR_COLOR_BASE)
    parser.add_argument("--lr-dir", type=float, default=LR_DIR_BASE)
    parser.add_argument("--lr-aniso", type=float, default=LR_ANISO_BASE)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--mask", default=None, help="Optional mask image (white=keep, black=ignore)")
    parser.add_argument("--init-from-sites", default=None,
                        help="Initialize from sites TXT file, overrides --sites")
    parser.add_argument("--render", dest="render_sites", default=None,
                        help="Render from sites TXT file")
    parser.add_argument("--out", dest="render_out", default=None,
                        help="Output PNG path for --render")
    parser.add_argument("--width", type=int, default=0, help="Render width (optional)")
    parser.add_argument("--height", type=int, default=0, help="Render height (optional)")
    parser.add_argument("--render-target", default=None,
                        help="Target image for PSNR in render mode")
    parser.add_argument("--render-cand-passes", type=int, default=16,
                        help="VPT passes for render mode")
    parser.add_argument("--render-no-jfa", action="store_true",
                        help="Disable JFA flood before render VPT passes")
    parser.add_argument("--render-jfa-rounds", type=int, default=1,
                        help="JFA reset rounds for render mode")
    args = parser.parse_args()

    if args.render_sites:
        render_from_sites(args)
        return

    if args.image is None:
        parser.error("image is required unless --render is specified")

    cand_hilbert_window = max(0, args.cand_hilbert_window)
    cand_hilbert_probes = max(0, args.cand_hilbert_probes)
    uses_hilbert = cand_hilbert_window > 0 and cand_hilbert_probes > 0
    cand_update_passes = max(1, int(args.cand_passes))
    cand_radius_scale = float(args.cand_radius_scale)
    cand_radius_probes = max(0, int(args.cand_radius_probes))
    cand_inject_count = max(0, int(args.cand_inject_count))
    lr_pos_base = float(args.lr_pos) * float(args.lr)
    lr_tau_base = float(args.lr_tau) * float(args.lr)
    lr_radius_base = float(args.lr_radius) * float(args.lr)
    lr_color_base = float(args.lr_color) * float(args.lr)
    lr_dir_base = float(args.lr_dir) * float(args.lr)
    lr_aniso_base = float(args.lr_aniso) * float(args.lr)

    target, width, height = load_image(args.image, max_dim=max(0, int(args.max_dim)))
    cand_downscale = max(1, int(args.cand_downscale))
    cand_width = max(1, (width + cand_downscale - 1) // cand_downscale)
    cand_height = max(1, (height + cand_downscale - 1) // cand_downscale)
    if args.mask:
        mask, mask_sum = load_mask(args.mask, width, height)
    else:
        mask = np.ones((height, width), dtype=np.float32)
        mask_sum = float(width * height)

    init_sites_np = None
    init_site_count = int(args.sites)
    if args.init_from_sites:
        init_sites_np, init_w, init_h = load_sites(args.init_from_sites)
        if init_w and init_h and (init_w != width or init_h != height):
            scale_x = float(width) / float(init_w)
            scale_y = float(height) / float(init_h)
            init_sites_np[:, 0] *= scale_x
            init_sites_np[:, 1] *= scale_y
            print(
                f"Scaling init sites from {init_w}x{init_h} to {width}x{height} "
                f"(scale {scale_x:.4f}, {scale_y:.4f})"
        )
        init_sites_np[:, 2] = INIT_LOG_TAU
        init_site_count = int(init_sites_np.shape[0])

    device = get_default_device()

    target_tex = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    write_texture_rgba32float(device, target_tex, target, width, height)
    mask_tex = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    mask_rgba = np.repeat(mask[:, :, None], 4, axis=2)
    mask_rgba[:, :, 3] = 1.0
    write_texture_rgba32float(device, mask_tex, mask_rgba, width, height)

    if args.target_bpp > 0.0:
        if args.max_sites > 0:
            max_sites_capacity = args.max_sites
        elif args.densify:
            max_sites_capacity = min(width * height * 2, max(init_site_count * 8, 8192))
        else:
            max_sites_capacity = width * height * 2
        max_sites = max(max_sites_capacity, init_site_count)
        densify_percentile, prune_percentile, final_sites, achieved_bpp = solve_target_bpp(
            args.target_bpp,
            width,
            height,
            init_site_count,
            max_sites,
            args.iters,
            args.densify,
            args.densify_start,
            args.densify_end,
            max(1, args.densify_freq),
            args.densify_percentile,
            args.prune_during_densify,
            args.prune_start,
            args.prune_end,
            max(1, args.prune_freq),
            args.prune_percentile,
            65536,
        )
        args.densify_percentile = densify_percentile
        args.prune_percentile = prune_percentile

    plan = plan_site_capacity(args, init_site_count, width * height)
    if plan.buffer_capacity != plan.requested_capacity:
        print(
            "Warning: expanding site buffer capacity from "
            f"{plan.requested_capacity} to {plan.buffer_capacity} to fit initial sites."
        )

    buffer_capacity = plan.buffer_capacity
    sites_buffer = device.create_buffer(
        size=buffer_capacity * SITE_FLOATS * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )
    packed_candidates_buffer = device.create_buffer(
        size=buffer_capacity * PACKED_CAND_BYTES,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    adam_buffer = device.create_buffer(
        size=buffer_capacity * ADAM_FLOATS * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )
    device.queue.write_buffer(adam_buffer, 0, np.zeros(buffer_capacity * ADAM_FLOATS, dtype=np.float32).tobytes())

    seed_counter = device.create_buffer_with_data(
        data=struct.pack("<I", 0),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    if init_site_count <= 0:
        raise RuntimeError("Initial site count must be positive.")

    if init_sites_np is None:
        grad_threshold = 0.01
        init_params = device.create_buffer_with_data(
            data=struct.pack("<IfIIffII", init_site_count, grad_threshold, 256, 0, INIT_LOG_TAU, INIT_RADIUS, 0, 0),
            usage=wgpu.BufferUsage.UNIFORM,
        )

        init_encoder = InitGradientEncoder(device)
        encoder = device.create_command_encoder()
        init_encoder.encode(encoder, sites_buffer, seed_counter, init_params, target_tex, mask_tex, init_site_count)
        device.queue.submit([encoder.finish()])

        sites_np = read_sites(device, sites_buffer, init_site_count)
        active_estimate = int((sites_np[:, 0] >= 0.0).sum())
    else:
        device.queue.write_buffer(sites_buffer, 0, init_sites_np.tobytes())
        active_estimate = int((init_sites_np[:, 0] >= 0.0).sum())
    _print_training_overview(
        args,
        device,
        width,
        height,
        init_site_count,
        active_estimate,
        plan,
        cand_downscale,
    )
    print("Logs | Iter | PSNR | Active | speed | elapsed")

    tex_desc = {
        "size": (cand_width, cand_height, 1),
        "format": wgpu.TextureFormat.rgba32uint,
        "usage": (wgpu.TextureUsage.STORAGE_BINDING
                  | wgpu.TextureUsage.TEXTURE_BINDING
                  | wgpu.TextureUsage.COPY_SRC
                  | wgpu.TextureUsage.COPY_DST),
    }
    cand0_a = device.create_texture(**tex_desc)
    cand0_b = device.create_texture(**tex_desc)
    cand1_a = device.create_texture(**tex_desc)
    cand1_b = device.create_texture(**tex_desc)

    render_tex = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba32float,
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING,
    )

    grad_buffers = [
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
        device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE),
    ]
    removal_delta = device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE)
    tau_grad_raw = device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE)
    tau_grad_tmp = device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE)
    tau_grad_tmp2 = device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE)

    stat_buffers = None
    if args.densify:
        stat_buffers = [
            device.create_buffer(size=buffer_capacity * 4, usage=wgpu.BufferUsage.STORAGE) for _ in range(8)
        ]

    pairs_buffer = None
    if plan.needs_pairs:
        pairs_buffer = device.create_buffer(size=plan.score_pairs_count * 8, usage=wgpu.BufferUsage.STORAGE)

    split_indices_buffer = None
    if args.densify:
        split_indices_buffer = device.create_buffer(
            size=plan.max_split_indices * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )

    prune_indices_buffer = None
    if plan.needs_prune:
        prune_indices_buffer = device.create_buffer(
            size=plan.max_split_indices * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )

    inv_scale_sq = 1.0 / (max(width, height) ** 2)
    max_cand_passes = cand_update_passes
    cand_params_buffers = [
        create_params_buffer(
            device,
            width,
            height,
            init_site_count,
            step=0,
            seed=0,
            radius_scale=cand_radius_scale,
            radius_probes=cand_radius_probes,
            inject_count=cand_inject_count,
            hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
            hilbert_window=cand_hilbert_window if uses_hilbert else 0,
            cand_downscale=cand_downscale,
            cand_width=cand_width,
            cand_height=cand_height,
        )
        for _ in range(max_cand_passes)
    ]
    render_params_buf = create_params_buffer(
        device,
        width,
        height,
        init_site_count,
        step=0,
        seed=0,
        radius_scale=cand_radius_scale,
        radius_probes=cand_radius_probes,
        inject_count=cand_inject_count,
        hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
        hilbert_window=cand_hilbert_window if uses_hilbert else 0,
        cand_downscale=cand_downscale,
        cand_width=cand_width,
        cand_height=cand_height,
    )
    grad_params_stats_buf = create_grad_params_buffer(device, init_site_count, inv_scale_sq, 0)
    grad_params_grad_buf = create_grad_params_buffer(device, init_site_count, inv_scale_sq, 0)
    adam_params_buf = create_adam_params_buffer(device, lr_pos_base, lr_tau_base, lr_radius_base,
                                                lr_color_base, lr_dir_base, lr_aniso_base,
                                                BETA1, BETA2, EPS, 1, width, height)
    clear_params_pre_buf = create_clear_params_buffer(device, init_site_count)
    clear_params_post_buf = create_clear_params_buffer(device, init_site_count)
    pack_params_buf = create_clear_params_buffer(device, init_site_count)

    split_params_buf = None
    if split_indices_buffer is not None:
        split_params_buf = create_split_params_buffer(device, 0, init_site_count)

    densify_indices_params_buf = None
    if split_indices_buffer is not None:
        densify_indices_params_buf = create_prune_params_buffer(device, 0)

    prune_indices_params_buf = None
    if plan.needs_prune:
        prune_indices_params_buf = create_prune_params_buffer(device, 0)

    densify_score_params_buf = None
    if stat_buffers is not None:
        densify_score_params_buf = create_densify_score_params_buffer(
            device, init_site_count, plan.score_pairs_count, 1.0, args.densify_score_alpha
        )

    prune_score_params_buf = None
    if plan.needs_prune:
        prune_score_params_buf = create_prune_score_params_buffer(
            device, init_site_count, plan.score_pairs_count, 1.0 / float(width * height)
        )

    tau_params_buf = create_tau_params_buffer(device, init_site_count, TAU_DIFFUSE_LAMBDA, cand_downscale)

    candidates_encoder = CandidatesEncoder(device)
    seed_encoder = SeedCandidatesEncoder(device)
    pack_encoder = CandidatePackEncoder(device)
    render_encoder = RenderEncoder(device)
    gradients_encoder = GradientsEncoder(device)
    adam_encoder = AdamEncoder(device)
    tau_encoder = TauEncoder(device)
    stats_encoder = StatsEncoder(device)
    split_encoder = SplitEncoder(device)
    prune_encoder = PruneEncoder(device)
    clear_encoder = ClearBufferEncoder(device)
    clear_i32_encoder = ClearBufferEncoder(device, use_i32=True)
    score_pairs_encoder = ScorePairsEncoder(device)
    write_indices_encoder = WriteIndicesEncoder(device)
    hilbert_encoder = HilbertEncoder(device) if uses_hilbert else None
    hilbert_resources = make_hilbert_resources(device, buffer_capacity) if uses_hilbert else None
    hilbert_params_buf = None
    if uses_hilbert and hilbert_resources is not None:
        hilbert_params_buf = create_hilbert_params_buffer(
            device,
            site_count=1,
            padded_count=hilbert_resources.padded_count,
            width=width,
            height=height,
            bits=hilbert_bits_for_size(width, height),
        )
    dummy_hilbert_buf = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE)

    update_params_buffer(
        device.queue,
        cand_params_buffers[0],
        width,
        height,
        init_site_count,
        0,
        0,
        radius_scale=cand_radius_scale,
        radius_probes=cand_radius_probes,
        inject_count=cand_inject_count,
        hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
        hilbert_window=cand_hilbert_window if uses_hilbert else 0,
        cand_downscale=cand_downscale,
        cand_width=cand_width,
        cand_height=cand_height,
    )
    encoder = device.create_command_encoder()
    candidates_encoder.encode_init(encoder, cand_params_buffers[0], cand0_a, cand1_a, cand_width, cand_height)
    encoder.copy_texture_to_texture(
        {"texture": cand0_a},
        {"texture": cand0_b},
        (cand_width, cand_height, 1),
    )
    seed_encoder.encode(encoder, sites_buffer, cand_params_buffers[0], cand0_a, cand0_b, init_site_count)
    device.queue.submit([encoder.finish()])

    cand0A, cand0B = cand0_b, cand0_a
    cand1A, cand1B = cand1_a, cand1_b

    actual_sites = init_site_count
    cleared_site_count = 0
    densify_enabled = args.densify
    densify_start = args.densify_start
    densify_end = args.densify_end
    densify_freq = max(1, args.densify_freq)
    densify_percentile = args.densify_percentile
    densify_score_alpha = args.densify_score_alpha
    prune_percentile = args.prune_percentile
    prune_start = args.prune_start
    prune_end = args.prune_end
    prune_freq = max(1, args.prune_freq)
    prune_during_densify = args.prune_during_densify
    aniso_start_iter = max(10, prune_start // 2)

    effective_prune_start = prune_start
    if densify_enabled and not prune_during_densify and prune_start < densify_end:
        effective_prune_start = densify_end

    train_start = time.time()
    best_psnr = 0.0
    jump_pass_index = 0
    radix_sort = RadixSortUInt2(device, plan.score_pairs_count) if plan.needs_pairs else None
    cand_update_freq = max(1, args.cand_freq)
    cand_update_passes = max(1, int(args.cand_passes))

    log_freq = max(args.log_freq, 1)
    for it in range(args.iters):
        pre_sites = actual_sites
        pre_active = active_estimate

        update_candidates = (it % cand_update_freq == 0)
        cand_passes = cand_update_passes

        should_densify = (
            densify_enabled
            and it >= densify_start
            and it <= densify_end
            and it % densify_freq == 0
            and pre_sites < buffer_capacity
        )
        should_prune = (
            prune_percentile > 0.0
            and it >= effective_prune_start
            and it < prune_end
            and it % prune_freq == 0
            and plan.needs_prune
        )

        num_to_split = 0
        if should_densify:
            desired = int(pre_active * densify_percentile)
            available = buffer_capacity - pre_sites
            num_to_split = min(desired, available, plan.max_split_indices)
        run_densify = should_densify and num_to_split > 0

        post_sites = pre_sites + num_to_split
        post_active = pre_active + num_to_split

        num_to_prune = 0
        if should_prune:
            desired = int(post_active * prune_percentile)
            num_to_prune = min(desired, plan.max_split_indices)
        post_active = max(0, post_active - num_to_prune)

        if update_candidates:
            for pass_idx in range(cand_passes):
                step_index = jump_pass_index + pass_idx
                step = pack_jump_step(step_index, cand_width, cand_height)
                step_high = (step_index >> 16) & 0xFFFF
                update_params_buffer(
                    device.queue,
                    cand_params_buffers[pass_idx],
                    width,
                    height,
                    pre_sites,
                    step,
                    step_high,
                    radius_scale=cand_radius_scale,
                    radius_probes=cand_radius_probes,
                    inject_count=cand_inject_count,
                    hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
                    hilbert_window=cand_hilbert_window if uses_hilbert else 0,
                    cand_downscale=cand_downscale,
                    cand_width=cand_width,
                    cand_height=cand_height,
                )

        update_clear_params_buffer(device.queue, clear_params_pre_buf, pre_sites)
        if update_candidates:
            update_clear_params_buffer(device.queue, pack_params_buf, pre_sites)
        if run_densify and stat_buffers is not None and pairs_buffer is not None and split_indices_buffer is not None:
            update_grad_params_buffer(device.queue, grad_params_stats_buf, pre_sites, inv_scale_sq, 0)
            update_densify_score_params_buffer(
                device.queue,
                densify_score_params_buf,
                pre_sites,
                plan.score_pairs_count,
                1.0,
                densify_score_alpha,
            )
            if densify_indices_params_buf is not None and split_params_buf is not None:
                update_prune_params_buffer(device.queue, densify_indices_params_buf, num_to_split)
                update_split_params_buffer(device.queue, split_params_buf, num_to_split, pre_sites)

        update_clear_params_buffer(device.queue, clear_params_post_buf, post_sites)
        update_grad_params_buffer(device.queue, grad_params_grad_buf, post_sites, inv_scale_sq, 1 if should_prune else 0)

        lr_dir = lr_dir_base if it >= aniso_start_iter else 0.0
        lr_aniso = lr_aniso_base if it >= aniso_start_iter else 0.0
        update_adam_params_buffer(
            device.queue,
            adam_params_buf,
            lr_pos_base,
            lr_tau_base,
            lr_radius_base,
            lr_color_base,
            lr_dir,
            lr_aniso,
            BETA1,
            BETA2,
            EPS,
            it + 1,
            width,
            height,
        )

        blend = float(it) / float(max(1, args.iters))
        lambda_value = TAU_DIFFUSE_LAMBDA * (0.1 + 0.9 * blend)
        update_tau_params_buffer(device.queue, tau_params_buf, post_sites, lambda_value, cand_downscale)

        if should_prune and prune_score_params_buf is not None:
            update_prune_score_params_buffer(
                device.queue,
                prune_score_params_buf,
                post_sites,
                plan.score_pairs_count,
                1.0 / float(width * height),
            )
            if num_to_prune > 0 and prune_indices_params_buf is not None:
                update_prune_params_buffer(device.queue, prune_indices_params_buf, num_to_prune)

        should_log = it % log_freq == 0 or it == args.iters - 1
        if should_log:
            update_params_buffer(
                device.queue,
                render_params_buf,
                width,
                height,
                post_sites,
                0,
                0,
                radius_scale=cand_radius_scale,
                radius_probes=cand_radius_probes,
                inject_count=cand_inject_count,
                hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
                hilbert_window=cand_hilbert_window if uses_hilbert else 0,
                cand_downscale=cand_downscale,
                cand_width=cand_width,
                cand_height=cand_height,
            )

        encoder = device.create_command_encoder()
        if update_candidates:
            if uses_hilbert:
                if hilbert_encoder is None or hilbert_resources is None or hilbert_params_buf is None:
                    raise RuntimeError("Hilbert resources are not initialized.")
                if (not hilbert_resources.ready) or (hilbert_resources.site_count != pre_sites):
                    update_hilbert_buffers(
                        device,
                        encoder,
                        hilbert_encoder,
                        hilbert_resources,
                        hilbert_params_buf,
                        sites_buffer,
                        pre_sites,
                        width,
                        height,
                    )
                    hilbert_resources.ready = True
                    hilbert_resources.site_count = pre_sites
            hilbert_order_buf = hilbert_resources.order if uses_hilbert and hilbert_resources is not None else dummy_hilbert_buf
            hilbert_pos_buf = hilbert_resources.pos if uses_hilbert and hilbert_resources is not None else dummy_hilbert_buf
            pack_encoder.encode(encoder, sites_buffer, packed_candidates_buffer, pack_params_buf, pre_sites)
            for pass_idx in range(cand_passes):
                candidates_encoder.encode_update(
                    encoder,
                    packed_candidates_buffer,
                    cand_params_buffers[pass_idx],
                    cand0A,
                    cand1A,
                    cand0B,
                    cand1B,
                    hilbert_order_buf,
                    hilbert_pos_buf,
                    cand_width,
                    cand_height,
                )
                cand0A, cand0B = cand0B, cand0A
                cand1A, cand1B = cand1B, cand1A
                jump_pass_index += 1

        if run_densify and stat_buffers is not None and pairs_buffer is not None and split_indices_buffer is not None:
            for buf in stat_buffers:
                clear_encoder.encode(encoder, buf, clear_params_pre_buf, pre_sites)

            stats_encoder.encode(
                encoder,
                sites_buffer,
                grad_params_stats_buf,
                cand0A,
                cand1A,
                target_tex,
                mask_tex,
                stat_buffers,
                width,
                height,
            )
            score_pairs_encoder.encode_densify(
                encoder,
                sites_buffer,
                stat_buffers[0],
                stat_buffers[1],
                pairs_buffer,
                densify_score_params_buf,
                plan.score_pairs_count,
            )
            if radix_sort is not None:
                radix_sort.encode(encoder, pairs_buffer, max_key_exclusive=0xffffffff)

            if densify_indices_params_buf is not None and split_params_buf is not None:
                write_indices_encoder.encode(
                    encoder, pairs_buffer, split_indices_buffer, densify_indices_params_buf, num_to_split
                )
                split_encoder.encode(
                    encoder,
                    sites_buffer,
                    adam_buffer,
                    split_indices_buffer,
                    stat_buffers,
                    split_params_buf,
                    target_tex,
                    num_to_split,
                )

        if should_prune:
            clear_encoder.encode(encoder, removal_delta, clear_params_post_buf, post_sites)
        if post_sites > cleared_site_count:
            for buf in grad_buffers:
                clear_i32_encoder.encode(encoder, buf, clear_params_post_buf, post_sites)
            cleared_site_count = post_sites
        gradients_encoder.encode(
            encoder,
            sites_buffer,
            grad_params_grad_buf,
            cand0A,
            cand1A,
            target_tex,
            mask_tex,
            grad_buffers,
            removal_delta,
            width,
            height,
        )

        if TAU_DIFFUSE_PASSES > 0 and TAU_DIFFUSE_LAMBDA > 0.0:
            tau_encoder.encode_extract(encoder, grad_buffers[2], tau_grad_raw, clear_params_post_buf, post_sites)
            current_in = tau_grad_raw
            current_out = tau_grad_tmp
            for _ in range(TAU_DIFFUSE_PASSES):
                tau_encoder.encode_diffuse(
                    encoder,
                    cand0A,
                    cand1A,
                    sites_buffer,
                    tau_grad_raw,
                    current_in,
                    current_out,
                    tau_params_buf,
                    post_sites,
                )
                if current_out is tau_grad_tmp:
                    current_in, current_out = current_out, tau_grad_tmp2
                else:
                    current_in, current_out = current_out, tau_grad_tmp

            tau_encoder.encode_writeback(encoder, current_in, grad_buffers[2], clear_params_post_buf, post_sites)

        adam_encoder.encode(encoder, sites_buffer, adam_buffer, grad_buffers, adam_params_buf, post_sites)

        if should_prune and pairs_buffer is not None and prune_indices_buffer is not None:
            score_pairs_encoder.encode_prune(
                encoder,
                sites_buffer,
                removal_delta,
                pairs_buffer,
                prune_score_params_buf,
                plan.score_pairs_count,
            )
            if radix_sort is not None:
                radix_sort.encode(encoder, pairs_buffer, max_key_exclusive=0xffffffff)

            if num_to_prune > 0 and prune_indices_params_buf is not None:
                write_indices_encoder.encode(
                    encoder, pairs_buffer, prune_indices_buffer, prune_indices_params_buf, num_to_prune
                )
                prune_encoder.encode(encoder, sites_buffer, prune_indices_buffer, prune_indices_params_buf, num_to_prune)

        if should_log:
            render_encoder.encode(
                encoder,
                sites_buffer,
                render_params_buf,
                cand0A,
                cand1A,
                render_tex,
                width,
                height,
            )
        device.queue.submit([encoder.finish()])

        if should_log:
            render_img = read_texture_rgba32float(device, render_tex, width, height)
            render_img = np.nan_to_num(render_img, nan=0.0, posinf=1.0, neginf=0.0)
            diff = render_img[:, :, :3] - target[:, :, :3]
            if mask_sum > 0.0:
                diff = diff[mask > 0.0]
                mse = float(np.mean(diff * diff)) if diff.size else 0.0
            else:
                mse = 0.0
            psnr = 20.0 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100.0
            best_psnr = max(best_psnr, psnr)
            elapsed = time.time() - train_start
            its_per_sec = (it + 1) / max(elapsed, 1e-6)
            print(f"Iter {it:4d} | PSNR: {psnr:.2f} dB | Active: {post_active}/{post_sites} | {its_per_sec:.1f} it/s | {elapsed:.1f}s")

        actual_sites = post_sites
        active_estimate = post_active

    train_time = time.time() - train_start
    results_dir = args.out_dir
    os.makedirs(results_dir, exist_ok=True)
    base = Path(args.image).stem
    out_image = os.path.join(results_dir, f"{base}.png")
    out_sites = os.path.join(results_dir, f"{base}_sites.txt")

    update_params_buffer(
        device.queue,
        render_params_buf,
        width,
        height,
        actual_sites,
        0,
        0,
        radius_scale=cand_radius_scale,
        radius_probes=cand_radius_probes,
        inject_count=cand_inject_count,
        hilbert_probes=cand_hilbert_probes if uses_hilbert else 0,
        hilbert_window=cand_hilbert_window if uses_hilbert else 0,
        cand_downscale=cand_downscale,
        cand_width=cand_width,
        cand_height=cand_height,
    )
    encoder = device.create_command_encoder()
    render_encoder.encode(
        encoder,
        sites_buffer,
        render_params_buf,
        cand0A,
        cand1A,
        render_tex,
        width,
        height,
    )
    device.queue.submit([encoder.finish()])

    save_texture_rgba32float(device, render_tex, width, height, out_image)
    sites_np = read_sites(device, sites_buffer, actual_sites)
    write_sites_txt(out_sites, sites_np, width, height)

    print(f"Saved: {out_image}")
    print(f"Saved: {out_sites}")
    total_time = time.time() - train_start
    print(f"Training time: {train_time:.2f} s")
    print(f"Total time: {total_time:.2f} s")


if __name__ == "__main__":
    main()
