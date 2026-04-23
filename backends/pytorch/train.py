#!/usr/bin/env python3
"""
PyTorch SAD backend: GPU-accelerated training via Metal (MPS) or CUDA extension.
This keeps only the high-performance GPU path (no pure-PyTorch training/decoder).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", Path(__file__).resolve().parents[2] / "training_config.json"))
BITS_PER_SITE = 16.0 * 8.0
MAX_SPLIT_INDICES = 65536


@dataclass
class SiteState:
    pos: torch.Tensor
    color: torch.Tensor
    log_tau: torch.Tensor
    radius: torch.Tensor
    aniso_dir: torch.Tensor
    log_aniso: torch.Tensor


def load_training_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise SystemExit("No CUDA/MPS device available. Use --backend metal/webgpu/cuda instead.")


def _init_summary(args: argparse.Namespace) -> str:
    if args.init_from_sites:
        return "sites"
    if args.init_per_pixel:
        return "per-pixel"
    return "gradient"


def _print_training_overview(args: argparse.Namespace,
                             device: torch.device,
                             width: int,
                             height: int,
                             actual_sites: int,
                             active_sites: int,
                             buffer_capacity: int,
                             cand_scale: int) -> None:
    bpp = f" | target-bpp={args.target_bpp:.3f}" if args.target_bpp and args.target_bpp > 0.0 else ""
    print(
        f"Training | backend=pytorch | device={device.type} | image={width}x{height} | "
        f"sites={active_sites}/{actual_sites} | iters={args.iters} | "
        f"log-freq={args.log_freq} | mask={'yes' if args.mask else 'no'} | "
        f"out={args.out_dir}{bpp}"
    )
    prune_end = args.prune_end if args.prune_end > 0 else args.iters - 1
    prune = (
        f"on {args.prune_percentile:.3f} @{args.prune_start}-{prune_end}/{max(1, args.prune_freq)}"
        if args.prune_percentile > 0.0
        else "off"
    )
    densify = f"on cap={buffer_capacity}" if args.densify else "off"
    hilbert = (
        f" hilbert={args.cand_hilbert_probes}x{args.cand_hilbert_window}"
        if args.cand_hilbert_probes > 0 and args.cand_hilbert_window > 0
        else ""
    )
    print(
        f"Schedule | init={_init_summary(args)} | densify={densify} | prune={prune} | "
        f"cand=freq {args.cand_freq}, passes {args.cand_passes}, downscale {cand_scale}x{hilbert}"
    )


def load_image(path: Path, max_dim: int = 0) -> Tuple[torch.Tensor, int, int]:
    img = Image.open(path).convert("RGB")
    if max_dim and max(img.size) > max_dim:
        scale = max_dim / float(max(img.size))
        new_w = max(1, int(round(img.size[0] * scale)))
        new_h = max(1, int(round(img.size[1] * scale)))
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    return tensor, tensor.shape[1], tensor.shape[0]


def load_mask(path: Optional[Path], width: int, height: int) -> Optional[torch.Tensor]:
    if path is None:
        return None
    img = Image.open(path).convert("L")
    if img.size != (width, height):
        img = img.resize((width, height), resample=Image.NEAREST)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def load_sites_txt(path: Path,
                   width_override: Optional[int] = None,
                   height_override: Optional[int] = None) -> Tuple[SiteState, int, int]:
    data = []
    header_w, header_h = None, None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "image size" in line.lower():
                    parts = line.split(":")[1].strip().split()
                    header_w, header_h = int(parts[0]), int(parts[1])
                continue
            parts = line.split()
            if len(parts) not in (7, 10):
                continue
            vals = [float(x) for x in parts]
            if len(vals) == 7:
                vals.extend([1.0, 0.0, 0.0])
            data.append(vals)

    if header_w is None or header_h is None:
        if width_override is None or height_override is None:
            raise ValueError("Image size not found in sites header")
        header_w, header_h = int(width_override), int(height_override)
    if not data:
        raise ValueError("No sites found in file")

    arr = torch.tensor(data, dtype=torch.float32)
    pos = arr[:, 0:2]
    color = arr[:, 2:5]
    log_tau = arr[:, 5]
    radius = arr[:, 6]
    aniso_dir = arr[:, 7:9]
    log_aniso = arr[:, 9]
    return (
        SiteState(pos=pos, color=color, log_tau=log_tau, radius=radius,
                  aniso_dir=aniso_dir, log_aniso=log_aniso),
        header_w,
        header_h,
    )


def load_sites_json(path: Path) -> Tuple[SiteState, int, int]:
    with open(path, "r") as f:
        j = json.load(f)
    width = int(j.get("image_width"))
    height = int(j.get("image_height"))
    sites = j.get("sites", [])
    if not sites:
        raise ValueError("No sites found in JSON")

    pos = []
    color = []
    log_tau = []
    radius = []
    aniso_dir = []
    log_aniso = []

    for s in sites:
        p = s.get("pos", [0.0, 0.0])
        c = s.get("color", [0.0, 0.0, 0.0])
        pos.append([p[0], p[1]])
        color.append([c[0], c[1], c[2]])
        log_tau.append(float(s.get("log_tau", 0.0)))
        radius.append(float(s.get("radius", 0.0)))
        aniso_dir.append(s.get("aniso_dir", [1.0, 0.0]))
        log_aniso.append(float(s.get("log_aniso", 0.0)))

    pos = torch.tensor(pos, dtype=torch.float32)
    color = torch.tensor(color, dtype=torch.float32)
    log_tau = torch.tensor(log_tau, dtype=torch.float32)
    radius = torch.tensor(radius, dtype=torch.float32)
    aniso_dir = torch.tensor(aniso_dir, dtype=torch.float32)
    log_aniso = torch.tensor(log_aniso, dtype=torch.float32)

    return SiteState(pos=pos, color=color, log_tau=log_tau, radius=radius,
                     aniso_dir=aniso_dir, log_aniso=log_aniso), width, height


def load_sites(path: Path,
               width_override: Optional[int] = None,
               height_override: Optional[int] = None) -> Tuple[SiteState, int, int]:
    if path.suffix.lower() == ".json":
        return load_sites_json(path)
    return load_sites_txt(path, width_override=width_override, height_override=height_override)


def scale_sites_to_target(sites: SiteState, from_w: int, from_h: int,
                           to_w: int, to_h: int) -> SiteState:
    scale_x = float(to_w) / float(from_w)
    scale_y = float(to_h) / float(from_h)
    pos = sites.pos.clone()
    pos[:, 0] *= scale_x
    pos[:, 1] *= scale_y
    return SiteState(
        pos=pos,
        color=sites.color.clone(),
        log_tau=sites.log_tau.clone(),
        radius=sites.radius.clone(),
        aniso_dir=sites.aniso_dir.clone(),
        log_aniso=sites.log_aniso.clone(),
    )


def init_sites_per_pixel(target: torch.Tensor, init_log_tau: float, init_radius: float) -> SiteState:
    h, w, _ = target.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    pos = torch.stack([xx.reshape(-1).float(), yy.reshape(-1).float()], dim=1)
    color = target.reshape(-1, 3).clone()
    log_tau = torch.full((pos.shape[0],), float(init_log_tau), dtype=torch.float32)
    radius = torch.full((pos.shape[0],), float(init_radius), dtype=torch.float32)
    aniso_dir = torch.zeros((pos.shape[0], 2), dtype=torch.float32)
    aniso_dir[:, 0] = 1.0
    log_aniso = torch.zeros((pos.shape[0],), dtype=torch.float32)
    return SiteState(pos=pos, color=color, log_tau=log_tau, radius=radius,
                     aniso_dir=aniso_dir, log_aniso=log_aniso)


def init_sites_placeholder(n_sites: int, init_log_tau: float, init_radius: float) -> SiteState:
    pos = torch.full((n_sites, 2), -1.0, dtype=torch.float32)
    color = torch.zeros((n_sites, 3), dtype=torch.float32)
    log_tau = torch.full((n_sites,), float(init_log_tau), dtype=torch.float32)
    radius = torch.full((n_sites,), float(init_radius), dtype=torch.float32)
    aniso_dir = torch.zeros((n_sites, 2), dtype=torch.float32)
    aniso_dir[:, 0] = 1.0
    log_aniso = torch.zeros((n_sites,), dtype=torch.float32)
    return SiteState(pos=pos, color=color, log_tau=log_tau, radius=radius,
                     aniso_dir=aniso_dir, log_aniso=log_aniso)


def pack_sites_padded(sites: SiteState, capacity: int, device: torch.device, stride: int = 12) -> torch.Tensor:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if stride not in (10, 12):
        raise ValueError("stride must be 10 or 12")
    out = torch.zeros((capacity, stride), dtype=torch.float32, device=device)
    out[:, 0:2] = -1.0
    n = min(sites.pos.shape[0], capacity)
    if n == 0:
        return out
    if stride == 12:
        out[:n, 0:2] = sites.pos[:n].float()
        out[:n, 2] = sites.log_tau[:n].float()
        out[:n, 3] = sites.radius[:n].float()
        out[:n, 4:7] = sites.color[:n].float()
        out[:n, 8:10] = sites.aniso_dir[:n].float()
        out[:n, 10] = sites.log_aniso[:n].float()
    else:
        out[:n, 0:2] = sites.pos[:n].float()
        out[:n, 2] = sites.log_tau[:n].float()
        out[:n, 3] = sites.radius[:n].float()
        out[:n, 4:7] = sites.color[:n].float()
        out[:n, 7:9] = sites.aniso_dir[:n].float()
        out[:n, 9] = sites.log_aniso[:n].float()
    return out


def unpack_sites_padded(packed: torch.Tensor) -> SiteState:
    packed = packed.float()
    if packed.size(1) == 12:
        return SiteState(
            pos=packed[:, 0:2].clone(),
            color=packed[:, 4:7].clone(),
            log_tau=packed[:, 2].clone(),
            radius=packed[:, 3].clone(),
            aniso_dir=packed[:, 8:10].clone(),
            log_aniso=packed[:, 10].clone(),
        )
    if packed.size(1) == 10:
        return SiteState(
            pos=packed[:, 0:2].clone(),
            color=packed[:, 4:7].clone(),
            log_tau=packed[:, 2].clone(),
            radius=packed[:, 3].clone(),
            aniso_dir=packed[:, 7:9].clone(),
            log_aniso=packed[:, 9].clone(),
        )
    raise ValueError("packed sites must have 10 or 12 columns")


def save_sites_txt(path: Path, sites: SiteState, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# SAD Sites (position_x, position_y, color_r, color_g, color_b, log_tau, radius, aniso_dir_x, aniso_dir_y, log_aniso)\n")
        f.write(f"# Image size: {width} {height}\n")
        f.write(f"# Total sites: {sites.pos.shape[0]}\n")
        f.write(f"# Active sites: {sites.pos.shape[0]}\n")
        for i in range(sites.pos.shape[0]):
            f.write("{:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}\n".format(
                sites.pos[i, 0].item(),
                sites.pos[i, 1].item(),
                sites.color[i, 0].item(),
                sites.color[i, 1].item(),
                sites.color[i, 2].item(),
                sites.log_tau[i].item(),
                sites.radius[i].item(),
                sites.aniso_dir[i, 0].item(),
                sites.aniso_dir[i, 1].item(),
                sites.log_aniso[i].item(),
            ))


def simulate_final_sites(init_sites, max_sites, iters, densify_enabled,
                         densify_start, densify_end, densify_freq, densify_percentile,
                         prune_during_densify, prune_start, prune_end, prune_freq, prune_percentile,
                         max_split_indices):
    actual_sites = init_sites
    active_estimate = init_sites
    max_sites = max(max_sites, init_sites)

    effective_prune_start = prune_start
    if densify_enabled and not prune_during_densify and prune_start < densify_end:
        effective_prune_start = densify_end

    if iters <= 0:
        return active_estimate

    for it in range(iters):
        if (densify_enabled and densify_percentile > 0.0 and it >= densify_start and
                it <= densify_end and (it % max(1, densify_freq) == 0) and actual_sites < max_sites):
            desired = int(active_estimate * densify_percentile)
            available = max_sites - actual_sites
            num_to_split = min(desired, available, max_split_indices)
            if num_to_split > 0:
                actual_sites += num_to_split
                active_estimate += num_to_split

        if (prune_percentile > 0.0 and it >= effective_prune_start and it < prune_end and
                (it % max(1, prune_freq) == 0)):
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


def _jump_step_for_index(step_index: int, width: int, height: int) -> int:
    max_dim = max(width, height)
    pow2 = 1
    while pow2 < max_dim:
        pow2 <<= 1
    if pow2 <= 1:
        return 1
    stages = 0
    tmp = pow2
    while tmp > 1:
        tmp >>= 1
        stages += 1
    stage = min(step_index, max(0, stages - 1))
    step = pow2 >> (stage + 1)
    return max(step, 1)


def _pack_jump_step(step_index: int, width: int, height: int) -> int:
    jump_step = min(_jump_step_for_index(step_index, width, height), 0xFFFF)
    return (jump_step << 16) | (step_index & 0xFFFF)


def _hilbert_bits_for_size(width: int, height: int) -> int:
    max_dim = max(width, height)
    n = 1
    bits = 0
    while n < max_dim:
        n <<= 1
        bits += 1
    return bits


def _candidate_update_plan(iter_idx: int, cand_freq: int, cand_passes: int,
                           init_per_pixel: bool, effective_prune_start: int):
    if init_per_pixel and iter_idx < effective_prune_start:
        return False, 0
    should = cand_freq > 0 and (iter_idx % cand_freq == 0)
    return should, (max(1, cand_passes) if should else 0)


def _compute_psnr(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (render[..., :3] - target[..., :3])
    if mask is not None:
        if mask.dim() == 2:
            mask_use = mask.unsqueeze(-1)
            mask_sum = float(mask.sum().item())
        else:
            mask_use = mask[..., :1]
            mask_sum = float(mask[..., 0].sum().item())
        diff = diff * mask_use
        denom = max(mask_sum, 1.0) * 3.0
    else:
        denom = float(diff.shape[0] * diff.shape[1] * 3)
    mse = (diff * diff).sum().item() / denom
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def load_sad_ops() -> torch._ops:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    py_path = str(repo_root / "backends" / "pytorch")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{py_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = py_path
    py_exec = os.environ.get("PYTHON", sys.executable)
    check = subprocess.run(
        [py_exec, "-c", "import torch_ext.sad_ops as ops; ops.render_sad_padded"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode != 0:
        build_script = repo_root / "backends" / "pytorch" / "build_ext.sh"
        if build_script.exists():
            print("PyTorch SAD extension missing; attempting CMake build...")
            build_env = env.copy()
            build_env["PYTHON"] = py_exec
            result = subprocess.run(["/bin/bash", str(build_script)], env=build_env)
            if result.returncode == 0:
                check = subprocess.run(
                    [py_exec, "-c", "import torch_ext.sad_ops as ops; ops.render_sad_padded"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        if check.returncode != 0:
            raise SystemExit(
                "PyTorch SAD extension failed to load. Rebuild it with the current Python:\n"
                f"  {py_exec} -m pip install -e backends/pytorch/torch_ext --no-build-isolation\n"
                "Or build directly:\n"
                "  backends/pytorch/build_ext.sh"
            )
    import torch_ext.sad_ops  # noqa: F401
    return torch.ops.sad


def render_from_sites(args) -> None:
    device = get_best_device() if args.device is None else torch.device(args.device)
    if device.type not in ("mps", "cuda"):
        raise SystemExit(f"PyTorch render requires MPS or CUDA. Got device: {device}.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this system.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this system.")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    width_override = args.width if args.width > 0 else None
    height_override = args.height if args.height > 0 else None
    sites, src_w, src_h = load_sites(Path(args.render_sites),
                                     width_override=width_override,
                                     height_override=height_override)

    width = args.width if args.width > 0 else src_w
    height = args.height if args.height > 0 else src_h
    if width <= 0 or height <= 0:
        raise SystemExit("Render requires --width/--height or a sites header with image size.")

    if src_w and src_h and (src_w != width or src_h != height):
        sites = scale_sites_to_target(sites, src_w, src_h, width, height)

    sites = SiteState(
        pos=sites.pos.to(device=device, dtype=torch.float32),
        color=sites.color.to(device=device, dtype=torch.float32),
        log_tau=sites.log_tau.to(device=device, dtype=torch.float32),
        radius=sites.radius.to(device=device, dtype=torch.float32),
        aniso_dir=sites.aniso_dir.to(device=device, dtype=torch.float32),
        log_aniso=sites.log_aniso.to(device=device, dtype=torch.float32),
    )

    vops = load_sad_ops()
    use_mps = device.type == "mps"
    use_cuda = device.type == "cuda"
    site_stride = 12 if use_mps else 10

    site_count = int(sites.pos.shape[0])
    sites_padded = pack_sites_padded(sites, site_count, device, stride=site_stride)

    inv_scale_sq = 1.0 / float(max(width, height) ** 2)
    cand_scale = max(1, int(args.cand_downscale))
    cand_width = max(1, (width + cand_scale - 1) // cand_scale)
    cand_height = max(1, (height + cand_scale - 1) // cand_scale)

    cand0A, cand1A = vops.init_candidates(cand_width, cand_height, site_count, 0)
    cand0B = torch.zeros_like(cand0A)
    cand1B = torch.zeros_like(cand1A)

    vops.jfa_seed(cand0A, sites_padded, site_count, cand_scale, cand_width, cand_height)
    if not args.render_no_jfa and not use_cuda:
        step = 1
        max_dim = max(cand_width, cand_height)
        while step < max_dim:
            step <<= 1
        step_size = step // 2
        read_tex = cand0A
        write_tex = cand0B
        while step_size >= 1:
            vops.jfa_flood(read_tex, write_tex, sites_padded, float(inv_scale_sq),
                           site_count, int(step_size), cand_scale, cand_width, cand_height)
            read_tex, write_tex = write_tex, read_tex
            step_size //= 2
        cand0A = read_tex
        cand0B = write_tex

    render_passes = max(0, int(args.render_cand_passes))
    if render_passes > 0:
        packed_sites = torch.zeros((site_count, 8), device=device, dtype=torch.float16)
        vops.pack_candidate_sites(sites_padded, packed_sites, site_count)

        uses_hilbert = args.cand_hilbert_probes > 0 and args.cand_hilbert_window > 0
        dummy_hilbert = torch.zeros((1,), device=device, dtype=torch.int32)
        if uses_hilbert:
            hilbert_bits = _hilbert_bits_for_size(width, height)
            hilbert_padded = ((site_count + 1023) // 1024) * 1024
            hilbert_pairs = torch.zeros((hilbert_padded, 2), device=device, dtype=torch.int32)
            hilbert_order = torch.zeros((site_count,), device=device, dtype=torch.int32)
            hilbert_pos = torch.zeros((site_count,), device=device, dtype=torch.int32)
            vops.build_hilbert_pairs(
                sites_padded, hilbert_pairs, site_count,
                int(hilbert_pairs.size(0)), int(width), int(height), int(hilbert_bits)
            )
            vops.radix_sort_pairs(hilbert_pairs, 0xFFFFFFFF)
            vops.write_hilbert_order(hilbert_pairs, hilbert_order, hilbert_pos, site_count)
            hilbert_order_buf = hilbert_order
            hilbert_pos_buf = hilbert_pos
        else:
            hilbert_order_buf = dummy_hilbert
            hilbert_pos_buf = dummy_hilbert

        jump_pass_index = 0
        for _ in range(render_passes):
            step = _pack_jump_step(jump_pass_index, cand_width, cand_height)
            step_high = jump_pass_index >> 16
            vops.update_candidates_compact(
                cand0A, cand1A, cand0B, cand1B,
                packed_sites,
                hilbert_order_buf,
                hilbert_pos_buf,
                float(inv_scale_sq),
                site_count,
                int(step),
                int(step_high),
                float(args.cand_radius_scale),
                int(args.cand_radius_probes),
                int(args.cand_inject),
                int(args.cand_hilbert_probes if uses_hilbert else 0),
                int(args.cand_hilbert_window if uses_hilbert else 0),
                int(cand_scale),
                int(width),
                int(height),
                int(cand_width),
                int(cand_height),
            )
            cand0A, cand0B = cand0B, cand0A
            cand1A, cand1B = cand1B, cand1A
            jump_pass_index += 1

    render = vops.render_sad_padded(
        cand0A, cand1A, sites_padded, float(inv_scale_sq),
        int(width), int(height), int(cand_width), int(cand_height)
    )
    if render.shape[-1] > 3:
        render = render[..., :3]

    render_img = render.detach().clamp(0, 1).cpu().numpy()
    out_path = args.render_out
    if out_path is None:
        stem = Path(args.render_sites).with_suffix("")
        out_path = str(stem) + "_render.png"
    Image.fromarray((render_img * 255.0).astype(np.uint8)).save(out_path)
    print(f"Saved: {out_path}")

    if args.render_target:
        target_cpu, tw, th = load_image(Path(args.render_target), max_dim=0)
        if tw != width or th != height:
            raise SystemExit(
                f"Render target size {tw}x{th} does not match render size {width}x{height}."
            )
        target = target_cpu.to(dtype=torch.float32)
        render_t = torch.from_numpy(render_img).to(dtype=torch.float32)
        psnr = _compute_psnr(render_t, target, None)
        print(f"Render PSNR: {psnr:.2f} dB")


def main() -> None:
    cfg = load_training_config(CONFIG_PATH)

    parser = argparse.ArgumentParser(description="PyTorch SAD training (GPU kernels).")
    parser.add_argument("image", nargs="?", help="Target image path")
    parser.add_argument("--render", dest="render_sites", default=None,
                        help="Render from a saved sites file (.txt or .json)")
    parser.add_argument("--out", dest="render_out", default=None)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--render-target", default=None,
                        help="Optional target image for PSNR in render mode")
    parser.add_argument("--render-cand-passes", type=int, default=16,
                        help="Candidate refinement passes in render mode")
    parser.add_argument("--render-no-jfa", action="store_true",
                        help="Disable JFA flood before render passes")

    parser.add_argument("--iters", type=int, default=cfg["DEFAULT_ITERS"])
    parser.add_argument("--sites", type=int, default=cfg["DEFAULT_SITES"])
    parser.add_argument("--max-sites", type=int, default=cfg["DEFAULT_MAX_SITES"])
    parser.add_argument("--max-dim", type=int, default=cfg.get("MAX_DIM", 0))
    parser.add_argument("--target-bpp", type=float, default=float(cfg.get("DEFAULT_TARGET_BPP", -1.0)))

    parser.add_argument("--init-per-pixel", action="store_true")
    parser.add_argument("--init-gradient", action="store_true")
    parser.add_argument("--init-gradient-alpha", type=float, default=1.0)
    parser.add_argument("--init-from-sites", default=None)

    parser.add_argument("--densify", dest="densify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--densify-start", type=int, default=cfg["DENSIFY_START"])
    parser.add_argument("--densify-end", type=int, default=cfg["DENSIFY_END"])
    parser.add_argument("--densify-freq", type=int, default=cfg["DENSIFY_FREQ"])
    parser.add_argument("--densify-percentile", type=float, default=cfg["DENSIFY_PERCENTILE"])
    parser.add_argument("--densify-score-alpha", type=float, default=cfg["DENSIFY_SCORE_ALPHA"])
    parser.add_argument("--prune-during-densify", dest="prune_during_densify",
                        action=argparse.BooleanOptionalAction,
                        default=bool(cfg.get("PRUNE_DURING_DENSIFY", True)))
    parser.add_argument("--prune-start", type=int, default=cfg["PRUNE_START"])
    parser.add_argument("--prune-end", type=int, default=cfg["PRUNE_END"])
    parser.add_argument("--prune-freq", type=int, default=cfg["PRUNE_FREQ"])
    parser.add_argument("--prune-percentile", type=float, default=cfg["PRUNE_PERCENTILE"])

    parser.add_argument("--cand-freq", type=int, default=cfg["CAND_UPDATE_FREQ"])
    parser.add_argument("--cand-passes", "--vpt", dest="cand_passes",
                        type=int, default=cfg.get("CAND_UPDATE_PASSES", 1))
    parser.add_argument("--cand-downscale", type=int, default=cfg.get("CAND_DOWNSCALE", 1))
    parser.add_argument("--cand-radius-scale", type=float, default=cfg.get("CAND_RADIUS_SCALE", 64.0))
    parser.add_argument("--cand-radius-probes", type=int, default=cfg.get("CAND_RADIUS_PROBES", 0))
    parser.add_argument("--cand-inject", "--inject-count", dest="cand_inject",
                        type=int, default=cfg.get("CAND_INJECT_COUNT", 0))
    parser.add_argument("--cand-hilbert-window", type=int, default=cfg.get("CAND_HILBERT_WINDOW", 0))
    parser.add_argument("--cand-hilbert-probes", type=int, default=cfg.get("CAND_HILBERT_PROBES", 0))

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--save-freq", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--mask", default=None)

    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--lr-pos", type=float, default=cfg["LR_POS_BASE"])
    parser.add_argument("--lr-tau", type=float, default=cfg["LR_TAU_BASE"])
    parser.add_argument("--lr-radius", type=float, default=cfg["LR_RADIUS_BASE"])
    parser.add_argument("--lr-color", type=float, default=cfg["LR_COLOR_BASE"])
    parser.add_argument("--lr-dir", type=float, default=cfg["LR_DIR_BASE"])
    parser.add_argument("--lr-aniso", type=float, default=cfg["LR_ANISO_BASE"])
    parser.add_argument("--beta1", type=float, default=cfg["BETA1"])
    parser.add_argument("--beta2", type=float, default=cfg["BETA2"])
    parser.add_argument("--eps", type=float, default=cfg["EPS"])
    parser.add_argument("--init-log-tau", type=float, default=cfg["INIT_LOG_TAU"])
    parser.add_argument("--init-radius", type=float, default=cfg["INIT_RADIUS"])

    parser.add_argument("--ssim", dest="ssim_metric", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--export-neighbors", action="store_true")
    parser.add_argument("--export-cand-passes", type=int, default=0)
    parser.add_argument("--packed-psnr", action="store_true")
    parser.add_argument("--trace-frame", type=int, default=None)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--viewer-freq", type=int, default=10)

    args = parser.parse_args()

    if args.render_sites:
        render_from_sites(args)
        return

    if args.image is None:
        parser.error("image is required")

    device = get_best_device() if args.device is None else torch.device(args.device)
    if device.type not in ("mps", "cuda"):
        raise SystemExit(f"PyTorch backend requires MPS or CUDA. Got device: {device}.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this system.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this system.")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    target_cpu, width, height = load_image(Path(args.image), args.max_dim)
    mask_cpu = load_mask(Path(args.mask) if args.mask else None, width, height)

    target = target_cpu.to(device=device, dtype=torch.float32)
    if mask_cpu is not None:
        mask = mask_cpu.to(device=device, dtype=torch.float32)
    else:
        mask = None

    use_gradient = args.init_gradient or (not args.init_per_pixel and not args.init_from_sites)
    init_gradient_flag = 0

    if args.init_from_sites:
        init_sites, init_w, init_h = load_sites(Path(args.init_from_sites))
        init_sites = scale_sites_to_target(init_sites, init_w, init_h, width, height)
        init_sites.log_tau = torch.full_like(init_sites.log_tau, float(args.init_log_tau))
        sites = init_sites
    elif args.init_per_pixel:
        sites = init_sites_per_pixel(target_cpu, args.init_log_tau, args.init_radius)
    elif use_gradient:
        init_gradient_flag = 1
        sites = init_sites_placeholder(args.sites, args.init_log_tau, args.init_radius)
    else:
        init_gradient_flag = 1
        sites = init_sites_placeholder(args.sites, args.init_log_tau, args.init_radius)

    sites = SiteState(
        pos=sites.pos.to(device=device, dtype=torch.float32),
        color=sites.color.to(device=device, dtype=torch.float32),
        log_tau=sites.log_tau.to(device=device, dtype=torch.float32),
        radius=sites.radius.to(device=device, dtype=torch.float32),
        aniso_dir=sites.aniso_dir.to(device=device, dtype=torch.float32),
        log_aniso=sites.log_aniso.to(device=device, dtype=torch.float32),
    )

    if args.target_bpp and args.target_bpp > 0.0:
        prune_end = args.prune_end if args.prune_end > 0 else args.iters - 1
        densify_pct, prune_pct, final_sites, achieved_bpp = solve_target_bpp(
            args.target_bpp,
            width,
            height,
            sites.pos.shape[0],
            args.max_sites,
            args.iters,
            args.densify,
            args.densify_start,
            args.densify_end,
            args.densify_freq,
            args.densify_percentile,
            args.prune_during_densify,
            args.prune_start,
            prune_end,
            args.prune_freq,
            args.prune_percentile,
            MAX_SPLIT_INDICES,
        )
        args.densify_percentile = densify_pct
        args.prune_percentile = prune_pct

    vops = load_sad_ops()

    use_mps = device.type == "mps"
    use_cuda = device.type == "cuda"
    site_stride = 12 if use_mps else 10
    adam_stride = 24 if use_mps else 20

    if use_mps:
        target_render = torch.cat(
            [target, torch.ones((height, width, 1), device=device, dtype=torch.float32)],
            dim=2,
        ).contiguous()
        if mask is None:
            mask_render = torch.ones((height, width, 4), device=device, dtype=torch.float32)
            mask_sum = float(width * height)
        else:
            mask_sum = float(mask.sum().item())
            mask_render = mask.unsqueeze(-1).repeat(1, 1, 4).contiguous()
    else:
        target_render = target.contiguous()
        if mask is None:
            mask_render = torch.ones((height, width), device=device, dtype=torch.float32)
            mask_sum = float(width * height)
        else:
            mask_sum = float(mask.sum().item())
            mask_render = mask.contiguous()

    num_pixels = width * height
    if args.max_sites > 0:
        max_sites_capacity = args.max_sites
    elif args.densify:
        max_sites_capacity = min(num_pixels * 2, max(args.sites * 8, 8192))
    else:
        max_sites_capacity = num_pixels * 2

    buffer_capacity = max(sites.pos.shape[0], max_sites_capacity if args.densify else sites.pos.shape[0])
    sites_padded = pack_sites_padded(sites, buffer_capacity, device, stride=site_stride)

    if use_mps:
        torch.mps.synchronize()
    elif use_cuda:
        torch.cuda.synchronize()

    # Modular training loop using GPU kernels via torch.ops.
    inv_scale_sq = 1.0 / float(max(width, height) ** 2)
    cand_scale = max(1, int(args.cand_downscale))
    cand_width = max(1, (width + cand_scale - 1) // cand_scale)
    cand_height = max(1, (height + cand_scale - 1) // cand_scale)

    # Optional gradient-based initialization (matches Metal backend).
    if init_gradient_flag:
        vops.init_gradient_weighted(
            sites_padded,
            target_render,
            mask_render,
            int(sites.pos.shape[0]),
            float(args.init_log_tau),
            float(args.init_radius),
            float(args.init_gradient_alpha),
        )

    # Candidate buffers (RGBA32Uint textures stored as [M,4] int32 tensors).
    cand0A, cand1A = vops.init_candidates(cand_width, cand_height, int(sites.pos.shape[0]), 0)
    cand0B = torch.zeros_like(cand0A)
    cand1B = torch.zeros_like(cand1A)

    # JFA seed + flood (Metal-style init). CUDA backend skips the flood pass to
    # match the native CUDA initialization sequence.
    vops.jfa_seed(cand0A, sites_padded, int(sites.pos.shape[0]), cand_scale, cand_width, cand_height)
    if not use_cuda:
        step = 1
        max_dim = max(cand_width, cand_height)
        while step < max_dim:
            step <<= 1
        step_size = step // 2
        read_tex = cand0A
        write_tex = cand0B
        while step_size >= 1:
            vops.jfa_flood(read_tex, write_tex, sites_padded, float(inv_scale_sq),
                           int(sites.pos.shape[0]), int(step_size), cand_scale, cand_width, cand_height)
            read_tex, write_tex = write_tex, read_tex
            step_size //= 2
        cand0A = read_tex
        cand0B = write_tex

    # Buffers
    packed_sites = torch.zeros((buffer_capacity, 8), device=device, dtype=torch.float16)
    adam_state = torch.zeros((buffer_capacity, adam_stride), device=device, dtype=torch.float32)
    grad_bufs = [torch.zeros((buffer_capacity,), device=device, dtype=torch.float32) for _ in range(10)]
    removal_delta = torch.zeros((buffer_capacity,), device=device, dtype=torch.float32)
    tau_grad_raw = torch.zeros((buffer_capacity,), device=device, dtype=torch.float32)
    tau_grad_tmp = torch.zeros((buffer_capacity,), device=device, dtype=torch.float32)

    needs_pairs = args.densify or args.prune_percentile > 0.0
    score_pairs_count = max(1, buffer_capacity) if needs_pairs else 0
    pairs_buffer = torch.zeros((score_pairs_count, 2), device=device, dtype=torch.int32) if needs_pairs else None
    split_indices = torch.zeros((MAX_SPLIT_INDICES,), device=device, dtype=torch.int32) if args.densify else None
    prune_indices = torch.zeros((MAX_SPLIT_INDICES,), device=device, dtype=torch.int32) if args.prune_percentile > 0.0 else None

    stat_buffers = None
    if args.densify:
        stat_buffers = [
            torch.zeros((buffer_capacity,), device=device, dtype=torch.float32),  # mass
            torch.zeros((buffer_capacity,), device=device, dtype=torch.float32),  # energy
            torch.zeros((buffer_capacity,), device=device, dtype=torch.float32),  # err_w
            torch.zeros((buffer_capacity,), device=device, dtype=torch.float32),  # err_wx
            torch.zeros((buffer_capacity,), device=device, dtype=torch.float32),  # err_wy
            torch.zeros((buffer_capacity,), device=device, dtype=torch.float32),  # err_wxx
            torch.zeros((buffer_capacity,), device=device, dtype=torch.float32),  # err_wxy
            torch.zeros((buffer_capacity,), device=device, dtype=torch.float32),  # err_wyy
        ]

    uses_hilbert = (args.cand_hilbert_probes > 0 and args.cand_hilbert_window > 0)
    dummy_hilbert = torch.zeros((1,), device=device, dtype=torch.int32)
    hilbert_pairs = None
    hilbert_order = None
    hilbert_pos = None
    hilbert_ready = False
    hilbert_site_count = 0
    hilbert_bits = _hilbert_bits_for_size(width, height)
    if uses_hilbert:
        hilbert_padded = ((buffer_capacity + 1023) // 1024) * 1024
        hilbert_pairs = torch.zeros((hilbert_padded, 2), device=device, dtype=torch.int32)
        hilbert_order = torch.zeros((buffer_capacity,), device=device, dtype=torch.int32)
        hilbert_pos = torch.zeros((buffer_capacity,), device=device, dtype=torch.int32)

    actual_sites = int(sites.pos.shape[0])
    active_estimate = int(sites.pos.shape[0])
    cleared_site_count = 0
    jump_pass_index = 0

    prune_end = args.prune_end if args.prune_end > 0 else args.iters - 1
    effective_prune_start = args.prune_start
    if args.densify and (not args.prune_during_densify) and args.prune_start < args.densify_end:
        effective_prune_start = args.densify_end
    _print_training_overview(
        args,
        device,
        width,
        height,
        actual_sites,
        active_estimate,
        buffer_capacity,
        cand_scale,
    )
    print("Logs | Iter | PSNR | Active | speed | elapsed")

    lr_pos = args.lr_pos * args.lr
    lr_tau = args.lr_tau * args.lr
    lr_radius = args.lr_radius * args.lr
    lr_color = args.lr_color * args.lr
    lr_dir = args.lr_dir * args.lr
    lr_aniso = args.lr_aniso * args.lr

    best_psnr = 0.0
    final_psnr = 0.0
    render_rgba = torch.zeros_like(target_render)

    start_time = time.time()

    for it in range(args.iters):
        should_log = (args.log_freq > 0) and (it % max(1, args.log_freq) == 0 or it == args.iters - 1)
        update_candidates, cand_passes = _candidate_update_plan(
            it, args.cand_freq, args.cand_passes, args.init_per_pixel, effective_prune_start
        )

        desired_splits = max(0, int(active_estimate * args.densify_percentile))
        should_densify = (
            args.densify
            and it >= args.densify_start
            and it <= args.densify_end
            and it % max(1, args.densify_freq) == 0
            and actual_sites < buffer_capacity
            and desired_splits > 0
        )

        should_prune = (
            args.prune_percentile > 0.0
            and it >= effective_prune_start
            and it < prune_end
            and it % max(1, args.prune_freq) == 0
            and pairs_buffer is not None
            and prune_indices is not None
        )

        use_concurrent_candidates = update_candidates and cand_passes == 1 and not should_densify
        pending_candidate_swap = False

        if update_candidates:
            vops.pack_candidate_sites(sites_padded, packed_sites, int(actual_sites))
            if uses_hilbert and hilbert_pairs is not None and hilbert_order is not None and hilbert_pos is not None:
                if (not hilbert_ready) or (hilbert_site_count != actual_sites):
                    vops.build_hilbert_pairs(sites_padded, hilbert_pairs, int(actual_sites),
                                             int(hilbert_pairs.size(0)), int(width), int(height), int(hilbert_bits))
                    vops.radix_sort_pairs(hilbert_pairs, 0xFFFFFFFF)
                    vops.write_hilbert_order(hilbert_pairs, hilbert_order, hilbert_pos, int(actual_sites))
                    hilbert_ready = True
                    hilbert_site_count = actual_sites

        if update_candidates and cand_passes > 0:
            if use_concurrent_candidates:
                step = _pack_jump_step(jump_pass_index, cand_width, cand_height)
                step_high = jump_pass_index >> 16
                vops.update_candidates_compact(
                    cand0A, cand1A, cand0B, cand1B,
                    packed_sites,
                    hilbert_order if uses_hilbert and hilbert_order is not None else dummy_hilbert,
                    hilbert_pos if uses_hilbert and hilbert_pos is not None else dummy_hilbert,
                    float(inv_scale_sq),
                    int(actual_sites),
                    int(step),
                    int(step_high),
                    float(args.cand_radius_scale),
                    int(args.cand_radius_probes),
                    int(args.cand_inject),
                    int(args.cand_hilbert_probes if uses_hilbert else 0),
                    int(args.cand_hilbert_window if uses_hilbert else 0),
                    int(cand_scale),
                    int(width),
                    int(height),
                    int(cand_width),
                    int(cand_height),
                )
                pending_candidate_swap = True
                jump_pass_index += 1
            else:
                for _ in range(cand_passes):
                    step = _pack_jump_step(jump_pass_index, cand_width, cand_height)
                    step_high = jump_pass_index >> 16
                    vops.update_candidates_compact(
                        cand0A, cand1A, cand0B, cand1B,
                        packed_sites,
                        hilbert_order if uses_hilbert and hilbert_order is not None else dummy_hilbert,
                        hilbert_pos if uses_hilbert and hilbert_pos is not None else dummy_hilbert,
                        float(inv_scale_sq),
                        int(actual_sites),
                        int(step),
                        int(step_high),
                        float(args.cand_radius_scale),
                        int(args.cand_radius_probes),
                        int(args.cand_inject),
                        int(args.cand_hilbert_probes if uses_hilbert else 0),
                        int(args.cand_hilbert_window if uses_hilbert else 0),
                        int(cand_scale),
                        int(width),
                        int(height),
                        int(cand_width),
                        int(cand_height),
                    )
                    cand0A, cand0B = cand0B, cand0A
                    cand1A, cand1B = cand1B, cand1A
                    jump_pass_index += 1

        if should_densify and stat_buffers is not None and pairs_buffer is not None and split_indices is not None:
            desired = max(0, int(active_estimate * args.densify_percentile))
            available = buffer_capacity - actual_sites
            num_to_split = min(desired, available, MAX_SPLIT_INDICES)
            if num_to_split > 0:
                for buf in stat_buffers:
                    vops.clear_buffer(buf, int(actual_sites))
                vops.compute_site_stats_tiled(
                    cand0A, cand1A, target_render, mask_render, sites_padded,
                    stat_buffers[0], stat_buffers[1],
                    stat_buffers[2], stat_buffers[3], stat_buffers[4],
                    stat_buffers[5], stat_buffers[6], stat_buffers[7],
                    float(inv_scale_sq), int(actual_sites), int(cand_width), int(cand_height),
                )
                vops.compute_densify_score_pairs(
                    sites_padded, stat_buffers[0], stat_buffers[1],
                    pairs_buffer, int(actual_sites), 1.0, float(args.densify_score_alpha), int(pairs_buffer.size(0))
                )
                vops.radix_sort_pairs(pairs_buffer, 0xFFFFFFFF)
                vops.write_split_indices(pairs_buffer, split_indices, int(num_to_split))
                vops.split_sites(
                    sites_padded, adam_state, split_indices,
                    stat_buffers[0], stat_buffers[2], stat_buffers[3], stat_buffers[4],
                    stat_buffers[5], stat_buffers[6], stat_buffers[7],
                    int(actual_sites), int(num_to_split), target_render
                )
                actual_sites += num_to_split
                active_estimate += num_to_split

        if should_prune:
            vops.clear_buffer(removal_delta, int(actual_sites))

        if actual_sites > cleared_site_count:
            for buf in grad_bufs:
                vops.clear_buffer(buf, int(actual_sites))
            cleared_site_count = actual_sites

        if args.ssim_weight > 0.0:
            render_rgba = vops.render_sad_padded(
                cand0A, cand1A, sites_padded, float(inv_scale_sq),
                int(width), int(height), int(cand_width), int(cand_height)
            )
        vops.compute_gradients_tiled(
            cand0A, cand1A,
            target_render, render_rgba, mask_render,
            sites_padded,
            grad_bufs[0], grad_bufs[1], grad_bufs[2], grad_bufs[3],
            grad_bufs[4], grad_bufs[5], grad_bufs[6],
            grad_bufs[7], grad_bufs[8], grad_bufs[9],
            removal_delta,
            float(inv_scale_sq), int(actual_sites),
            int(1 if should_prune else 0), float(args.ssim_weight),
            int(cand_width), int(cand_height),
        )

        if pending_candidate_swap:
            cand0A, cand0B = cand0B, cand0A
            cand1A, cand1B = cand1B, cand1A

        # Tau diffusion (same schedule as Metal backend)
        if args.ssim_weight >= 0.0:
            tau_grad_raw[:actual_sites].copy_(grad_bufs[2][:actual_sites])
            current_in = grad_bufs[2]
            current_out = tau_grad_tmp
            blend = float(it) / max(1.0, float(args.iters))
            lam = 0.05 * (0.1 + 0.9 * blend)
            for _ in range(4):
                vops.tau_diffuse(cand0A, cand1A, sites_padded,
                                 tau_grad_raw, current_in, current_out,
                                 int(actual_sites), float(lam), int(cand_scale),
                                 int(cand_width), int(cand_height))
                current_in, current_out = current_out, current_in
            if current_in is not grad_bufs[2]:
                grad_bufs[2][:actual_sites].copy_(current_in[:actual_sites])

        vops.adam_update(
            sites_padded, adam_state,
            grad_bufs[0], grad_bufs[1], grad_bufs[2], grad_bufs[3],
            grad_bufs[4], grad_bufs[5], grad_bufs[6],
            grad_bufs[7], grad_bufs[8], grad_bufs[9],
            float(lr_pos), float(lr_tau), float(lr_radius), float(lr_color),
            float(lr_dir), float(lr_aniso),
            float(args.beta1), float(args.beta2), float(args.eps),
            int(it + 1), int(width), int(height),
        )
        if use_cuda:
            for buf in grad_bufs:
                vops.clear_buffer(buf, int(actual_sites))

        if should_prune and pairs_buffer is not None and prune_indices is not None:
            delta_norm = 1.0 / max(1.0, float(mask_sum))
            vops.compute_prune_score_pairs(
                sites_padded, removal_delta, pairs_buffer,
                int(actual_sites), float(delta_norm), int(pairs_buffer.size(0))
            )
            vops.radix_sort_pairs(pairs_buffer, 0xFFFFFFFF)
            desired_prunes = max(0, int(active_estimate * args.prune_percentile))
            num_to_prune = min(desired_prunes, MAX_SPLIT_INDICES)
            if num_to_prune > 0:
                vops.write_split_indices(pairs_buffer, prune_indices, int(num_to_prune))
                vops.prune_sites(sites_padded, prune_indices, int(num_to_prune))
                active_estimate = max(0, active_estimate - num_to_prune)

        if should_log:
            render_rgba = vops.render_sad_padded(
                cand0A, cand1A, sites_padded, float(inv_scale_sq),
                int(width), int(height), int(cand_width), int(cand_height)
            )
            psnr = _compute_psnr(render_rgba, target_render, mask_render)
            best_psnr = max(best_psnr, psnr)
            final_psnr = psnr
            elapsed = max(time.time() - start_time, 1e-6)
            its_per_sec = (it + 1) / elapsed
            print(
                f"Iter {it:4d} | PSNR: {psnr:.2f} dB | Active: {active_estimate}/{actual_sites} | "
                f"{its_per_sec:.1f} it/s | {elapsed:.1f}s"
            )

    train_time = time.time() - start_time

    # Final candidate refresh (optional)
    final_cand_passes = int(args.export_cand_passes) if args.export_cand_passes > 0 else max(1, int(args.cand_passes))
    if final_cand_passes > 0:
        vops.pack_candidate_sites(sites_padded, packed_sites, int(actual_sites))
        for _ in range(final_cand_passes):
            step = _pack_jump_step(jump_pass_index, cand_width, cand_height)
            step_high = jump_pass_index >> 16
            vops.update_candidates_compact(
                cand0A, cand1A, cand0B, cand1B,
                packed_sites,
                hilbert_order if uses_hilbert and hilbert_order is not None else dummy_hilbert,
                hilbert_pos if uses_hilbert and hilbert_pos is not None else dummy_hilbert,
                float(inv_scale_sq),
                int(actual_sites),
                int(step),
                int(step_high),
                float(args.cand_radius_scale),
                int(args.cand_radius_probes),
                int(args.cand_inject),
                int(args.cand_hilbert_probes if uses_hilbert else 0),
                int(args.cand_hilbert_window if uses_hilbert else 0),
                int(cand_scale),
                int(width),
                int(height),
                int(cand_width),
                int(cand_height),
            )
            cand0A, cand0B = cand0B, cand0A
            cand1A, cand1B = cand1B, cand1A
            jump_pass_index += 1

    # Final render for output
    render_rgba = vops.render_sad_padded(
        cand0A, cand1A, sites_padded, float(inv_scale_sq),
        int(width), int(height), int(cand_width), int(cand_height)
    )

    sites_cpu = sites_padded[:actual_sites].detach().cpu()
    active_mask = sites_cpu[:, 0] >= 0
    sites_cpu = sites_cpu[active_mask]
    sites = unpack_sites_padded(sites_cpu)

    render_img = render_rgba.detach().clamp(0, 1)[..., :3].cpu()
    final_img = (render_img * 255.0).to(torch.uint8).numpy()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_img).save(out_dir / "final.png")
    save_sites_txt(out_dir / "sites.txt", sites, width, height)

    print(f"Final PSNR: {final_psnr:.2f} dB (best: {best_psnr:.2f} dB)")

    print(f"Saved final render + sites to {out_dir}")
    total_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} s")
    print(f"Total time: {total_time:.2f} s")


if __name__ == "__main__":
    main()
