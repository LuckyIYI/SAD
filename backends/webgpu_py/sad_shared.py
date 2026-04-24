"""Shared helpers for SAD training and visualization."""

import json
import struct
from pathlib import Path

import numpy as np
import wgpu

CONFIG_PATH = Path(__file__).resolve().parents[2] / "training_config.json"


def _load_training_config(path: Path) -> dict:
    defaults = {
        "CAND_RADIUS_SCALE": 64.0,
        "CAND_RADIUS_PROBES": 0,
        "CAND_INJECT_COUNT": 16,
        "CAND_DOWNSCALE": 1,
        "CAND_HILBERT_WINDOW": 0,
        "CAND_HILBERT_PROBES": 0,
    }
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for key in defaults:
                if key in data:
                    defaults[key] = data[key]
        except (OSError, json.JSONDecodeError):
            pass
    return defaults


_CONFIG = _load_training_config(CONFIG_PATH)
DEFAULT_RADIUS_SCALE = float(_CONFIG["CAND_RADIUS_SCALE"])
DEFAULT_RADIUS_PROBES = int(_CONFIG["CAND_RADIUS_PROBES"])
DEFAULT_INJECT_COUNT = int(_CONFIG["CAND_INJECT_COUNT"])
DEFAULT_CAND_DOWNSCALE = int(_CONFIG["CAND_DOWNSCALE"])
DEFAULT_HILBERT_WINDOW = int(_CONFIG["CAND_HILBERT_WINDOW"])
DEFAULT_HILBERT_PROBES = int(_CONFIG["CAND_HILBERT_PROBES"])


def pack_jump_step(step_index: int, width: int, height: int) -> int:
    max_dim = max(width, height)
    pow2 = 1
    while pow2 < max_dim:
        pow2 <<= 1
    if pow2 <= 1:
        return 1
    stages = pow2.bit_length() - 1
    stage = min(step_index, max(0, stages - 1))
    step = pow2 >> (stage + 1)
    step = max(step, 1)
    step = min(step, 0xFFFF)
    return (step << 16) | (step_index & 0xFFFF)


def create_params_buffer(device, width, height, site_count, step=0, seed=0,
                         radius_scale=DEFAULT_RADIUS_SCALE,
                         radius_probes=DEFAULT_RADIUS_PROBES,
                         inject_count=DEFAULT_INJECT_COUNT,
                         hilbert_probes=DEFAULT_HILBERT_PROBES,
                         hilbert_window=DEFAULT_HILBERT_WINDOW,
                         cand_downscale=DEFAULT_CAND_DOWNSCALE,
                         cand_width=None,
                         cand_height=None):
    if cand_width is None:
        cand_width = width
    if cand_height is None:
        cand_height = height
    cand_downscale = max(1, int(cand_downscale))
    scale = max(width, height)
    inv_scale_sq = 1.0 / (scale * scale)
    data = struct.pack(
        "<IIIIfIfIIIIIIIII",
        int(width),
        int(height),
        int(site_count),
        int(step),
        float(inv_scale_sq),
        int(seed),
        float(radius_scale),
        int(radius_probes),
        int(inject_count),
        int(hilbert_probes),
        int(hilbert_window),
        int(cand_downscale),
        int(cand_width),
        int(cand_height),
        0,
        0,
    )
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def update_params_buffer(queue, buffer, width, height, site_count, step, seed,
                         radius_scale=DEFAULT_RADIUS_SCALE,
                         radius_probes=DEFAULT_RADIUS_PROBES,
                         inject_count=DEFAULT_INJECT_COUNT,
                         hilbert_probes=DEFAULT_HILBERT_PROBES,
                         hilbert_window=DEFAULT_HILBERT_WINDOW,
                         cand_downscale=DEFAULT_CAND_DOWNSCALE,
                         cand_width=None,
                         cand_height=None):
    if cand_width is None:
        cand_width = width
    if cand_height is None:
        cand_height = height
    cand_downscale = max(1, int(cand_downscale))
    scale = max(width, height)
    inv_scale_sq = 1.0 / (scale * scale)
    data = struct.pack(
        "<IIIIfIfIIIIIIIII",
        int(width),
        int(height),
        int(site_count),
        int(step),
        float(inv_scale_sq),
        int(seed),
        float(radius_scale),
        int(radius_probes),
        int(inject_count),
        int(hilbert_probes),
        int(hilbert_window),
        int(cand_downscale),
        int(cand_width),
        int(cand_height),
        0,
        0,
    )
    queue.write_buffer(buffer, 0, data)


def create_hilbert_params_buffer(device, site_count, padded_count, width, height, bits):
    data = struct.pack(
        "<8I",
        int(site_count),
        int(padded_count),
        int(width),
        int(height),
        int(bits),
        0,
        0,
        0,
    )
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def update_hilbert_params_buffer(queue, buffer, site_count, padded_count, width, height, bits):
    data = struct.pack(
        "<8I",
        int(site_count),
        int(padded_count),
        int(width),
        int(height),
        int(bits),
        0,
        0,
        0,
    )
    queue.write_buffer(buffer, 0, data)


def create_clear_params_buffer(device, count):
    data = struct.pack("<IIII", int(count), 0, 0, 0)
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)


def write_texture_rgba32float(device, texture, data, width, height):
    bytes_per_pixel = 16
    bytes_per_row = width * bytes_per_pixel
    aligned = (bytes_per_row + 255) // 256 * 256
    if aligned == bytes_per_row:
        device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data.tobytes(),
            {"bytes_per_row": bytes_per_row, "rows_per_image": height},
            (width, height, 1),
        )
        return

    padded = np.zeros((height, aligned // 4), dtype=np.float32)
    row_f = data.reshape(height, width * 4)
    padded[:, :width * 4] = row_f

    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        padded.tobytes(),
        {"bytes_per_row": aligned, "rows_per_image": height},
        (width, height, 1),
    )


def read_texture_rgba32float(device, texture, width, height) -> np.ndarray:
    bytes_per_pixel = 16
    bytes_per_row = width * bytes_per_pixel
    aligned_bytes_per_row = ((bytes_per_row + 255) // 256) * 256

    data = device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"bytes_per_row": aligned_bytes_per_row, "rows_per_image": height},
        (width, height, 1),
    )

    data_u8 = np.frombuffer(data, dtype=np.uint8)
    img_data = np.zeros((height, width, 4), dtype=np.float32)
    for y in range(height):
        row_start = y * aligned_bytes_per_row
        row_bytes = data_u8[row_start: row_start + bytes_per_row]
        row_f32 = row_bytes.view(np.float32).reshape(width, 4)
        img_data[y, :, :] = row_f32

    return img_data
