"""WebGPU encoder helpers mirroring the Metal/CUDA backend structure."""

from __future__ import annotations

import wgpu

from sad_shaders import (
    INIT_GRADIENT_SHADER,
    INIT_CAND_SHADER,
    SEED_CAND_SHADER,
    VPT_SHADER,
    CANDIDATE_PACK_SHADER,
    JFA_CLEAR_SHADER,
    JFA_FLOOD_SHADER,
    RENDER_SHADER,
    GRADIENTS_TILED_SHADER,
    ADAM_SHADER,
    STATS_SHADER,
    SPLIT_SHADER,
    PRUNE_SHADER,
    CLEAR_BUFFER_U32_SHADER,
    CLEAR_BUFFER_I32_SHADER,
    TAU_EXTRACT_SHADER,
    TAU_DIFFUSE_SHADER,
    TAU_WRITEBACK_SHADER,
    SCORE_PAIRS_DENSIFY_SHADER,
    SCORE_PAIRS_PRUNE_SHADER,
    WRITE_INDICES_SHADER,
    HILBERT_SHADER,
)


def _make_pipeline(device: wgpu.GPUDevice, shader_code: str, entry_point: str = "main") -> wgpu.GPUComputePipeline:
    module = device.create_shader_module(code=shader_code)
    return device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": entry_point})


def _get_texture_view(view_cache: dict[int, wgpu.GPUTextureView], texture: wgpu.GPUTexture) -> wgpu.GPUTextureView:
    key = id(texture)
    view = view_cache.get(key)
    if view is None:
        view = texture.create_view()
        view_cache[key] = view
    return view


def _bind_group_key(layout: wgpu.GPUBindGroupLayout, entries: list[dict]) -> tuple:
    key_parts = [id(layout)]
    for entry in entries:
        binding = entry["binding"]
        resource = entry["resource"]
        if isinstance(resource, dict) and "buffer" in resource:
            key_parts.append(("b", binding, id(resource["buffer"])))
        else:
            key_parts.append(("t", binding, id(resource)))
    return tuple(key_parts)


def _get_bind_group(device: wgpu.GPUDevice, cache: dict[tuple, wgpu.GPUBindGroup],
                    layout: wgpu.GPUBindGroupLayout, entries: list[dict]) -> wgpu.GPUBindGroup:
    key = _bind_group_key(layout, entries)
    bg = cache.get(key)
    if bg is None:
        bg = device.create_bind_group(layout=layout, entries=entries)
        cache[key] = bg
    return bg


class InitGradientEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, INIT_GRADIENT_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               seed_counter: wgpu.GPUBuffer, init_params: wgpu.GPUBuffer,
               target_tex: wgpu.GPUTexture, mask_tex: wgpu.GPUTexture,
               num_sites: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": seed_counter}},
            {"binding": 2, "resource": {"buffer": init_params}},
            {"binding": 3, "resource": _get_texture_view(self._view_cache, target_tex)},
            {"binding": 4, "resource": _get_texture_view(self._view_cache, mask_tex)},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((num_sites + 63) // 64, 1, 1)
        cpass.end()


class CandidatesEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.init_pipeline = _make_pipeline(device, INIT_CAND_SHADER)
        self.update_pipeline = _make_pipeline(device, VPT_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode_init(self, encoder: wgpu.GPUCommandEncoder, params_buf: wgpu.GPUBuffer,
                    cand0: wgpu.GPUTexture, cand1: wgpu.GPUTexture,
                    width: int, height: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": params_buf}},
            {"binding": 1, "resource": _get_texture_view(self._view_cache, cand0)},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand1)},
        ]
        layout = self.init_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.init_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((width + 15) // 16, (height + 15) // 16, 1)
        cpass.end()

    def encode_update(self, encoder: wgpu.GPUCommandEncoder, packed_sites_buffer: wgpu.GPUBuffer,
                      params_buf: wgpu.GPUBuffer, cand0_in: wgpu.GPUTexture,
                      cand1_in: wgpu.GPUTexture, cand0_out: wgpu.GPUTexture,
                      cand1_out: wgpu.GPUTexture, hilbert_order: wgpu.GPUBuffer,
                      hilbert_pos: wgpu.GPUBuffer, width: int, height: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": packed_sites_buffer}},
            {"binding": 1, "resource": {"buffer": params_buf}},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand0_in)},
            {"binding": 3, "resource": _get_texture_view(self._view_cache, cand1_in)},
            {"binding": 4, "resource": _get_texture_view(self._view_cache, cand0_out)},
            {"binding": 5, "resource": _get_texture_view(self._view_cache, cand1_out)},
            {"binding": 6, "resource": {"buffer": hilbert_order}},
            {"binding": 7, "resource": {"buffer": hilbert_pos}},
        ]
        layout = self.update_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.update_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((width + 15) // 16, (height + 15) // 16, 1)
        cpass.end()


class HilbertEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.build_pipeline = _make_pipeline(device, HILBERT_SHADER, entry_point="buildHilbertPairs")
        self.write_pipeline = _make_pipeline(device, HILBERT_SHADER, entry_point="writeHilbertOrder")
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode_pairs(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
                     pairs_buffer: wgpu.GPUBuffer, params_buf: wgpu.GPUBuffer,
                     padded_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": pairs_buffer}},
            {"binding": 2, "resource": {"buffer": params_buf}},
        ]
        layout = self.build_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.build_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((padded_count + 255) // 256, 1, 1)
        cpass.end()

    def encode_order(self, encoder: wgpu.GPUCommandEncoder, pairs_buffer: wgpu.GPUBuffer,
                     order_buffer: wgpu.GPUBuffer, pos_buffer: wgpu.GPUBuffer,
                     params_buf: wgpu.GPUBuffer, site_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": pairs_buffer}},
            {"binding": 1, "resource": {"buffer": order_buffer}},
            {"binding": 2, "resource": {"buffer": pos_buffer}},
            {"binding": 3, "resource": {"buffer": params_buf}},
        ]
        layout = self.write_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.write_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((site_count + 255) // 256, 1, 1)
        cpass.end()


class SeedCandidatesEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, SEED_CAND_SHADER)
        self.update_pipeline = _make_pipeline(device, VPT_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               params_buf: wgpu.GPUBuffer, cand0_in: wgpu.GPUTexture,
               cand0_out: wgpu.GPUTexture, site_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": params_buf}},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand0_in)},
            {"binding": 3, "resource": _get_texture_view(self._view_cache, cand0_out)},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((site_count + 63) // 64, 1, 1)
        cpass.end()

    def encode_update(self, encoder: wgpu.GPUCommandEncoder, packed_sites_buffer: wgpu.GPUBuffer,
                      params_buf: wgpu.GPUBuffer, cand0_in: wgpu.GPUTexture,
                      cand1_in: wgpu.GPUTexture, cand0_out: wgpu.GPUTexture,
                      cand1_out: wgpu.GPUTexture, hilbert_order: wgpu.GPUBuffer,
                      hilbert_pos: wgpu.GPUBuffer, width: int, height: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": packed_sites_buffer}},
            {"binding": 1, "resource": {"buffer": params_buf}},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand0_in)},
            {"binding": 3, "resource": _get_texture_view(self._view_cache, cand1_in)},
            {"binding": 4, "resource": _get_texture_view(self._view_cache, cand0_out)},
            {"binding": 5, "resource": _get_texture_view(self._view_cache, cand1_out)},
            {"binding": 6, "resource": {"buffer": hilbert_order}},
            {"binding": 7, "resource": {"buffer": hilbert_pos}},
        ]
        layout = self.update_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.update_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((width + 15) // 16, (height + 15) // 16, 1)
        cpass.end()


class JFAClearEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, JFA_CLEAR_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, params_buf: wgpu.GPUBuffer,
               cand0: wgpu.GPUTexture, cand1: wgpu.GPUTexture,
               width: int, height: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": params_buf}},
            {"binding": 1, "resource": _get_texture_view(self._view_cache, cand0)},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand1)},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((width + 15) // 16, (height + 15) // 16, 1)
        cpass.end()


class JFAFloodEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, JFA_FLOOD_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               params_buf: wgpu.GPUBuffer, cand_in: wgpu.GPUTexture,
               cand_out: wgpu.GPUTexture, width: int, height: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": params_buf}},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand_in)},
            {"binding": 3, "resource": _get_texture_view(self._view_cache, cand_out)},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((width + 15) // 16, (height + 15) // 16, 1)
        cpass.end()


class CandidatePackEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, CANDIDATE_PACK_SHADER)
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               packed_buffer: wgpu.GPUBuffer, params_buf: wgpu.GPUBuffer, site_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": packed_buffer}},
            {"binding": 2, "resource": {"buffer": params_buf}},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((site_count + 255) // 256, 1, 1)
        cpass.end()


class RenderEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, RENDER_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               params_buf: wgpu.GPUBuffer, cand0: wgpu.GPUTexture, cand1: wgpu.GPUTexture,
               output_tex: wgpu.GPUTexture, width: int, height: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": params_buf}},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand0)},
            {"binding": 3, "resource": _get_texture_view(self._view_cache, cand1)},
            {"binding": 4, "resource": _get_texture_view(self._view_cache, output_tex)},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((width + 15) // 16, (height + 15) // 16, 1)
        cpass.end()


class GradientsEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, GRADIENTS_TILED_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               grad_params: wgpu.GPUBuffer, cand0: wgpu.GPUTexture, cand1: wgpu.GPUTexture,
               target_tex: wgpu.GPUTexture, mask_tex: wgpu.GPUTexture,
               grad_buffers: list[wgpu.GPUBuffer],
               removal_delta: wgpu.GPUBuffer, width: int, height: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": grad_params}},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand0)},
            {"binding": 3, "resource": _get_texture_view(self._view_cache, cand1)},
            {"binding": 4, "resource": _get_texture_view(self._view_cache, target_tex)},
            {"binding": 5, "resource": {"buffer": grad_buffers[0]}},
            {"binding": 6, "resource": {"buffer": grad_buffers[1]}},
            {"binding": 7, "resource": {"buffer": grad_buffers[2]}},
            {"binding": 8, "resource": {"buffer": grad_buffers[3]}},
            {"binding": 9, "resource": {"buffer": grad_buffers[4]}},
            {"binding": 10, "resource": {"buffer": grad_buffers[5]}},
            {"binding": 11, "resource": {"buffer": grad_buffers[6]}},
            {"binding": 12, "resource": {"buffer": grad_buffers[7]}},
            {"binding": 13, "resource": {"buffer": grad_buffers[8]}},
            {"binding": 14, "resource": {"buffer": grad_buffers[9]}},
            {"binding": 15, "resource": {"buffer": removal_delta}},
            {"binding": 16, "resource": _get_texture_view(self._view_cache, mask_tex)},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((width + 15) // 16, (height + 15) // 16, 1)
        cpass.end()


class AdamEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, ADAM_SHADER)
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               adam_buffer: wgpu.GPUBuffer, grad_buffers: list[wgpu.GPUBuffer],
               adam_params: wgpu.GPUBuffer, site_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": adam_buffer}},
            {"binding": 2, "resource": {"buffer": grad_buffers[0]}},
            {"binding": 3, "resource": {"buffer": grad_buffers[1]}},
            {"binding": 4, "resource": {"buffer": grad_buffers[2]}},
            {"binding": 5, "resource": {"buffer": grad_buffers[3]}},
            {"binding": 6, "resource": {"buffer": grad_buffers[4]}},
            {"binding": 7, "resource": {"buffer": grad_buffers[5]}},
            {"binding": 8, "resource": {"buffer": grad_buffers[6]}},
            {"binding": 9, "resource": {"buffer": grad_buffers[7]}},
            {"binding": 10, "resource": {"buffer": grad_buffers[8]}},
            {"binding": 11, "resource": {"buffer": grad_buffers[9]}},
            {"binding": 12, "resource": {"buffer": adam_params}},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((site_count + 255) // 256, 1, 1)
        cpass.end()


class TauEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.extract_pipeline = _make_pipeline(device, TAU_EXTRACT_SHADER)
        self.diffuse_pipeline = _make_pipeline(device, TAU_DIFFUSE_SHADER)
        self.writeback_pipeline = _make_pipeline(device, TAU_WRITEBACK_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode_extract(self, encoder: wgpu.GPUCommandEncoder, grad_in: wgpu.GPUBuffer,
                       grad_out: wgpu.GPUBuffer, clear_params: wgpu.GPUBuffer,
                       site_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": grad_in}},
            {"binding": 1, "resource": {"buffer": grad_out}},
            {"binding": 2, "resource": {"buffer": clear_params}},
        ]
        layout = self.extract_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.extract_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((site_count + 255) // 256, 1, 1)
        cpass.end()

    def encode_diffuse(self, encoder: wgpu.GPUCommandEncoder, cand0: wgpu.GPUTexture,
                       cand1: wgpu.GPUTexture, sites_buffer: wgpu.GPUBuffer,
                       grad_raw: wgpu.GPUBuffer, grad_in: wgpu.GPUBuffer,
                       grad_out: wgpu.GPUBuffer, tau_params: wgpu.GPUBuffer,
                       site_count: int) -> None:
        entries = [
            {"binding": 0, "resource": _get_texture_view(self._view_cache, cand0)},
            {"binding": 1, "resource": _get_texture_view(self._view_cache, cand1)},
            {"binding": 2, "resource": {"buffer": sites_buffer}},
            {"binding": 3, "resource": {"buffer": grad_raw}},
            {"binding": 4, "resource": {"buffer": grad_in}},
            {"binding": 5, "resource": {"buffer": grad_out}},
            {"binding": 6, "resource": {"buffer": tau_params}},
        ]
        layout = self.diffuse_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.diffuse_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((site_count + 255) // 256, 1, 1)
        cpass.end()

    def encode_writeback(self, encoder: wgpu.GPUCommandEncoder, grad_in: wgpu.GPUBuffer,
                         grad_out: wgpu.GPUBuffer, clear_params: wgpu.GPUBuffer,
                         site_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": grad_in}},
            {"binding": 1, "resource": {"buffer": grad_out}},
            {"binding": 2, "resource": {"buffer": clear_params}},
        ]
        layout = self.writeback_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.writeback_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((site_count + 255) // 256, 1, 1)
        cpass.end()


class StatsEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, STATS_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               grad_params: wgpu.GPUBuffer, cand0: wgpu.GPUTexture, cand1: wgpu.GPUTexture,
               target_tex: wgpu.GPUTexture, mask_tex: wgpu.GPUTexture,
               stat_buffers: list[wgpu.GPUBuffer],
               width: int, height: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": grad_params}},
            {"binding": 2, "resource": _get_texture_view(self._view_cache, cand0)},
            {"binding": 3, "resource": _get_texture_view(self._view_cache, cand1)},
            {"binding": 4, "resource": _get_texture_view(self._view_cache, target_tex)},
            {"binding": 5, "resource": {"buffer": stat_buffers[0]}},
            {"binding": 6, "resource": {"buffer": stat_buffers[1]}},
            {"binding": 7, "resource": {"buffer": stat_buffers[2]}},
            {"binding": 8, "resource": {"buffer": stat_buffers[3]}},
            {"binding": 9, "resource": {"buffer": stat_buffers[4]}},
            {"binding": 10, "resource": {"buffer": stat_buffers[5]}},
            {"binding": 11, "resource": {"buffer": stat_buffers[6]}},
            {"binding": 12, "resource": {"buffer": stat_buffers[7]}},
            {"binding": 13, "resource": _get_texture_view(self._view_cache, mask_tex)},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((width + 15) // 16, (height + 15) // 16, 1)
        cpass.end()


class SplitEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, SPLIT_SHADER)
        self._view_cache: dict[int, wgpu.GPUTextureView] = {}
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               adam_buffer: wgpu.GPUBuffer, split_indices: wgpu.GPUBuffer,
               stat_buffers: list[wgpu.GPUBuffer], split_params: wgpu.GPUBuffer,
               target_tex: wgpu.GPUTexture, num_to_split: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": adam_buffer}},
            {"binding": 2, "resource": {"buffer": split_indices}},
            {"binding": 3, "resource": {"buffer": stat_buffers[0]}},
            {"binding": 4, "resource": {"buffer": stat_buffers[2]}},
            {"binding": 5, "resource": {"buffer": stat_buffers[3]}},
            {"binding": 6, "resource": {"buffer": stat_buffers[4]}},
            {"binding": 7, "resource": {"buffer": stat_buffers[5]}},
            {"binding": 8, "resource": {"buffer": stat_buffers[6]}},
            {"binding": 9, "resource": {"buffer": stat_buffers[7]}},
            {"binding": 10, "resource": {"buffer": split_params}},
            {"binding": 11, "resource": _get_texture_view(self._view_cache, target_tex)},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((num_to_split + 255) // 256, 1, 1)
        cpass.end()


class PruneEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, PRUNE_SHADER)
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
               indices: wgpu.GPUBuffer, prune_params: wgpu.GPUBuffer,
               count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": indices}},
            {"binding": 2, "resource": {"buffer": prune_params}},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((count + 255) // 256, 1, 1)
        cpass.end()


class ClearBufferEncoder:
    def __init__(self, device: wgpu.GPUDevice, *, use_i32: bool = False) -> None:
        self.device = device
        shader = CLEAR_BUFFER_I32_SHADER if use_i32 else CLEAR_BUFFER_U32_SHADER
        self.pipeline = _make_pipeline(device, shader)
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, buffer: wgpu.GPUBuffer,
               clear_params: wgpu.GPUBuffer, count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": buffer}},
            {"binding": 1, "resource": {"buffer": clear_params}},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((count + 255) // 256, 1, 1)
        cpass.end()


class ScorePairsEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.densify_pipeline = _make_pipeline(device, SCORE_PAIRS_DENSIFY_SHADER)
        self.prune_pipeline = _make_pipeline(device, SCORE_PAIRS_PRUNE_SHADER)
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode_densify(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
                       mass: wgpu.GPUBuffer, energy: wgpu.GPUBuffer, pairs: wgpu.GPUBuffer,
                       params_buf: wgpu.GPUBuffer, pair_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": mass}},
            {"binding": 2, "resource": {"buffer": energy}},
            {"binding": 3, "resource": {"buffer": pairs}},
            {"binding": 4, "resource": {"buffer": params_buf}},
        ]
        layout = self.densify_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.densify_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((pair_count + 255) // 256, 1, 1)
        cpass.end()

    def encode_prune(self, encoder: wgpu.GPUCommandEncoder, sites_buffer: wgpu.GPUBuffer,
                     removal_delta: wgpu.GPUBuffer, pairs: wgpu.GPUBuffer,
                     params_buf: wgpu.GPUBuffer, pair_count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sites_buffer}},
            {"binding": 1, "resource": {"buffer": removal_delta}},
            {"binding": 2, "resource": {"buffer": pairs}},
            {"binding": 3, "resource": {"buffer": params_buf}},
        ]
        layout = self.prune_pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.prune_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((pair_count + 255) // 256, 1, 1)
        cpass.end()


class WriteIndicesEncoder:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.pipeline = _make_pipeline(device, WRITE_INDICES_SHADER)
        self._bg_cache: dict[tuple, wgpu.GPUBindGroup] = {}

    def encode(self, encoder: wgpu.GPUCommandEncoder, sorted_pairs: wgpu.GPUBuffer,
               indices: wgpu.GPUBuffer, params_buf: wgpu.GPUBuffer,
               count: int) -> None:
        entries = [
            {"binding": 0, "resource": {"buffer": sorted_pairs}},
            {"binding": 1, "resource": {"buffer": indices}},
            {"binding": 2, "resource": {"buffer": params_buf}},
        ]
        layout = self.pipeline.get_bind_group_layout(0)
        bg = _get_bind_group(self.device, self._bg_cache, layout, entries)
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((count + 255) // 256, 1, 1)
        cpass.end()
