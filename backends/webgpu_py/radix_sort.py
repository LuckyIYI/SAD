"""Radix sort helper for uint2 pairs in WebGPU."""

from __future__ import annotations

import struct

import wgpu

from sad_shaders import RADIX_SORT_SHADER


class RadixSortUInt2:
    def __init__(self, device: wgpu.GPUDevice, padded_count: int) -> None:
        self.device = device
        self.padded_count = int(padded_count)

        block_size = 256
        grain = 4
        elements_per_block = block_size * grain

        self.grid_size = (self.padded_count + elements_per_block - 1) // elements_per_block
        self.hist_length = 256 * self.grid_size
        self.hist_blocks = (self.hist_length + block_size - 1) // block_size

        self.hist_buffer = device.create_buffer(
            size=self.hist_length * 4,
            usage=wgpu.BufferUsage.STORAGE,
        )
        self.block_sums = device.create_buffer(
            size=self.hist_blocks * 4,
            usage=wgpu.BufferUsage.STORAGE,
        )
        self.scratch = device.create_buffer(
            size=self.padded_count * 8,
            usage=wgpu.BufferUsage.STORAGE,
        )
        self.params_buffers = []
        for shift in (0, 8, 16, 24):
            data = struct.pack("<4I", self.padded_count, int(shift), 0, 0)
            self.params_buffers.append(
                device.create_buffer_with_data(
                    data=data,
                    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                )
            )

        module = device.create_shader_module(code=RADIX_SORT_SHADER)
        self.hist_pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "radixHistogramUInt2"},
        )
        self.scan_blocks_pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "radixScanHistogramBlocks"},
        )
        self.scan_block_sums_pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "radixExclusiveScanBlockSums"},
        )
        self.apply_offsets_pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "radixApplyOffsets"},
        )
        self.scatter_pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "radixScatterUInt2"},
        )

    def encode(self, encoder: wgpu.GPUCommandEncoder, data_buffer: wgpu.GPUBuffer,
               max_key_exclusive: int = 0xffffffff) -> None:
        if self.padded_count <= 0:
            return

        passes = 2 if max_key_exclusive <= (1 << 16) else 4
        input_buf = data_buffer
        output_buf = self.scratch

        for pass_index in range(passes):
            params_buf = self.params_buffers[pass_index]

            hist_bg = self.device.create_bind_group(
                layout=self.hist_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": input_buf}},
                    {"binding": 1, "resource": {"buffer": self.hist_buffer}},
                    {"binding": 2, "resource": {"buffer": params_buf}},
                ],
            )
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self.hist_pipeline)
            cpass.set_bind_group(0, hist_bg)
            cpass.dispatch_workgroups(self.grid_size, 1, 1)
            cpass.end()

            scan_bg = self.device.create_bind_group(
                layout=self.scan_blocks_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self.hist_buffer}},
                    {"binding": 1, "resource": {"buffer": self.block_sums}},
                    {"binding": 2, "resource": {"buffer": params_buf}},
                ],
            )
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self.scan_blocks_pipeline)
            cpass.set_bind_group(0, scan_bg)
            cpass.dispatch_workgroups(self.hist_blocks, 1, 1)
            cpass.end()

            scan_sums_bg = self.device.create_bind_group(
                layout=self.scan_block_sums_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self.block_sums}},
                    {"binding": 1, "resource": {"buffer": params_buf}},
                ],
            )
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self.scan_block_sums_pipeline)
            cpass.set_bind_group(0, scan_sums_bg)
            cpass.dispatch_workgroups(1, 1, 1)
            cpass.end()

            apply_bg = self.device.create_bind_group(
                layout=self.apply_offsets_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self.hist_buffer}},
                    {"binding": 1, "resource": {"buffer": self.block_sums}},
                    {"binding": 2, "resource": {"buffer": params_buf}},
                ],
            )
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self.apply_offsets_pipeline)
            cpass.set_bind_group(0, apply_bg)
            cpass.dispatch_workgroups(self.hist_blocks, 1, 1)
            cpass.end()

            scatter_bg = self.device.create_bind_group(
                layout=self.scatter_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": input_buf}},
                    {"binding": 1, "resource": {"buffer": output_buf}},
                    {"binding": 2, "resource": {"buffer": self.hist_buffer}},
                    {"binding": 3, "resource": {"buffer": params_buf}},
                ],
            )
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self.scatter_pipeline)
            cpass.set_bind_group(0, scatter_bg)
            cpass.dispatch_workgroups(self.grid_size, 1, 1)
            cpass.end()

            input_buf, output_buf = output_buf, input_buf

        if passes % 2 != 0:
            raise RuntimeError("RadixSortUInt2 expects an even number of passes.")
