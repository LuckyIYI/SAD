// Direct port of backends/webgpu_py/radix_sort.py.

export class RadixSortUInt2 {
  constructor(device, shaders, paddedCount) {
    this.device = device;
    this.paddedCount = paddedCount | 0;

    const blockSize = 256;
    const grain = 4;
    const elementsPerBlock = blockSize * grain;

    this.gridSize = Math.floor((this.paddedCount + elementsPerBlock - 1) / elementsPerBlock);
    this.histLength = 256 * this.gridSize;
    this.histBlocks = Math.floor((this.histLength + blockSize - 1) / blockSize);

    this.histBuffer = device.createBuffer({
      size: Math.max(4, this.histLength * 4),
      usage: GPUBufferUsage.STORAGE,
    });
    this.blockSums = device.createBuffer({
      size: Math.max(4, this.histBlocks * 4),
      usage: GPUBufferUsage.STORAGE,
    });
    this.scratch = device.createBuffer({
      size: Math.max(8, this.paddedCount * 8),
      usage: GPUBufferUsage.STORAGE,
    });

    this.paramsBuffers = [];
    for (const shift of [0, 8, 16, 24]) {
      const bytes = new ArrayBuffer(16);
      const u = new Uint32Array(bytes);
      u[0] = this.paddedCount >>> 0;
      u[1] = shift >>> 0;
      u[2] = 0;
      u[3] = 0;
      const buf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, bytes);
      this.paramsBuffers.push(buf);
    }

    const module = device.createShaderModule({ code: shaders.RADIX_SORT });
    this.histPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "radixHistogramUInt2" },
    });
    this.scanBlocksPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "radixScanHistogramBlocks" },
    });
    this.scanBlockSumsPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "radixExclusiveScanBlockSums" },
    });
    this.applyOffsetsPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "radixApplyOffsets" },
    });
    this.scatterPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "radixScatterUInt2" },
    });
  }

  encode(encoder, dataBuffer, maxKeyExclusive = 0xffffffff) {
    if (this.paddedCount <= 0) return;

    const passes = maxKeyExclusive <= (1 << 16) ? 2 : 4;
    let inputBuf = dataBuffer;
    let outputBuf = this.scratch;

    for (let passIndex = 0; passIndex < passes; passIndex += 1) {
      const paramsBuf = this.paramsBuffers[passIndex];

      {
        const bg = this.device.createBindGroup({
          layout: this.histPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: inputBuf } },
            { binding: 1, resource: { buffer: this.histBuffer } },
            { binding: 2, resource: { buffer: paramsBuf } },
          ],
        });
        const p = encoder.beginComputePass();
        p.setPipeline(this.histPipeline);
        p.setBindGroup(0, bg);
        p.dispatchWorkgroups(this.gridSize, 1, 1);
        p.end();
      }

      {
        const bg = this.device.createBindGroup({
          layout: this.scanBlocksPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.histBuffer } },
            { binding: 1, resource: { buffer: this.blockSums } },
            { binding: 2, resource: { buffer: paramsBuf } },
          ],
        });
        const p = encoder.beginComputePass();
        p.setPipeline(this.scanBlocksPipeline);
        p.setBindGroup(0, bg);
        p.dispatchWorkgroups(this.histBlocks, 1, 1);
        p.end();
      }

      {
        const bg = this.device.createBindGroup({
          layout: this.scanBlockSumsPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.blockSums } },
            { binding: 1, resource: { buffer: paramsBuf } },
          ],
        });
        const p = encoder.beginComputePass();
        p.setPipeline(this.scanBlockSumsPipeline);
        p.setBindGroup(0, bg);
        p.dispatchWorkgroups(1, 1, 1);
        p.end();
      }

      {
        const bg = this.device.createBindGroup({
          layout: this.applyOffsetsPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.histBuffer } },
            { binding: 1, resource: { buffer: this.blockSums } },
            { binding: 2, resource: { buffer: paramsBuf } },
          ],
        });
        const p = encoder.beginComputePass();
        p.setPipeline(this.applyOffsetsPipeline);
        p.setBindGroup(0, bg);
        p.dispatchWorkgroups(this.histBlocks, 1, 1);
        p.end();
      }

      {
        const bg = this.device.createBindGroup({
          layout: this.scatterPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: inputBuf } },
            { binding: 1, resource: { buffer: outputBuf } },
            { binding: 2, resource: { buffer: this.histBuffer } },
            { binding: 3, resource: { buffer: paramsBuf } },
          ],
        });
        const p = encoder.beginComputePass();
        p.setPipeline(this.scatterPipeline);
        p.setBindGroup(0, bg);
        p.dispatchWorkgroups(this.gridSize, 1, 1);
        p.end();
      }

      [inputBuf, outputBuf] = [outputBuf, inputBuf];
    }

    if (passes % 2 !== 0) {
      throw new Error("RadixSortUInt2 expects an even number of passes.");
    }
  }
}
