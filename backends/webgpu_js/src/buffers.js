// Site / Adam / gradient / stats / pairs / indices buffer helpers.
// Mirrors the inline `device.create_buffer` calls in train_wgpu.py main().

import { SITE_FLOATS, ADAM_FLOATS, PACKED_CAND_BYTES } from "./params.js";

export function createSitesBuffer(device, capacity) {
  return device.createBuffer({
    size: capacity * SITE_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

export function createAdamBuffer(device, capacity) {
  const buf = device.createBuffer({
    size: capacity * ADAM_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const zeros = new Float32Array(capacity * ADAM_FLOATS);
  device.queue.writeBuffer(buf, 0, zeros);
  return buf;
}

export function createPackedCandidatesBuffer(device, capacity) {
  return device.createBuffer({
    size: capacity * PACKED_CAND_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

export function createSeedCounter(device) {
  const buf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, new Uint32Array([0]));
  return buf;
}

// Consolidated gradient buffer: `grads[idx * 10 + channel]`, atomic<i32>.
// This replaces the 10 separate per-channel buffers the Python/Metal backends
// use, so we stay within WebGPU's 10-storage-buffer-per-stage limit.
export function createGradsBuffer(device, capacity) {
  return device.createBuffer({
    size: capacity * 10 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

export function createAtomicU32Buffer(device, count) {
  return device.createBuffer({
    size: Math.max(1, count) * 4,
    usage: GPUBufferUsage.STORAGE,
  });
}

export function createStatBuffers(device, capacity) {
  return Array.from({ length: 8 }, () => createAtomicU32Buffer(device, capacity));
}

export function createPairsBuffer(device, pairCount) {
  return device.createBuffer({
    size: Math.max(8, pairCount * 8),
    usage: GPUBufferUsage.STORAGE,
  });
}

export function createIndicesBuffer(device, maxCount) {
  return device.createBuffer({
    size: Math.max(4, maxCount * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

export function createHilbertBuffers(device, capacity) {
  const order = device.createBuffer({
    size: Math.max(4, capacity * 4),
    usage: GPUBufferUsage.STORAGE,
  });
  const pos = device.createBuffer({
    size: Math.max(4, capacity * 4),
    usage: GPUBufferUsage.STORAGE,
  });
  const paddedCount = Math.floor((capacity + 1023) / 1024) * 1024;
  const pairs = device.createBuffer({
    size: Math.max(8, paddedCount * 8),
    usage: GPUBufferUsage.STORAGE,
  });
  return { order, pos, pairs, paddedCount };
}

export function createDummyStorageBuffer(device) {
  return device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });
}

// Read `count` sites from `sitesBuffer` as a Float32Array of shape (count, SITE_FLOATS).
// Mirrors read_sites() in train_wgpu.py.
export async function readSites(device, sitesBuffer, count) {
  const size = count * SITE_FLOATS * 4;
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(sitesBuffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange()).slice();
  staging.unmap();
  staging.destroy();
  return data;
}
