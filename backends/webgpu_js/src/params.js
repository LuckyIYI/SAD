// Direct port of backends/webgpu_py/sad_shared.py + the per-pipeline param
// buffer creators sprinkled through backends/webgpu_py/train_wgpu.py.
// Every struct layout here matches the WGSL declarations in sad_shared.wgsl.

export const SITE_FLOATS = 10;
export const ADAM_FLOATS = 24;
export const BITS_PER_SITE = 16.0 * 8.0;
export const PACKED_CAND_BYTES = 16;
export const TAU_DIFFUSE_PASSES = 4;
export const TAU_DIFFUSE_LAMBDA = 0.05;

function u32View(buffer, offsetBytes, nU32) {
  return new Uint32Array(buffer, offsetBytes, nU32);
}
function f32View(buffer, offsetBytes, nF32) {
  return new Float32Array(buffer, offsetBytes, nF32);
}

// ---------------------------------------------------------------------------
// Params (used by seed_cand, jfa_flood, vpt, candidate_pack (as ClearParams),
// render_compute, etc.)
// Layout (matches sad_shared.py create_params_buffer, struct "<IIIIfIfIIIIIIIII"):
//   u32 width, u32 height, u32 siteCount, u32 step,
//   f32 invScaleSq, u32 seed, f32 radiusScale, u32 radiusProbes,
//   u32 injectCount, u32 hilbertProbes, u32 hilbertWindow, u32 candDownscale,
//   u32 candWidth, u32 candHeight, u32 _pad0, u32 _pad1   (total 16*4 = 64 bytes)
// ---------------------------------------------------------------------------

export const PARAMS_SIZE = 64;

export function writeParams(
  view,
  { width, height, siteCount, step = 0, seed = 0,
    radiusScale, radiusProbes, injectCount,
    hilbertProbes, hilbertWindow, candDownscale,
    candWidth, candHeight },
) {
  const scale = Math.max(width, height);
  const invScaleSq = 1.0 / (scale * scale);
  const cd = Math.max(1, candDownscale | 0);
  const u = new Uint32Array(view.buffer, view.byteOffset, 16);
  const f = new Float32Array(view.buffer, view.byteOffset, 16);
  u[0] = width >>> 0;
  u[1] = height >>> 0;
  u[2] = siteCount >>> 0;
  u[3] = step >>> 0;
  f[4] = invScaleSq;
  u[5] = seed >>> 0;
  f[6] = radiusScale;
  u[7] = radiusProbes >>> 0;
  u[8] = injectCount >>> 0;
  u[9] = hilbertProbes >>> 0;
  u[10] = hilbertWindow >>> 0;
  u[11] = cd >>> 0;
  u[12] = candWidth >>> 0;
  u[13] = candHeight >>> 0;
  u[14] = 0;
  u[15] = 0;
}

export function createParamsBuffer(device, opts) {
  const buf = device.createBuffer({
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateParamsBuffer(device.queue, buf, opts);
  return buf;
}

export function updateParamsBuffer(queue, buffer, opts) {
  const bytes = new ArrayBuffer(PARAMS_SIZE);
  writeParams(new DataView(bytes), opts);
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// InitParams (init_gradient.wgsl):
//   u32 numSites, f32 gradThreshold, u32 maxAttempts, u32 _pad,
//   f32 initLogTau, f32 initRadius, u32 _pad1, u32 _pad2  (32 bytes)
// Python equivalent: struct "<IfIIffII" in train_wgpu.main()
// ---------------------------------------------------------------------------

export function createInitParamsBuffer(device, numSites, gradThreshold, maxAttempts,
                                       initLogTau, initRadius) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = numSites >>> 0;
  f[1] = gradThreshold;
  u[2] = maxAttempts >>> 0;
  u[3] = 0;
  f[4] = initLogTau;
  f[5] = initRadius;
  u[6] = 0;
  u[7] = 0;
  const buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, bytes);
  return buf;
}

// ---------------------------------------------------------------------------
// GradParams (gradients_tiled.wgsl, stats.wgsl):
//   u32 siteCount, u32 computeRemoval, u32 _pad0, u32 _pad1,
//   f32 invScaleSq, f32 _pad2, f32 _pad3, f32 _pad4     (32 bytes)
// ---------------------------------------------------------------------------

export function createGradParamsBuffer(device, siteCount, invScaleSq, computeRemoval) {
  const buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateGradParamsBuffer(device.queue, buf, siteCount, invScaleSq, computeRemoval);
  return buf;
}

export function updateGradParamsBuffer(queue, buffer, siteCount, invScaleSq, computeRemoval) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = siteCount >>> 0;
  u[1] = computeRemoval >>> 0;
  u[2] = 0;
  u[3] = 0;
  f[4] = invScaleSq;
  f[5] = 0;
  f[6] = 0;
  f[7] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// AdamParams (adam.wgsl):
//   f32 lrPos, lrTau, lrRadius, lrColor, lrDir, lrAniso, beta1, beta2, eps,
//   u32 t, u32 width, u32 height, u32 _pad   (13 slots, buffer padded to 64 bytes
//   to match the WGSL struct roundUp alignment).
// ---------------------------------------------------------------------------

export function createAdamParamsBuffer(device, lrPos, lrTau, lrRadius, lrColor,
                                       lrDir, lrAniso, beta1, beta2, eps,
                                       t, width, height) {
  const buf = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateAdamParamsBuffer(device.queue, buf, lrPos, lrTau, lrRadius, lrColor,
                          lrDir, lrAniso, beta1, beta2, eps, t, width, height);
  return buf;
}

export function updateAdamParamsBuffer(queue, buffer, lrPos, lrTau, lrRadius, lrColor,
                                        lrDir, lrAniso, beta1, beta2, eps,
                                        t, width, height) {
  const bytes = new ArrayBuffer(64);
  const f = new Float32Array(bytes);
  const u = new Uint32Array(bytes);
  f[0] = lrPos;
  f[1] = lrTau;
  f[2] = lrRadius;
  f[3] = lrColor;
  f[4] = lrDir;
  f[5] = lrAniso;
  f[6] = beta1;
  f[7] = beta2;
  f[8] = eps;
  u[9] = t >>> 0;
  u[10] = width >>> 0;
  u[11] = height >>> 0;
  u[12] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// SplitParams (split.wgsl):
//   u32 numToSplit, u32 currentSiteCount, u32 _pad, u32 _pad   (16 bytes)
// ---------------------------------------------------------------------------

export function createSplitParamsBuffer(device, numToSplit, currentSiteCount) {
  const buf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateSplitParamsBuffer(device.queue, buf, numToSplit, currentSiteCount);
  return buf;
}

export function updateSplitParamsBuffer(queue, buffer, numToSplit, currentSiteCount) {
  const bytes = new ArrayBuffer(16);
  const u = new Uint32Array(bytes);
  u[0] = numToSplit >>> 0;
  u[1] = currentSiteCount >>> 0;
  u[2] = 0;
  u[3] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// PruneParams / ClearParams / WriteIndicesParams: single u32 count + pad
// (used by prune.wgsl, clear_buffer_*.wgsl, candidate_pack.wgsl, write_indices.wgsl,
// tau_extract.wgsl, tau_writeback.wgsl).
// ---------------------------------------------------------------------------

export function createCountParamsBuffer(device, count) {
  const buf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateCountParamsBuffer(device.queue, buf, count);
  return buf;
}

export function updateCountParamsBuffer(queue, buffer, count) {
  const bytes = new ArrayBuffer(16);
  const u = new Uint32Array(bytes);
  u[0] = count >>> 0;
  u[1] = 0;
  u[2] = 0;
  u[3] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// Aliases to match Python naming
export const createClearParamsBuffer = createCountParamsBuffer;
export const createPruneParamsBuffer = createCountParamsBuffer;
export const updateClearParamsBuffer = updateCountParamsBuffer;
export const updatePruneParamsBuffer = updateCountParamsBuffer;

// ---------------------------------------------------------------------------
// TauDiffuseParams (tau_diffuse.wgsl):
//   u32 siteCount, u32 candDownscale, u32 _pad, u32 _pad,
//   f32 lambda, f32 _pad, f32 _pad, f32 _pad   (32 bytes)
// ---------------------------------------------------------------------------

export function createTauParamsBuffer(device, siteCount, lambda, candDownscale) {
  const buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateTauParamsBuffer(device.queue, buf, siteCount, lambda, candDownscale);
  return buf;
}

export function updateTauParamsBuffer(queue, buffer, siteCount, lambda, candDownscale) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = siteCount >>> 0;
  u[1] = (candDownscale | 0) >>> 0;
  u[2] = 0;
  u[3] = 0;
  f[4] = lambda;
  f[5] = 0;
  f[6] = 0;
  f[7] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// DensifyScoreParams (score_pairs_densify.wgsl):
//   u32 siteCount, u32 pairCount, u32 _pad, u32 _pad,
//   f32 minMass, f32 scoreAlpha, f32 _pad, f32 _pad  (32 bytes)
// ---------------------------------------------------------------------------

export function createDensifyScoreParamsBuffer(device, siteCount, pairCount, minMass, scoreAlpha) {
  const buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateDensifyScoreParamsBuffer(device.queue, buf, siteCount, pairCount, minMass, scoreAlpha);
  return buf;
}

export function updateDensifyScoreParamsBuffer(queue, buffer, siteCount, pairCount,
                                                minMass, scoreAlpha) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = siteCount >>> 0;
  u[1] = pairCount >>> 0;
  u[2] = 0;
  u[3] = 0;
  f[4] = minMass;
  f[5] = scoreAlpha;
  f[6] = 0;
  f[7] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// PruneScoreParams (score_pairs_prune.wgsl):
//   u32 siteCount, u32 pairCount, u32 _pad, u32 _pad,
//   f32 deltaNorm, f32 _pad, f32 _pad, f32 _pad   (32 bytes)
// ---------------------------------------------------------------------------

export function createPruneScoreParamsBuffer(device, siteCount, pairCount, deltaNorm) {
  const buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updatePruneScoreParamsBuffer(device.queue, buf, siteCount, pairCount, deltaNorm);
  return buf;
}

export function updatePruneScoreParamsBuffer(queue, buffer, siteCount, pairCount, deltaNorm) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = siteCount >>> 0;
  u[1] = pairCount >>> 0;
  u[2] = 0;
  u[3] = 0;
  f[4] = deltaNorm;
  f[5] = 0;
  f[6] = 0;
  f[7] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// HilbertParams (hilbert.wgsl):
//   u32 siteCount, u32 paddedCount, u32 width, u32 height, u32 bits,
//   u32 _pad, u32 _pad, u32 _pad   (32 bytes)
// ---------------------------------------------------------------------------

export function createHilbertParamsBuffer(device, siteCount, paddedCount, width, height, bits) {
  const buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  updateHilbertParamsBuffer(device.queue, buf, siteCount, paddedCount, width, height, bits);
  return buf;
}

export function updateHilbertParamsBuffer(queue, buffer, siteCount, paddedCount, width, height, bits) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  u[0] = siteCount >>> 0;
  u[1] = paddedCount >>> 0;
  u[2] = width >>> 0;
  u[3] = height >>> 0;
  u[4] = bits >>> 0;
  u[5] = 0;
  u[6] = 0;
  u[7] = 0;
  queue.writeBuffer(buffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// JFA step packing — direct port of sad_shared.py:pack_jump_step.
// ---------------------------------------------------------------------------

export function packJumpStep(stepIndex, width, height) {
  const maxDim = Math.max(width, height);
  let pow2 = 1;
  while (pow2 < maxDim) pow2 <<= 1;
  if (pow2 <= 1) return 1;
  const stages = 32 - Math.clz32(pow2) - 1;
  let stage;
  if (stages <= 0) stage = 0;
  else stage = stepIndex >= stages ? stages - 1 : stepIndex;
  let step = pow2 >> (stage + 1);
  step = Math.max(step, 1);
  step = Math.min(step, 0xffff);
  return ((step << 16) | (stepIndex & 0xffff)) >>> 0;
}

export function hilbertBitsForSize(width, height) {
  const maxDim = Math.max(width, height);
  let n = 1;
  let bits = 0;
  while (n < maxDim) {
    n <<= 1;
    bits += 1;
  }
  return Math.max(bits, 1);
}
