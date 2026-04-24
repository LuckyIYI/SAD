const fileInput = document.getElementById("fileInput");
const renderModeSelect = document.getElementById("renderModeSelect");
const dotsToggle = document.getElementById("dotsToggle");
const exportBtn = document.getElementById("exportBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("gpuCanvas");
const fallback = document.getElementById("fallback");

let device = null;
let context = null;
let format = null;
let pipelines = null;
let buffers = null;
let textures = null;
let initPromise = null;
let siteCount = 0;
let imageWidth = 0;
let imageHeight = 0;
let zoom = 1.0;
let pan = { x: 0, y: 0 };
let renderQueued = false;
let candidateWidth = 0;
let candidateHeight = 0;
let currentCand0 = null;
let currentCand1 = null;
let altCand0 = null;
let altCand1 = null;
let warmupInFlight = false;
const TOPK_WARMUP_PASSES = 32;

let siteData = [];
let siteBufferSize = 0;
let packedSiteBufferSize = 0;
let hilbertBufferSize = 0;

// Parameter ranges for visualization
let minTau = 0;
let maxTau = 1;
let minAniso = 0;
let maxAniso = 1;
let minRadius = 0;
let maxRadius = 1;

const invalidId = 0xffffffff;
const PACKED_CAND_BYTES = 16;

const vertexData = new Float32Array([
  -1, -1,
  1, -1,
  -1, 1,
  1, 1,
]);
let VPT_RADIUS_SCALE = 64;
let VPT_RADIUS_PROBES = 0;
let VPT_INJECT_COUNT = 32;
let VPT_HILBERT_PROBES = 0;
let VPT_HILBERT_WINDOW = 0;
let candDownscale = 1;

const SHADER_URL = "../shared/sad_shared.wgsl";
const CONFIG_URL = "../../training_config.json";
let shaderCode = "";
let sharedConfig = null;

async function loadConfig() {
  if (sharedConfig) {
    return sharedConfig;
  }
  const url = new URL(CONFIG_URL, window.location.href);
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load config: ${response.status} ${response.statusText}`);
    }
    sharedConfig = await response.json();
  } catch (err) {
    // The viewer is intentionally usable from file:// with the embedded WGSL
    // fallback. If config fetch is blocked there, keep the built-in defaults.
    sharedConfig = null;
  }
  return sharedConfig;
}

function applyConfig(config) {
  if (!config) return;
  if (Number.isFinite(config.CAND_RADIUS_SCALE)) {
    VPT_RADIUS_SCALE = config.CAND_RADIUS_SCALE;
  }
  if (Number.isFinite(config.CAND_RADIUS_PROBES)) {
    VPT_RADIUS_PROBES = config.CAND_RADIUS_PROBES;
  }
  if (Number.isFinite(config.CAND_INJECT_COUNT)) {
    VPT_INJECT_COUNT = config.CAND_INJECT_COUNT;
  }
  if (Number.isFinite(config.CAND_HILBERT_PROBES)) {
    VPT_HILBERT_PROBES = Math.max(0, Math.floor(config.CAND_HILBERT_PROBES));
  }
  if (Number.isFinite(config.CAND_HILBERT_WINDOW)) {
    VPT_HILBERT_WINDOW = Math.max(0, Math.floor(config.CAND_HILBERT_WINDOW));
  }
  if (Number.isFinite(config.CAND_DOWNSCALE)) {
    candDownscale = Math.max(1, Math.floor(config.CAND_DOWNSCALE));
  }
}

async function loadShaderCode() {
  if (shaderCode) {
    return;
  }
  const url = new URL(SHADER_URL, window.location.href);
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load WGSL: ${response.status} ${response.statusText}`);
    }
    shaderCode = await response.text();
  } catch (err) {
    if (window.SAD_SHADER_CODE) {
      shaderCode = window.SAD_SHADER_CODE;
    } else {
      throw err;
    }
  }
  if (!shaderCode.trim()) {
    throw new Error("WGSL file is empty.");
  }
}



function setStatus(message) {
  statusEl.textContent = message;
}

function updateActionButtons() {
  const hasSites = siteCount > 0;
  exportBtn.disabled = !hasSites;
}

updateActionButtons();

function parseSites(text) {
  const lines = text.split(/\r?\n/);
  const data = [];
  let width = 0;
  let height = 0;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      if (trimmed.startsWith("#")) {
        const match = trimmed.match(/image size\s*:?\s*(\d+)\s+(\d+)/i);
        if (match) {
          width = Math.max(1, parseInt(match[1], 10));
          height = Math.max(1, parseInt(match[2], 10));
        }
      }
      continue;
    }
    const parts = trimmed.split(/\s+/);
    if (parts.length !== 7 && parts.length !== 10) {
      continue;
    }
    const vals = parts.map(Number);
    if (vals.some((v) => Number.isNaN(v))) {
      continue;
    }
    if (vals.length === 7) {
      vals.push(1, 0, 0);
    }
    data.push(vals);
  }
  return { sites: data, width, height };
}

async function initWebGPU() {
  if (!navigator.gpu) {
    fallback.classList.remove("hidden");
    setStatus("WebGPU is not available in this browser.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    fallback.classList.remove("hidden");
    setStatus("No WebGPU adapter found.");
    return;
  }
  applyConfig(await loadConfig());
  await loadShaderCode();
  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });
  pipelines = createPipelines(device);
  buffers = createBuffers(device);
  resizeCanvas();
  window.addEventListener("resize", () => {
    resizeCanvas();
    requestRender();
  });
  setStatus("Ready. Choose a sites.txt file.");
}

async function ensureInitialized() {
  if (!initPromise) {
    initPromise = initWebGPU();
  }
  await initPromise;
  if (!device || !buffers) {
    throw new Error("WebGPU initialization failed.");
  }
}

function createPipelines(device) {
  const module = device.createShaderModule({ code: shaderCode });

  const clearPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "clearCandidates" },
  });

  const copyPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "copyCandidates" },
  });

  const seedPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "jfaSeed" },
  });

  const floodPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "jfaFlood" },
  });

  const packPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "packCandidateSites" },
  });

  const updatePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "updateCandidatesPacked" },
  });

  const initPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "initCandidates" },
  });

  const quadPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vsMain",
      buffers: [
        {
          arrayStride: 8,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
          ],
        },
      ],
    },
    fragment: {
      module,
      entryPoint: "fsMain",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-strip" },
  });

  return {
    clearPipeline,
    copyPipeline,
    seedPipeline,
    floodPipeline,
    packPipeline,
    updatePipeline,
    initPipeline,
    quadPipeline,
  };
}

function createBuffers(device) {
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(vertexBuffer.getMappedRange()).set(vertexData);
  vertexBuffer.unmap();

  const paramsBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const viewBuffer = device.createBuffer({
    size: 72,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const updateViewBuffer = device.createBuffer({
    size: 72,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const packParamsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const dummyHilbertBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dummyHilbertBuffer, 0, new Uint32Array([0]).buffer);
  return {
    vertexBuffer,
    paramsBuffer,
    viewBuffer,
    updateViewBuffer,
    packParamsBuffer,
    dummyHilbertBuffer,
    packedSitesBuffer: null,
    hilbertOrderBuffer: null,
    hilbertPosBuffer: null,
  };
}

function createCandidateTextures(width, height) {
  const desc = {
    size: [width, height],
    format: "rgba32uint",
    usage: GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.TEXTURE_BINDING
      | GPUTextureUsage.COPY_SRC
      | GPUTextureUsage.COPY_DST,
  };
  return {
    cand0A: device.createTexture(desc),
    cand1A: device.createTexture(desc),
    cand0B: device.createTexture(desc),
    cand1B: device.createTexture(desc),
  };
}

function jumpStepForIndex(stepIndex, width, height) {
  const maxDim = Math.max(width, height);
  let pow2 = 1;
  while (pow2 < maxDim) pow2 <<= 1;
  if (pow2 <= 1) return 1;
  let stages = 0;
  let tmp = pow2;
  while (tmp > 1) {
    tmp >>= 1;
    stages += 1;
  }
  const stage = stages > 0 ? Math.min(stepIndex, stages - 1) : 0;
  const step = pow2 >> (stage + 1);
  return Math.max(step, 1);
}

function packJumpStep(stepIndex, width, height) {
  const step = Math.min(jumpStepForIndex(stepIndex, width, height), 0xffff);
  return (((step << 16) >>> 0) | (stepIndex & 0xffff)) >>> 0;
}

function updateParamsBuffer(step = 0, seed = 12345) {
  const scale = Math.max(imageWidth, imageHeight);
  const invScaleSq = 1.0 / (scale * scale);
  const data = new ArrayBuffer(64);
  const u32 = new Uint32Array(data);
  const f32 = new Float32Array(data);
  u32[0] = imageWidth;
  u32[1] = imageHeight;
  u32[2] = siteCount;
  u32[3] = step;
  f32[4] = invScaleSq;
  u32[5] = seed;
  f32[6] = VPT_RADIUS_SCALE;
  u32[7] = VPT_RADIUS_PROBES;
  u32[8] = VPT_INJECT_COUNT;
  u32[9] = VPT_HILBERT_PROBES;
  u32[10] = VPT_HILBERT_WINDOW;
  u32[11] = candDownscale;
  u32[12] = candidateWidth || imageWidth;
  u32[13] = candidateHeight || imageHeight;
  u32[14] = 0;
  u32[15] = 0;
  device.queue.writeBuffer(buffers.paramsBuffer, 0, data);
}

function updateViewBuffer() {
  writeViewBufferData(buffers.viewBuffer, {
    imageWidth,
    imageHeight,
    canvasWidth: canvas.width,
    canvasHeight: canvas.height,
    panX: pan.x,
    panY: pan.y,
    zoom,
    renderMode: parseInt(renderModeSelect.value, 10),
    showDots: dotsToggle.checked,
  });
}

function writeViewBufferData(targetBuffer, {
  imageWidth: imgW,
  imageHeight: imgH,
  canvasWidth,
  canvasHeight,
  panX,
  panY,
  zoom: zoomValue,
  renderMode = 0,
  showDots = false,
}) {
  const view = new Float32Array(18);
  view[0] = imgW;
  view[1] = imgH;
  view[2] = canvasWidth;
  view[3] = canvasHeight;
  view[4] = panX;
  view[5] = panY;
  view[6] = zoomValue;
  const viewU32 = new Uint32Array(view.buffer);
  viewU32[7] = renderMode;
  viewU32[8] = showDots ? 1 : 0;
  viewU32[9] = 0; // _pad0
  view[10] = minTau;
  view[11] = maxTau;
  view[12] = minAniso;
  view[13] = maxAniso;
  view[14] = minRadius;
  view[15] = maxRadius;
  view[16] = 0; // _pad1
  device.queue.writeBuffer(targetBuffer, 0, view.buffer);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
  if (imageWidth > 0 && imageHeight > 0) {
    fitView();
  }
}

function ensureViewCandidates() {
  if (!device || imageWidth === 0 || imageHeight === 0) {
    return;
  }
  const targetCandWidth = Math.max(1, Math.ceil(imageWidth / candDownscale));
  const targetCandHeight = Math.max(1, Math.ceil(imageHeight / candDownscale));
  if (!textures || candidateWidth !== targetCandWidth || candidateHeight !== targetCandHeight) {
    textures = createCandidateTextures(targetCandWidth, targetCandHeight);
    candidateWidth = targetCandWidth;
    candidateHeight = targetCandHeight;
    currentCand0 = textures.cand0A;
    currentCand1 = textures.cand1A;
    altCand0 = textures.cand0B;
    altCand1 = textures.cand1B;
    textures.finalCand0 = null;
    textures.finalCand1 = null;
  }
}

function markViewDirty() {
  requestRender();
}

function syncCandidateState() {
  if (!textures || !textures.finalCand0 || !textures.finalCand1) {
    return;
  }
  currentCand0 = textures.finalCand0;
  currentCand1 = textures.finalCand1;
  altCand0 = currentCand0 === textures.cand0A ? textures.cand0B : textures.cand0A;
  altCand1 = currentCand1 === textures.cand1A ? textures.cand1B : textures.cand1A;
}

function fitView() {
  if (imageWidth === 0 || imageHeight === 0) {
    return;
  }
  const zoomX = canvas.width / imageWidth;
  const zoomY = canvas.height / imageHeight;
  zoom = Math.min(zoomX, zoomY);
  pan = { x: 0, y: 0 };
  updateViewBuffer();
  markViewDirty();
}

function createParamsBufferForSize(width, height, step, seed = 0,
                                   radiusScale = VPT_RADIUS_SCALE,
                                   radiusProbes = VPT_RADIUS_PROBES,
                                   injectCount = VPT_INJECT_COUNT,
                                   candWidth = width,
                                   candHeight = height,
                                   candScale = candDownscale) {
  const scale = Math.max(width, height);
  const invScaleSq = 1.0 / (scale * scale);
  const data = new ArrayBuffer(64);
  const u32 = new Uint32Array(data);
  const f32 = new Float32Array(data);
  u32[0] = width;
  u32[1] = height;
  u32[2] = siteCount;
  u32[3] = step;
  f32[4] = invScaleSq;
  u32[5] = seed;
  f32[6] = radiusScale;
  u32[7] = radiusProbes;
  u32[8] = injectCount;
  u32[9] = VPT_HILBERT_PROBES;
  u32[10] = VPT_HILBERT_WINDOW;
  u32[11] = candScale;
  u32[12] = candWidth;
  u32[13] = candHeight;
  u32[14] = 0;
  u32[15] = 0;
  const buf = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

function usesHilbertProbes() {
  return VPT_HILBERT_PROBES > 0 && VPT_HILBERT_WINDOW > 0;
}

function hilbertBitsForSize(width, height) {
  const maxDim = Math.max(width, height);
  let n = 1;
  let bits = 0;
  while (n < maxDim) {
    n <<= 1;
    bits += 1;
  }
  return Math.max(bits, 1);
}

function hilbertIndex(x, y, bits) {
  let xi = x >>> 0;
  let yi = y >>> 0;
  let index = 0;
  const mask = bits >= 32 ? 0xffffffff : ((1 << bits) >>> 0) - 1;
  for (let i = bits - 1; i >= 0; i -= 1) {
    const rx = (xi >>> i) & 1;
    const ry = (yi >>> i) & 1;
    const d = (3 * rx) ^ ry;
    index = (index | (d << (2 * i))) >>> 0;
    if (ry === 0) {
      if (rx === 1) {
        xi = (mask - xi) >>> 0;
        yi = (mask - yi) >>> 0;
      }
      const tmp = xi;
      xi = yi;
      yi = tmp;
    }
  }
  return index >>> 0;
}

function ensurePackedBuffers(count) {
  if (!device || !buffers || count <= 0) {
    return;
  }
  const packedSize = Math.max(1, count) * PACKED_CAND_BYTES;
  if (!buffers.packedSitesBuffer || packedSiteBufferSize !== packedSize) {
    if (buffers.packedSitesBuffer) {
      buffers.packedSitesBuffer.destroy();
    }
    buffers.packedSitesBuffer = device.createBuffer({
      size: packedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    packedSiteBufferSize = packedSize;
  }
  if (usesHilbertProbes()) {
    const hilbertSize = Math.max(1, count) * 4;
    if (!buffers.hilbertOrderBuffer || hilbertBufferSize !== hilbertSize) {
      if (buffers.hilbertOrderBuffer) {
        buffers.hilbertOrderBuffer.destroy();
      }
      if (buffers.hilbertPosBuffer) {
        buffers.hilbertPosBuffer.destroy();
      }
      buffers.hilbertOrderBuffer = device.createBuffer({
        size: hilbertSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      buffers.hilbertPosBuffer = device.createBuffer({
        size: hilbertSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      hilbertBufferSize = hilbertSize;
    }
  }
}

function updateHilbertBuffers(sites) {
  if (!usesHilbertProbes() || !device || !buffers) {
    return;
  }
  if (imageWidth <= 0 || imageHeight <= 0) {
    return;
  }
  const count = sites.length;
  if (count <= 0) {
    return;
  }
  ensurePackedBuffers(count);
  const bits = hilbertBitsForSize(imageWidth, imageHeight);
  const maxX = Math.max(0, imageWidth - 1);
  const maxY = Math.max(0, imageHeight - 1);
  const pairs = new Array(count);
  for (let i = 0; i < count; i += 1) {
    const site = sites[i];
    const px = Math.min(maxX, Math.max(0, Math.trunc(site[0])));
    const py = Math.min(maxY, Math.max(0, Math.trunc(site[1])));
    const key = hilbertIndex(px, py, bits);
    pairs[i] = { key, idx: i };
  }
  pairs.sort((a, b) => (a.key - b.key) || (a.idx - b.idx));
  const order = new Uint32Array(count);
  const pos = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) {
    const idx = pairs[i].idx;
    order[i] = idx;
    pos[idx] = i;
  }
  device.queue.writeBuffer(buffers.hilbertOrderBuffer, 0, order.buffer);
  device.queue.writeBuffer(buffers.hilbertPosBuffer, 0, pos.buffer);
}

function updatePackedSites(count = siteCount) {
  if (!device || !buffers || !pipelines || !buffers.siteBuffer || count <= 0) {
    return;
  }
  ensurePackedBuffers(count);
  const params = new Uint32Array(4);
  params[0] = count;
  device.queue.writeBuffer(buffers.packParamsBuffer, 0, params.buffer);
  const encoder = device.createCommandEncoder();
  const packPass = encoder.beginComputePass();
  packPass.setPipeline(pipelines.packPipeline);
  packPass.setBindGroup(0, device.createBindGroup({
    layout: pipelines.packPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.siteBuffer } },
      { binding: 1, resource: { buffer: buffers.packedSitesBuffer } },
      { binding: 2, resource: { buffer: buffers.packParamsBuffer } },
    ],
  }));
  packPass.dispatchWorkgroups(Math.ceil(count / 256));
  packPass.end();
  device.queue.submit([encoder.finish()]);
}

function getHilbertBuffers() {
  if (usesHilbertProbes() && buffers && buffers.hilbertOrderBuffer && buffers.hilbertPosBuffer) {
    return { order: buffers.hilbertOrderBuffer, pos: buffers.hilbertPosBuffer };
  }
  if (buffers && buffers.dummyHilbertBuffer) {
    return { order: buffers.dummyHilbertBuffer, pos: buffers.dummyHilbertBuffer };
  }
  return { order: null, pos: null };
}

function warmUpCandidates({
  targetTextures = textures,
  width = canvas.width,
  height = canvas.height,
  viewOverride = null,
} = {}) {
  if (!device || !targetTextures || !buffers || !buffers.siteBuffer || siteCount === 0) {
    return Promise.resolve();
  }

  const candW = Math.max(1, Math.ceil(width / candDownscale));
  const candH = Math.max(1, Math.ceil(height / candDownscale));

  if (!buffers.packedSitesBuffer) {
    updatePackedSites(siteCount);
  }
  if (usesHilbertProbes() && (!buffers.hilbertOrderBuffer || !buffers.hilbertPosBuffer)) {
    updateHilbertBuffers(siteData);
  }

  if (viewOverride) {
    writeViewBufferData(buffers.updateViewBuffer, viewOverride);
  }

  const allParamBuffers = [];
  const encoder = device.createCommandEncoder();

  // Match Metal's algorithm: random init + N pairs of (JFA + VPT)
  const numWarmupPasses = TOPK_WARMUP_PASSES;
  const initSeed = 12345;

  let cand0A = targetTextures.cand0A;
  let cand1A = targetTextures.cand1A;
  let cand0B = targetTextures.cand0B;
  let cand1B = targetTextures.cand1B;

  // Step 1: Initialize with random candidates (like Metal's initCandidates)
  const initParams = createParamsBufferForSize(width, height, 0, initSeed, VPT_RADIUS_SCALE,
                                               VPT_RADIUS_PROBES, VPT_INJECT_COUNT,
                                               candW, candH, candDownscale);
  allParamBuffers.push(initParams);
  const initPass = encoder.beginComputePass();
  initPass.setPipeline(pipelines.initPipeline);
  initPass.setBindGroup(0, device.createBindGroup({
    layout: pipelines.initPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: initParams } },
      { binding: 1, resource: cand0A.createView() },
      { binding: 2, resource: cand1A.createView() },
    ],
  }));
  initPass.dispatchWorkgroups(Math.ceil(candW / 8), Math.ceil(candH / 8));
  initPass.end();

  // Step 2: single JFA kickstart, then VPT passes
  const numJFAFloodPasses = Math.ceil(Math.log2(Math.max(candW, candH)));
  if (numJFAFloodPasses > 0) {
    const clearParams = createParamsBufferForSize(width, height, 0, 0,
                                                 VPT_RADIUS_SCALE, VPT_RADIUS_PROBES, VPT_INJECT_COUNT,
                                                 candW, candH, candDownscale);
    allParamBuffers.push(clearParams);
    const clearA = encoder.beginComputePass();
    clearA.setPipeline(pipelines.clearPipeline);
    clearA.setBindGroup(0, device.createBindGroup({
      layout: pipelines.clearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: clearParams } },
        { binding: 1, resource: cand0A.createView() },
      ],
    }));
    clearA.dispatchWorkgroups(Math.ceil(candW / 8), Math.ceil(candH / 8));
    clearA.end();

    const clearB = encoder.beginComputePass();
    clearB.setPipeline(pipelines.clearPipeline);
    clearB.setBindGroup(0, device.createBindGroup({
      layout: pipelines.clearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: clearParams } },
        { binding: 1, resource: cand1A.createView() },
      ],
    }));
    clearB.dispatchWorkgroups(Math.ceil(candW / 8), Math.ceil(candH / 8));
    clearB.end();

    const seedParams = createParamsBufferForSize(width, height, 0, 0,
                                                VPT_RADIUS_SCALE, VPT_RADIUS_PROBES, VPT_INJECT_COUNT,
                                                candW, candH, candDownscale);
    allParamBuffers.push(seedParams);
    const seedPass = encoder.beginComputePass();
    seedPass.setPipeline(pipelines.seedPipeline);
    seedPass.setBindGroup(0, device.createBindGroup({
      layout: pipelines.seedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.siteBuffer } },
        { binding: 1, resource: { buffer: seedParams } },
        { binding: 3, resource: cand0A.createView() },
        { binding: 4, resource: cand0B.createView() },
      ],
    }));
    seedPass.dispatchWorkgroups(Math.ceil(siteCount / 64));
    seedPass.end();

    [cand0A, cand0B] = [cand0B, cand0A];

    // Flood passes: ping-pong between cand0A and cand0B
    let floodSrc = cand0A;
    let floodDst = cand0B;
    for (let fp = 0; fp < numJFAFloodPasses; fp++) {
      const stepSize = 1 << (numJFAFloodPasses - 1 - fp);
      const floodParams = createParamsBufferForSize(width, height, stepSize, 0,
                                                   VPT_RADIUS_SCALE, VPT_RADIUS_PROBES, VPT_INJECT_COUNT,
                                                   candW, candH, candDownscale);
      allParamBuffers.push(floodParams);

      const floodPass = encoder.beginComputePass();
      floodPass.setPipeline(pipelines.floodPipeline);
      floodPass.setBindGroup(0, device.createBindGroup({
        layout: pipelines.floodPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.siteBuffer } },
          { binding: 1, resource: { buffer: floodParams } },
          { binding: 3, resource: floodSrc.createView() },
          { binding: 4, resource: floodDst.createView() },
        ],
      }));
      floodPass.dispatchWorkgroups(Math.ceil(candW / 8), Math.ceil(candH / 8));
      floodPass.end();

      [floodSrc, floodDst] = [floodDst, floodSrc];
    }

    // After flood, result is in floodSrc. Ensure it's in cand0A
    if (numJFAFloodPasses % 2 === 1) {
      [cand0A, cand0B] = [cand0B, cand0A];
    }
  }

  for (let warmupPass = 0; warmupPass < numWarmupPasses; warmupPass++) {
    const hilbertBuffers = getHilbertBuffers();
    const vptStep = packJumpStep(warmupPass, candW, candH);
    const vptSeed = (warmupPass >>> 16);
    const vptParams = createParamsBufferForSize(width, height, vptStep, vptSeed,
                                                VPT_RADIUS_SCALE, VPT_RADIUS_PROBES, VPT_INJECT_COUNT,
                                                candW, candH, candDownscale);
    allParamBuffers.push(vptParams);
    const vptPass = encoder.beginComputePass();
    vptPass.setPipeline(pipelines.updatePipeline);
    vptPass.setBindGroup(0, device.createBindGroup({
      layout: pipelines.updatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.packedSitesBuffer } },
        { binding: 1, resource: { buffer: vptParams } },
        { binding: 2, resource: cand0A.createView() },
        { binding: 3, resource: cand1A.createView() },
        { binding: 4, resource: cand0B.createView() },
        { binding: 5, resource: cand1B.createView() },
        { binding: 6, resource: { buffer: hilbertBuffers.order } },
        { binding: 7, resource: { buffer: hilbertBuffers.pos } },
      ],
    }));
    vptPass.dispatchWorkgroups(Math.ceil(candW / 16), Math.ceil(candH / 16));
    vptPass.end();

    // Swap: result now in cand0B/cand1B, make them the new A
    [cand0A, cand0B] = [cand0B, cand0A];
    [cand1A, cand1B] = [cand1B, cand1A];
  }

  // Final result is in cand0A/cand1A
  const srcCand0 = cand0A;
  const srcCand1 = cand1A;

  targetTextures.finalCand0 = srcCand0;
  targetTextures.finalCand1 = srcCand1;

  device.queue.submit([encoder.finish()]);

  return device.queue.onSubmittedWorkDone().then(() => {
    allParamBuffers.forEach(b => b.destroy());
  });
}

function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderFrame();
  });
}

function renderFrame() {
  if (!device || !textures || !buffers || !buffers.siteBuffer || siteCount === 0 || warmupInFlight) {
    return;
  }

  ensureViewCandidates();
  if (!currentCand0 || !currentCand1 || !altCand0 || !altCand1) {
    return;
  }
  if (!buffers.packedSitesBuffer) {
    updatePackedSites(siteCount);
  }
  if (usesHilbertProbes() && (!buffers.hilbertOrderBuffer || !buffers.hilbertPosBuffer)) {
    updateHilbertBuffers(siteData);
  }

  const encoder = device.createCommandEncoder();
  updateParamsBuffer(0, 0);

  updateViewBuffer();
  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
      clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
    }],
  });

  renderPass.setPipeline(pipelines.quadPipeline);
  renderPass.setVertexBuffer(0, buffers.vertexBuffer);
  renderPass.setBindGroup(2, device.createBindGroup({
    layout: pipelines.quadPipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: buffers.siteBuffer } },
      { binding: 1, resource: { buffer: buffers.paramsBuffer } },
      { binding: 2, resource: { buffer: buffers.viewBuffer } },
      { binding: 3, resource: currentCand0.createView() },
      { binding: 4, resource: currentCand1.createView() },
    ],
  }));
  renderPass.draw(4, 1, 0, 0);
  renderPass.end();

  device.queue.submit([encoder.finish()]);
}

function computeParameterRanges(sites) {
  minTau = Infinity;
  maxTau = -Infinity;
  minAniso = Infinity;
  maxAniso = -Infinity;
  minRadius = Infinity;
  maxRadius = -Infinity;

  for (const site of sites) {
    if (site[0] < 0) continue; // Skip deactivated sites
    const logTau = site[5];
    const radiusSq = site[6];
    const logAniso = site[9];

    minTau = Math.min(minTau, logTau);
    maxTau = Math.max(maxTau, logTau);
    minAniso = Math.min(minAniso, logAniso);
    maxAniso = Math.max(maxAniso, logAniso);
    minRadius = Math.min(minRadius, radiusSq);
    maxRadius = Math.max(maxRadius, radiusSq);
  }

  // Ensure non-zero range
  if (maxTau === minTau) maxTau = minTau + 1;
  if (maxAniso === minAniso) maxAniso = minAniso + 1;
  if (maxRadius === minRadius) maxRadius = minRadius + 1;
}

function uploadSites(sites) {
  siteCount = sites.length;
  siteData = sites;
  computeParameterRanges(sites);
  const floatsPerSite = 10;
  const data = new Float32Array(siteCount * floatsPerSite);
  for (let i = 0; i < siteCount; i += 1) {
    const s = sites[i];
    const offset = i * floatsPerSite;
    data[offset + 0] = s[0];  // position.x
    data[offset + 1] = s[1];  // position.y
    data[offset + 2] = s[5];  // log_tau
    data[offset + 3] = s[6];  // radius_sq
    data[offset + 4] = s[2];  // color.r
    data[offset + 5] = s[3];  // color.g
    data[offset + 6] = s[4];  // color.b
    data[offset + 7] = s[7];  // aniso_dir.x
    data[offset + 8] = s[8];  // aniso_dir.y
    data[offset + 9] = s[9];  // log_aniso
  }
  if (!buffers.siteBuffer || siteBufferSize !== data.byteLength) {
    if (buffers.siteBuffer) {
      buffers.siteBuffer.destroy();
    }
    buffers.siteBuffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    siteBufferSize = data.byteLength;
  }
  device.queue.writeBuffer(buffers.siteBuffer, 0, data.buffer);
  ensurePackedBuffers(siteCount);
  updateHilbertBuffers(sites);
  updatePackedSites(siteCount);
  updateActionButtons();
}

async function applySites(sites, { fit = false, warmup = false } = {}) {
  uploadSites(sites);
  ensureViewCandidates();
  if (fit) {
    fitView();
  }
  if (warmup) {
    warmupInFlight = true;
    try {
      const viewOverride = {
        imageWidth,
        imageHeight,
        canvasWidth: imageWidth,
        canvasHeight: imageHeight,
        panX: 0,
        panY: 0,
        zoom: 1.0,
        renderMode: 0,
        showDots: false,
      };
      await warmUpCandidates({
        targetTextures: textures,
        width: imageWidth,
        height: imageHeight,
        viewOverride,
      });
    } finally {
      warmupInFlight = false;
    }
    syncCandidateState();
  }
  updateViewBuffer();
  requestRender();
}

async function loadSitesFromInput(inputEl) {
  try {
    await ensureInitialized();
  } catch (err) {
    setStatus(`Load failed: ${err.message || err}`);
    return;
  }
  const file = inputEl.files[0];
  if (!file) {
    setStatus("Choose a .txt file first.");
    return;
  }
  const fileLabel = inputEl.closest(".file")?.querySelector("span");
  if (fileLabel) {
    fileLabel.textContent = file.name;
  }
  const text = await file.text();
  const parsed = parseSites(text);
  const sites = parsed.sites;
  if (sites.length === 0) {
    setStatus("No valid sites found.");
    return;
  }
  if (parsed.width <= 0 || parsed.height <= 0) {
    setStatus("Missing image size in TXT header (expected: # Image size: W H).");
    return;
  }
  imageWidth = parsed.width;
  imageHeight = parsed.height;
  await applySites(sites, { fit: true, warmup: true });
  setStatus(`Loaded ${sites.length} sites. Render size ${imageWidth}x${imageHeight}.`);
}

fileInput.addEventListener("change", () => {
  loadSitesFromInput(fileInput).catch((err) => {
    setStatus(`Load failed: ${err.message || err}`);
  });
});
renderModeSelect.addEventListener("change", requestRender);
dotsToggle.addEventListener("change", requestRender);
exportBtn.addEventListener("click", exportPNG);

async function exportPNG() {
  try {
    await ensureInitialized();
  } catch (err) {
    setStatus(`Load failed: ${err.message || err}`);
    return;
  }
  if (!device || !textures || !buffers || !buffers.siteBuffer || siteCount === 0) {
    setStatus("Nothing to export.");
    return;
  }

  setStatus("Exporting...");

  const exportCandW = Math.max(1, Math.ceil(imageWidth / candDownscale));
  const exportCandH = Math.max(1, Math.ceil(imageHeight / candDownscale));
  const exportTextures = createCandidateTextures(exportCandW, exportCandH);
  const exportView = {
    imageWidth,
    imageHeight,
    canvasWidth: imageWidth,
    canvasHeight: imageHeight,
    panX: 0,
    panY: 0,
    zoom: 1.0,
    renderMode: 0,
    showDots: false,
  };
  await warmUpCandidates({
    targetTextures: exportTextures,
    width: imageWidth,
    height: imageHeight,
    viewOverride: exportView,
  });

  // Create render texture at exact image resolution
  const exportTexture = device.createTexture({
    size: [imageWidth, imageHeight],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // Create buffer to read back pixels
  const bytesPerRow = Math.ceil(imageWidth * 4 / 256) * 256;
  const readBuffer = device.createBuffer({
    size: bytesPerRow * imageHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Update params for export (1:1 zoom, no pan)
  const scale = Math.max(imageWidth, imageHeight);
  const invScaleSq = 1.0 / (scale * scale);
  const paramsData = new ArrayBuffer(64);
  const paramsU32 = new Uint32Array(paramsData);
  const paramsF32 = new Float32Array(paramsData);
  paramsU32[0] = imageWidth;
  paramsU32[1] = imageHeight;
  paramsU32[2] = siteCount;
  paramsU32[3] = 0;
  paramsF32[4] = invScaleSq;
  paramsF32[6] = VPT_RADIUS_SCALE;
  paramsU32[7] = VPT_RADIUS_PROBES;
  paramsU32[8] = VPT_INJECT_COUNT;
  paramsU32[9] = VPT_HILBERT_PROBES;
  paramsU32[10] = VPT_HILBERT_WINDOW;
  paramsU32[11] = candDownscale;
  paramsU32[12] = candidateWidth || imageWidth;
  paramsU32[13] = candidateHeight || imageHeight;
  paramsU32[14] = 0;
  paramsU32[15] = 0;
  device.queue.writeBuffer(buffers.paramsBuffer, 0, paramsData);

  // Update view params for 1:1 export (no zoom, no pan, canvas = image size)
  writeViewBufferData(buffers.viewBuffer, exportView);

  const encoder = device.createCommandEncoder();

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: exportTexture.createView(),
      loadOp: "clear",
      storeOp: "store",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  });

  renderPass.setPipeline(pipelines.quadPipeline);
  renderPass.setVertexBuffer(0, buffers.vertexBuffer);
  renderPass.setBindGroup(2, device.createBindGroup({
    layout: pipelines.quadPipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: buffers.siteBuffer } },
      { binding: 1, resource: { buffer: buffers.paramsBuffer } },
      { binding: 2, resource: { buffer: buffers.viewBuffer } },
      { binding: 3, resource: exportTextures.finalCand0.createView() },
      { binding: 4, resource: exportTextures.finalCand1.createView() },
    ],
  }));
  renderPass.draw(4, 1, 0, 0);
  renderPass.end();

  // Copy render texture to read buffer
  encoder.copyTextureToBuffer(
    { texture: exportTexture },
    { buffer: readBuffer, bytesPerRow },
    [imageWidth, imageHeight]
  );

  device.queue.submit([encoder.finish()]);

  // Read back and create image
  await readBuffer.mapAsync(GPUMapMode.READ);
  const data = new Uint8Array(readBuffer.getMappedRange());

  // Create canvas for PNG export
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = imageWidth;
  exportCanvas.height = imageHeight;
  const ctx = exportCanvas.getContext("2d");
  const imgData = ctx.createImageData(imageWidth, imageHeight);

  // Copy data (handle bytesPerRow padding)
  for (let y = 0; y < imageHeight; y++) {
    for (let x = 0; x < imageWidth; x++) {
      const srcIdx = y * bytesPerRow + x * 4;
      const dstIdx = (y * imageWidth + x) * 4;
      imgData.data[dstIdx + 0] = data[srcIdx + 0];
      imgData.data[dstIdx + 1] = data[srcIdx + 1];
      imgData.data[dstIdx + 2] = data[srcIdx + 2];
      imgData.data[dstIdx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  readBuffer.unmap();

  // Download
  const link = document.createElement("a");
  link.download = `webgpu_render_${imageWidth}x${imageHeight}.png`;
  link.href = exportCanvas.toDataURL("image/png");
  link.click();

  // Cleanup
  exportTexture.destroy();
  readBuffer.destroy();
  exportTextures.cand0A.destroy();
  exportTextures.cand0B.destroy();
  exportTextures.cand1A.destroy();
  exportTextures.cand1B.destroy();

  // Restore view params
  updateViewBuffer();

  setStatus(`Exported ${imageWidth}x${imageHeight} PNG.`);
}

// Expose for testing
window.viewer = {
  get device() { return device; },
  get buffers() { return buffers; },
  get textures() { return textures; },
  get pipelines() { return pipelines; },
  get siteCount() { return siteCount; },
  get imageWidth() { return imageWidth; },
  get imageHeight() { return imageHeight; },
  setDimensions(w, h) { imageWidth = w; imageHeight = h; requestRender(); },
  setSiteCount(n) { siteCount = n; },
  createCandidateTextures,
  warmUpCandidates,
  uploadSites,
  setTextures(t) { textures = t; },
};

initPromise = initWebGPU().catch((err) => {
  console.error(err);
  setStatus(`Load failed: ${err.message || err}`);
});
