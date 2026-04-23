// Direct port of backends/webgpu_py/train_wgpu.py main() training loop.
// Every numbered stage and flag mirrors the Python source — keep this lined up
// with that file so cross-backend behaviour stays in sync.

import {
  SITE_FLOATS,
  ADAM_FLOATS,
  BITS_PER_SITE,
  PACKED_CAND_BYTES,
  TAU_DIFFUSE_PASSES,
  TAU_DIFFUSE_LAMBDA,
  PARAMS_SIZE,
  createParamsBuffer,
  updateParamsBuffer,
  createInitParamsBuffer,
  createGradParamsBuffer,
  updateGradParamsBuffer,
  createAdamParamsBuffer,
  updateAdamParamsBuffer,
  createSplitParamsBuffer,
  updateSplitParamsBuffer,
  createPruneParamsBuffer,
  updatePruneParamsBuffer,
  createCountParamsBuffer,
  updateCountParamsBuffer,
  createClearParamsBuffer,
  updateClearParamsBuffer,
  createTauParamsBuffer,
  updateTauParamsBuffer,
  createDensifyScoreParamsBuffer,
  updateDensifyScoreParamsBuffer,
  createPruneScoreParamsBuffer,
  updatePruneScoreParamsBuffer,
  createHilbertParamsBuffer,
  updateHilbertParamsBuffer,
  packJumpStep,
  hilbertBitsForSize,
} from "./params.js";

import {
  createSitesBuffer,
  createAdamBuffer,
  createPackedCandidatesBuffer,
  createSeedCounter,
  createGradsBuffer,
  createAtomicU32Buffer,
  createStatBuffers,
  createPairsBuffer,
  createIndicesBuffer,
  createHilbertBuffers,
  createDummyStorageBuffer,
  readSites,
} from "./buffers.js";

import {
  createRgba32FloatTexture,
  createCandidateTexture,
  writeFloat32ToRgba32FloatTexture,
  readRgba32FloatTexture,
  whiteMaskRgba32Float,
} from "./textures.js";

import {
  InitGradientEncoder,
  CandidatesEncoder,
  SeedCandidatesEncoder,
  JFAClearEncoder,
  JFAFloodEncoder,
  CandidatePackEncoder,
  RenderEncoder,
  RenderHashedEncoder,
  RenderTauHeatmapEncoder,
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
} from "./encoders.js";

import { RadixSortUInt2 } from "./radix_sort.js";

// Small "simulate final site count" helper — matches simulate_final_sites().
function simulateFinalSites(initSites, maxSites, iters, densifyEnabled, densifyStart,
                            densifyEnd, densifyFreq, densifyPercentile, pruneDuringDensify,
                            pruneStart, pruneEnd, pruneFreq, prunePercentile, maxSplitIndices) {
  let actualSites = initSites;
  let activeEstimate = initSites;
  maxSites = Math.max(maxSites, initSites);

  let effectivePruneStart = pruneStart;
  if (densifyEnabled && !pruneDuringDensify && pruneStart < densifyEnd) {
    effectivePruneStart = densifyEnd;
  }

  for (let it = 0; it < iters; it += 1) {
    if (densifyEnabled && densifyPercentile > 0 && it >= densifyStart &&
        it <= densifyEnd && (it % densifyFreq === 0) && actualSites < maxSites) {
      const desired = Math.floor(activeEstimate * densifyPercentile);
      const available = maxSites - actualSites;
      const numToSplit = Math.min(desired, available, maxSplitIndices);
      if (numToSplit > 0) {
        actualSites += numToSplit;
        activeEstimate += numToSplit;
      }
    }
    if (prunePercentile > 0 && it >= effectivePruneStart && it < pruneEnd && (it % pruneFreq === 0)) {
      const desired = Math.floor(activeEstimate * prunePercentile);
      const numToPrune = Math.min(desired, maxSplitIndices);
      if (numToPrune > 0) {
        activeEstimate = Math.max(0, activeEstimate - numToPrune);
      }
    }
  }
  return activeEstimate;
}

// Mirrors solve_target_bpp().
export function solveTargetBpp(args, width, height, initSiteCount, maxSites, iters) {
  const targetSites = Math.max(1, Math.round(args.targetBpp * width * height / BITS_PER_SITE));
  const baseDensify = args.densifyPercentile;
  const basePrune = args.prunePercentile;
  const maxBase = Math.max(baseDensify, basePrune);
  const maxSplitIndices = 65536;

  if (maxBase <= 0) {
    const final = simulateFinalSites(initSiteCount, maxSites, iters, args.densify,
      args.densifyStart, args.densifyEnd, Math.max(1, args.densifyFreq), 0,
      args.pruneDuringDensify, args.pruneStart, args.pruneEnd, Math.max(1, args.pruneFreq),
      0, maxSplitIndices);
    return { densifyPercentile: 0, prunePercentile: 0, finalSites: final,
             achievedBpp: final * BITS_PER_SITE / (width * height) };
  }

  const maxPct = 0.95;
  let sMax = maxPct / maxBase;
  if (sMax > 50) sMax = 50;

  const evalSites = (scale) => {
    const d = args.densify ? Math.min(maxPct, baseDensify * scale) : 0;
    const p = Math.min(maxPct, basePrune * scale);
    return simulateFinalSites(initSiteCount, maxSites, iters, args.densify,
      args.densifyStart, args.densifyEnd, Math.max(1, args.densifyFreq), d,
      args.pruneDuringDensify, args.pruneStart, args.pruneEnd, Math.max(1, args.pruneFreq),
      p, maxSplitIndices);
  };

  let bestScale = 0;
  let bestSites = evalSites(0);
  let bestErr = Math.abs(bestSites - targetSites);
  const samples = 80;
  for (let i = 0; i <= samples; i += 1) {
    const s = sMax * i / samples;
    const sites = evalSites(s);
    const err = Math.abs(sites - targetSites);
    if (err < bestErr) { bestErr = err; bestScale = s; bestSites = sites; }
  }

  let step = sMax / samples;
  for (let k = 0; k < 20; k += 1) {
    let improved = false;
    const s0 = bestScale - step;
    const s1 = bestScale + step;
    if (s0 >= 0) {
      const sites = evalSites(s0);
      const err = Math.abs(sites - targetSites);
      if (err < bestErr) { bestErr = err; bestScale = s0; bestSites = sites; improved = true; }
    }
    if (s1 <= sMax) {
      const sites = evalSites(s1);
      const err = Math.abs(sites - targetSites);
      if (err < bestErr) { bestErr = err; bestScale = s1; bestSites = sites; improved = true; }
    }
    if (!improved) step *= 0.5;
  }

  const densify = args.densify ? Math.min(maxPct, baseDensify * bestScale) : 0;
  const prune = Math.min(maxPct, basePrune * bestScale);
  return {
    densifyPercentile: densify,
    prunePercentile: prune,
    finalSites: bestSites,
    achievedBpp: bestSites * BITS_PER_SITE / (width * height),
  };
}

// Mirrors plan_site_capacity().
function planSiteCapacity(args, initialSiteCount, numPixels) {
  const needsPairs = Boolean(args.densify) || args.prunePercentile > 0;
  const needsPrune = args.prunePercentile > 0;
  let maxSitesCapacity;
  if (args.maxSites > 0) {
    maxSitesCapacity = args.maxSites;
  } else if (args.densify) {
    maxSitesCapacity = Math.min(numPixels * 2, Math.max(initialSiteCount * 8, 8192));
  } else {
    maxSitesCapacity = numPixels * 2;
  }
  const requestedCapacity = args.densify ? maxSitesCapacity : initialSiteCount;
  const bufferCapacity = Math.max(initialSiteCount, requestedCapacity);
  const scorePairsCount = needsPairs ? Math.max(1, bufferCapacity) : 0;
  return {
    densifyEnabled: Boolean(args.densify),
    needsPairs,
    needsPrune,
    maxSitesCapacity,
    requestedCapacity,
    bufferCapacity,
    scorePairsCount,
    maxSplitIndices: 65536,
  };
}

// Collect default args from a training_config.json-style object + user
// overrides. Overrides use the argument name (e.g. "iters") and take precedence
// over the matching config-style key (e.g. "DEFAULT_ITERS").
export function buildArgs(config, overrides = {}) {
  // `argName` is what the rest of the code uses; `configKey` is the
  // training_config.json-style key; `fallback` applies if neither source has a
  // value. Overrides always check `argName` first.
  const pick = (argName, configKey, fallback) => {
    if (overrides[argName] !== undefined) return overrides[argName];
    if (configKey && config[configKey] !== undefined) return config[configKey];
    return fallback;
  };
  const args = {
    iters: Number(pick("iters", "DEFAULT_ITERS", 2000)),
    sites: Number(pick("sites", "DEFAULT_SITES", 65536)),
    maxSites: Number(pick("maxSites", "DEFAULT_MAX_SITES", 70000)),
    densify: Boolean(pick("densify", null, true)),
    densifyStart: Number(pick("densifyStart", "DENSIFY_START", 20)),
    densifyEnd: Number(pick("densifyEnd", "DENSIFY_END", 3500)),
    densifyFreq: Number(pick("densifyFreq", "DENSIFY_FREQ", 20)),
    densifyPercentile: Number(pick("densifyPercentile", "DENSIFY_PERCENTILE", 0.01)),
    densifyScoreAlpha: Number(pick("densifyScoreAlpha", "DENSIFY_SCORE_ALPHA", 0.7)),
    pruneDuringDensify: Boolean(pick("pruneDuringDensify", "PRUNE_DURING_DENSIFY", true)),
    pruneStart: Number(pick("pruneStart", "PRUNE_START", 100)),
    pruneEnd: Number(pick("pruneEnd", "PRUNE_END", 3600)),
    pruneFreq: Number(pick("pruneFreq", "PRUNE_FREQ", 20)),
    prunePercentile: Number(pick("prunePercentile", "PRUNE_PERCENTILE", 0.01)),
    targetBpp: Number(pick("targetBpp", "DEFAULT_TARGET_BPP", -1.0)),
    candFreq: Number(pick("candFreq", "CAND_UPDATE_FREQ", 1)),
    candHilbertWindow: Number(pick("candHilbertWindow", "CAND_HILBERT_WINDOW", 0)),
    candHilbertProbes: Number(pick("candHilbertProbes", "CAND_HILBERT_PROBES", 0)),
    candDownscale: Math.max(1, Number(pick("candDownscale", "CAND_DOWNSCALE", 1))),
    candUpdatePasses: Math.max(1, Number(pick("candUpdatePasses", "CAND_UPDATE_PASSES", 1))),
    candRadiusScale: Number(pick("candRadiusScale", "CAND_RADIUS_SCALE", 64.0)),
    candRadiusProbes: Number(pick("candRadiusProbes", "CAND_RADIUS_PROBES", 0)),
    candInjectCount: Number(pick("candInjectCount", "CAND_INJECT_COUNT", 16)),
    logFreq: Math.max(1, Number(pick("logFreq", null, 1000))),
    viewerFreq: Math.max(1, Number(pick("viewerFreq", null, 10))),
    lrPosBase: Number(pick("lrPosBase", "LR_POS_BASE", 0.05)),
    lrTauBase: Number(pick("lrTauBase", "LR_TAU_BASE", 0.01)),
    lrRadiusBase: Number(pick("lrRadiusBase", "LR_RADIUS_BASE", 0.02)),
    lrColorBase: Number(pick("lrColorBase", "LR_COLOR_BASE", 0.02)),
    lrDirBase: Number(pick("lrDirBase", "LR_DIR_BASE", 0.02)),
    lrAnisoBase: Number(pick("lrAnisoBase", "LR_ANISO_BASE", 0.02)),
    initLogTau: Number(config.INIT_LOG_TAU),
    initRadius: Number(config.INIT_RADIUS),
    beta1: Number(pick("beta1", "BETA1", 0.9)),
    beta2: Number(pick("beta2", "BETA2", 0.999)),
    eps: Number(pick("eps", "EPS", 1e-8)),
    maxDim: Number(pick("maxDim", "MAX_DIM", 2048)),
  };
  if (!Number.isFinite(args.initLogTau) || !Number.isFinite(args.initRadius)) {
    throw new Error("training_config.json missing INIT_LOG_TAU / INIT_RADIUS (required).");
  }
  return args;
}

// Cooperative-yield helper — allows the browser to render and the caller to
// cancel long runs.
async function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export class Trainer {
  constructor(device, shaders, args, { onLog = null, signal = null } = {}) {
    this.device = device;
    this.shaders = shaders;
    this.args = args;
    this.onLog = onLog;
    this.signal = signal;

    this.usesHilbert = args.candHilbertWindow > 0 && args.candHilbertProbes > 0;

    // Encoders
    this.initEnc = new InitGradientEncoder(device, shaders);
    this.candsEnc = new CandidatesEncoder(device, shaders);
    this.seedEnc = new SeedCandidatesEncoder(device, shaders);
    this.jfaClearEnc = new JFAClearEncoder(device, shaders);
    this.jfaFloodEnc = new JFAFloodEncoder(device, shaders);
    this.packEnc = new CandidatePackEncoder(device, shaders);
    this.renderEnc = new RenderEncoder(device, shaders);
    this.renderHashedEnc = new RenderHashedEncoder(device, shaders);
    this.renderTauEnc = new RenderTauHeatmapEncoder(device, shaders);
    this.gradsEnc = new GradientsEncoder(device, shaders);
    this.adamEnc = new AdamEncoder(device, shaders);
    this.tauEnc = new TauEncoder(device, shaders);
    this.statsEnc = new StatsEncoder(device, shaders);
    this.splitEnc = new SplitEncoder(device, shaders);
    this.pruneEnc = new PruneEncoder(device, shaders);
    this.clearEnc = new ClearBufferEncoder(device, shaders, { useI32: false });
    this.clearI32Enc = new ClearBufferEncoder(device, shaders, { useI32: true });
    this.scoreEnc = new ScorePairsEncoder(device, shaders);
    this.writeIdxEnc = new WriteIndicesEncoder(device, shaders);
    this.hilbertEnc = this.usesHilbert ? new HilbertEncoder(device, shaders) : null;
  }

  async prepare({ width, height, targetRgba32f, maskRgba32f, maskSum, initSites, initSiteCount }) {
    const args = this.args;
    const device = this.device;
    const shaders = this.shaders;

    this.width = width;
    this.height = height;
    this.candDownscale = Math.max(1, args.candDownscale);
    this.candWidth = Math.max(1, Math.floor((width + this.candDownscale - 1) / this.candDownscale));
    this.candHeight = Math.max(1, Math.floor((height + this.candDownscale - 1) / this.candDownscale));
    this.target = targetRgba32f; // keep a CPU copy for PSNR
    this.maskSum = maskSum;
    this.maskCpu = null;
    if (maskRgba32f) {
      this.maskCpu = new Float32Array(width * height);
      for (let i = 0; i < width * height; i += 1) this.maskCpu[i] = maskRgba32f[i * 4];
    }

    // Textures
    this.targetTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: true });
    writeFloat32ToRgba32FloatTexture(device, this.targetTex, targetRgba32f, width, height);
    this.maskTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: true });
    const maskData = maskRgba32f || whiteMaskRgba32Float(width, height).rgba;
    writeFloat32ToRgba32FloatTexture(device, this.maskTex, maskData, width, height);

    // Optional target-BPP solver — runs pure-JS before allocating anything else.
    if (args.targetBpp > 0) {
      let maxSitesCapacity;
      if (args.maxSites > 0) maxSitesCapacity = args.maxSites;
      else if (args.densify) maxSitesCapacity = Math.min(width * height * 2, Math.max(initSiteCount * 8, 8192));
      else maxSitesCapacity = width * height * 2;
      const maxSites = Math.max(maxSitesCapacity, initSiteCount);
      const solved = solveTargetBpp(args, width, height, initSiteCount, maxSites, args.iters);
      args.densifyPercentile = solved.densifyPercentile;
      args.prunePercentile = solved.prunePercentile;
      this._log(
        `Target BPP: ${args.targetBpp.toFixed(3)} -> densify_percentile=${solved.densifyPercentile.toFixed(4)}, ` +
        `prune_percentile=${solved.prunePercentile.toFixed(4)}, predicted final sites ${solved.finalSites} ` +
        `(${solved.achievedBpp.toFixed(3)} bpp)`,
      );
    }

    // Plan capacity
    this.plan = planSiteCapacity(args, initSiteCount, width * height);
    this.bufferCapacity = this.plan.bufferCapacity;

    // Buffers
    this.sitesBuf = createSitesBuffer(device, this.bufferCapacity);
    this.packedSitesBuf = createPackedCandidatesBuffer(device, this.bufferCapacity);
    this.adamBuf = createAdamBuffer(device, this.bufferCapacity);
    this.seedCounter = createSeedCounter(device);
    this.grads = createGradsBuffer(device, this.bufferCapacity);
    this.removalDelta = createAtomicU32Buffer(device, this.bufferCapacity);
    this.tauGradRaw = createAtomicU32Buffer(device, this.bufferCapacity);
    this.tauGradTmp = createAtomicU32Buffer(device, this.bufferCapacity);
    this.tauGradTmp2 = createAtomicU32Buffer(device, this.bufferCapacity);
    this.statBufs = args.densify ? createStatBuffers(device, this.bufferCapacity) : null;
    this.pairsBuf = this.plan.needsPairs ? createPairsBuffer(device, this.plan.scorePairsCount) : null;
    this.splitIndicesBuf = args.densify ? createIndicesBuffer(device, this.plan.maxSplitIndices) : null;
    this.pruneIndicesBuf = this.plan.needsPrune ? createIndicesBuffer(device, this.plan.maxSplitIndices) : null;

    // Candidate textures
    this.cand0A = createCandidateTexture(device, this.candWidth, this.candHeight);
    this.cand0B = createCandidateTexture(device, this.candWidth, this.candHeight);
    this.cand1A = createCandidateTexture(device, this.candWidth, this.candHeight);
    this.cand1B = createCandidateTexture(device, this.candWidth, this.candHeight);

    // Render texture
    this.renderTex = createRgba32FloatTexture(device, width, height, { storage: true, copyDst: false, copySrc: true });

    // Param buffers
    this.candUpdatePasses = Math.max(1, args.candUpdatePasses);
    this.candParams = [];
    for (let i = 0; i < this.candUpdatePasses; i += 1) {
      this.candParams.push(createParamsBuffer(device, this._paramsArgs(initSiteCount, 0, 0)));
    }
    this.renderParams = createParamsBuffer(device, this._paramsArgs(initSiteCount, 0, 0));

    const invScaleSq = 1.0 / Math.pow(Math.max(width, height), 2);
    this.invScaleSq = invScaleSq;
    this.gradParamsStats = createGradParamsBuffer(device, initSiteCount, invScaleSq, 0);
    this.gradParamsGrad = createGradParamsBuffer(device, initSiteCount, invScaleSq, 0);
    this.adamParams = createAdamParamsBuffer(device,
      args.lrPosBase, args.lrTauBase, args.lrRadiusBase, args.lrColorBase,
      args.lrDirBase, args.lrAnisoBase, args.beta1, args.beta2, args.eps,
      1, width, height);
    this.clearParamsPre = createClearParamsBuffer(device, initSiteCount);
    this.clearParamsPost = createClearParamsBuffer(device, initSiteCount);
    this.packParams = createClearParamsBuffer(device, initSiteCount);
    this.splitParams = this.splitIndicesBuf ? createSplitParamsBuffer(device, 0, initSiteCount) : null;
    this.densifyIndicesParams = this.splitIndicesBuf ? createPruneParamsBuffer(device, 0) : null;
    this.pruneIndicesParams = this.plan.needsPrune ? createPruneParamsBuffer(device, 0) : null;
    this.densifyScoreParams = this.statBufs
      ? createDensifyScoreParamsBuffer(device, initSiteCount, this.plan.scorePairsCount, 1.0, args.densifyScoreAlpha)
      : null;
    this.pruneScoreParams = this.plan.needsPrune
      ? createPruneScoreParamsBuffer(device, initSiteCount, this.plan.scorePairsCount, 1.0 / (width * height))
      : null;
    this.tauParams = createTauParamsBuffer(device, initSiteCount, TAU_DIFFUSE_LAMBDA, this.candDownscale);

    // Hilbert resources
    this.dummyHilbertBuf = createDummyStorageBuffer(device);
    if (this.usesHilbert) {
      this.hilbertRes = createHilbertBuffers(device, this.bufferCapacity);
      this.hilbertReady = false;
      this.hilbertSiteCount = 0;
      this.hilbertParams = createHilbertParamsBuffer(
        device, 1, this.hilbertRes.paddedCount, width, height, hilbertBitsForSize(width, height),
      );
      this.hilbertSort = new RadixSortUInt2(device, shaders, this.hilbertRes.paddedCount);
    } else {
      this.hilbertRes = null;
      this.hilbertReady = true;
      this.hilbertSiteCount = 0;
      this.hilbertParams = null;
      this.hilbertSort = null;
    }

    // Densify/prune radix sort (different padded count than Hilbert's)
    this.radixSort = this.plan.needsPairs
      ? new RadixSortUInt2(device, shaders, this.plan.scorePairsCount)
      : null;

    // Initialize sites
    if (initSites) {
      const maxBytes = this.bufferCapacity * SITE_FLOATS * 4;
      const src = initSites.byteLength <= maxBytes
        ? initSites
        : new Float32Array(initSites.buffer, initSites.byteOffset, maxBytes / 4);
      device.queue.writeBuffer(this.sitesBuf, 0, src);
      this.actualSites = initSiteCount;
      this.activeEstimate = 0;
      for (let i = 0; i < initSiteCount; i += 1) {
        if (initSites[i * SITE_FLOATS] >= 0) this.activeEstimate += 1;
      }
      this._log(`Loaded init sites: ${initSiteCount} (active ${this.activeEstimate})`);
    } else {
      const gradThreshold = 0.01;
      const initParamsBuf = createInitParamsBuffer(device, initSiteCount, gradThreshold, 256,
                                                    args.initLogTau, args.initRadius);
      const encoder = device.createCommandEncoder();
      this.initEnc.encode(encoder, this.sitesBuf, this.seedCounter, initParamsBuf,
                           this.targetTex, this.maskTex, initSiteCount);
      device.queue.submit([encoder.finish()]);
      const sitesData = await readSites(device, this.sitesBuf, initSiteCount);
      this.actualSites = initSiteCount;
      this.activeEstimate = 0;
      for (let i = 0; i < initSiteCount; i += 1) {
        if (sitesData[i * SITE_FLOATS] >= 0) this.activeEstimate += 1;
      }
      this._log(`Initialized sites: ${initSiteCount} (active ${this.activeEstimate})`);
    }

    // Prime candidate state (init + seed) — mirrors the one-time setup in Python.
    {
      updateParamsBuffer(device.queue, this.candParams[0], this._paramsArgs(initSiteCount, 0, 0));
      const encoder = device.createCommandEncoder();
      this.candsEnc.encodeInit(encoder, this.candParams[0], this.cand0A, this.cand1A,
                                this.candWidth, this.candHeight);
      encoder.copyTextureToTexture(
        { texture: this.cand0A },
        { texture: this.cand0B },
        [this.candWidth, this.candHeight, 1],
      );
      this.seedEnc.encode(encoder, this.sitesBuf, this.candParams[0], this.cand0A, this.cand0B,
                           initSiteCount);
      device.queue.submit([encoder.finish()]);
      // After seed, the freshly-written texture for cand0 is cand0B; swap so
      // cand0A is current. cand1 was written by init into cand1A, so leave it.
      [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
    }

    this.clearedSiteCount = 0;
    this.jumpPassIndex = 0;
    this.aniStartIter = Math.max(10, Math.floor(args.pruneStart / 2));
    this.effectivePruneStart = args.pruneStart;
    if (args.densify && !args.pruneDuringDensify && args.pruneStart < args.densifyEnd) {
      this.effectivePruneStart = args.densifyEnd;
      this._log(`Note: Adjusting prune start from ${args.pruneStart} -> ${this.effectivePruneStart} ` +
                 `to avoid pruning during densification.`);
    }
    this.trainStartMs = performance.now();
    this.bestPsnr = 0;
  }

  _paramsArgs(siteCount, step, seed) {
    return {
      width: this.width,
      height: this.height,
      siteCount,
      step,
      seed,
      radiusScale: this.args.candRadiusScale,
      radiusProbes: this.args.candRadiusProbes,
      injectCount: this.args.candInjectCount,
      hilbertProbes: this.usesHilbert ? this.args.candHilbertProbes : 0,
      hilbertWindow: this.usesHilbert ? this.args.candHilbertWindow : 0,
      candDownscale: this.candDownscale,
      candWidth: this.candWidth,
      candHeight: this.candHeight,
    };
  }

  _log(msg) { if (this.onLog) this.onLog({ type: "msg", message: msg }); }

  _hilbertOrderBuf() {
    return this.usesHilbert && this.hilbertRes ? this.hilbertRes.order : this.dummyHilbertBuf;
  }
  _hilbertPosBuf() {
    return this.usesHilbert && this.hilbertRes ? this.hilbertRes.pos : this.dummyHilbertBuf;
  }

  _maybeRefreshHilbert(encoder, preSites) {
    if (!this.usesHilbert) return;
    if (this.hilbertReady && this.hilbertSiteCount === preSites) return;
    const bits = hilbertBitsForSize(this.width, this.height);
    updateHilbertParamsBuffer(this.device.queue, this.hilbertParams,
      preSites, this.hilbertRes.paddedCount, this.width, this.height, bits);
    this.hilbertEnc.encodePairs(encoder, this.sitesBuf, this.hilbertRes.pairs, this.hilbertParams,
      this.hilbertRes.paddedCount);
    const maxKeyExclusive = bits >= 16 ? 0xffffffff : (1 << (bits * 2));
    this.hilbertSort.encode(encoder, this.hilbertRes.pairs, maxKeyExclusive);
    this.hilbertEnc.encodeOrder(encoder, this.hilbertRes.pairs, this.hilbertRes.order,
      this.hilbertRes.pos, this.hilbertParams, preSites);
    this.hilbertReady = true;
    this.hilbertSiteCount = preSites;
  }

  async run() {
    const args = this.args;
    const iters = args.iters;
    const logFreq = args.logFreq;
    const viewerFreq = Math.max(1, args.viewerFreq || 10);
    const candUpdateFreq = Math.max(1, args.candFreq);
    const device = this.device;
    const queue = device.queue;

    for (let it = 0; it < iters; it += 1) {
      if (this.signal && this.signal.aborted) {
        this._log("Training aborted.");
        break;
      }

      const preSites = this.actualSites;
      const preActive = this.activeEstimate;
      const updateCandidates = (it % candUpdateFreq === 0);
      const candPasses = this.candUpdatePasses;

      const shouldDensify =
        args.densify &&
        it >= args.densifyStart &&
        it <= args.densifyEnd &&
        (it % args.densifyFreq === 0) &&
        preSites < this.bufferCapacity;
      const shouldPrune =
        args.prunePercentile > 0 &&
        it >= this.effectivePruneStart &&
        it < args.pruneEnd &&
        (it % args.pruneFreq === 0) &&
        this.plan.needsPrune;

      let numToSplit = 0;
      if (shouldDensify) {
        const desired = Math.floor(preActive * args.densifyPercentile);
        const available = this.bufferCapacity - preSites;
        numToSplit = Math.min(desired, available, this.plan.maxSplitIndices);
      }
      const runDensify = shouldDensify && numToSplit > 0;

      const postSites = preSites + numToSplit;
      let postActive = preActive + numToSplit;
      let numToPrune = 0;
      if (shouldPrune) {
        const desired = Math.floor(postActive * args.prunePercentile);
        numToPrune = Math.min(desired, this.plan.maxSplitIndices);
      }
      postActive = Math.max(0, postActive - numToPrune);

      // Update candidate params for each pass
      if (updateCandidates) {
        for (let p = 0; p < candPasses; p += 1) {
          const stepIndex = this.jumpPassIndex + p;
          const step = packJumpStep(stepIndex, this.candWidth, this.candHeight);
          const seed = (stepIndex >>> 16) & 0xffff;
          updateParamsBuffer(queue, this.candParams[p], this._paramsArgs(preSites, step, seed));
        }
      }

      updateCountParamsBuffer(queue, this.clearParamsPre, preSites);
      if (updateCandidates) updateCountParamsBuffer(queue, this.packParams, preSites);
      if (runDensify) {
        updateGradParamsBuffer(queue, this.gradParamsStats, preSites, this.invScaleSq, 0);
        updateDensifyScoreParamsBuffer(queue, this.densifyScoreParams, preSites,
          this.plan.scorePairsCount, 1.0, args.densifyScoreAlpha);
        updatePruneParamsBuffer(queue, this.densifyIndicesParams, numToSplit);
        updateSplitParamsBuffer(queue, this.splitParams, numToSplit, preSites);
      }

      updateCountParamsBuffer(queue, this.clearParamsPost, postSites);
      updateGradParamsBuffer(queue, this.gradParamsGrad, postSites, this.invScaleSq, shouldPrune ? 1 : 0);

      const lrDir = it >= this.aniStartIter ? args.lrDirBase : 0;
      const lrAniso = it >= this.aniStartIter ? args.lrAnisoBase : 0;
      updateAdamParamsBuffer(queue, this.adamParams,
        args.lrPosBase, args.lrTauBase, args.lrRadiusBase, args.lrColorBase,
        lrDir, lrAniso, args.beta1, args.beta2, args.eps, it + 1, this.width, this.height);

      const blend = it / Math.max(1, iters);
      const lambda = TAU_DIFFUSE_LAMBDA * (0.1 + 0.9 * blend);
      updateTauParamsBuffer(queue, this.tauParams, postSites, lambda, this.candDownscale);

      if (shouldPrune) {
        updatePruneScoreParamsBuffer(queue, this.pruneScoreParams, postSites,
          this.plan.scorePairsCount, 1.0 / (this.width * this.height));
        if (numToPrune > 0) updatePruneParamsBuffer(queue, this.pruneIndicesParams, numToPrune);
      }

      const shouldLog = (it % logFreq === 0) || (it === iters - 1);
      const shouldPreview = shouldLog || (it % viewerFreq === 0);
      if (shouldLog || shouldPreview) {
        updateParamsBuffer(queue, this.renderParams, this._paramsArgs(postSites, 0, 0));
      }

      const encoder = device.createCommandEncoder();

      // Candidate update
      if (updateCandidates) {
        this._maybeRefreshHilbert(encoder, preSites);
        this.packEnc.encode(encoder, this.sitesBuf, this.packedSitesBuf, this.packParams, preSites);
        for (let p = 0; p < candPasses; p += 1) {
          this.candsEnc.encodeUpdate(encoder,
            this.packedSitesBuf, this.candParams[p],
            this.cand0A, this.cand1A, this.cand0B, this.cand1B,
            this._hilbertOrderBuf(), this._hilbertPosBuf(),
            this.candWidth, this.candHeight);
          [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
          [this.cand1A, this.cand1B] = [this.cand1B, this.cand1A];
          this.jumpPassIndex += 1;
        }
      }

      // Densify: stats → score → sort → write indices → split
      if (runDensify) {
        for (const buf of this.statBufs) {
          this.clearEnc.encode(encoder, buf, this.clearParamsPre, preSites);
        }
        this.statsEnc.encode(encoder,
          this.sitesBuf, this.gradParamsStats,
          this.cand0A, this.cand1A,
          this.targetTex, this.maskTex,
          this.statBufs, this.width, this.height);
        this.scoreEnc.encodeDensify(encoder, this.sitesBuf,
          this.statBufs[0], this.statBufs[1],
          this.pairsBuf, this.densifyScoreParams, this.plan.scorePairsCount);
        if (this.radixSort) this.radixSort.encode(encoder, this.pairsBuf, 0xffffffff);
        this.writeIdxEnc.encode(encoder, this.pairsBuf, this.splitIndicesBuf,
          this.densifyIndicesParams, numToSplit);
        this.splitEnc.encode(encoder, this.sitesBuf, this.adamBuf, this.splitIndicesBuf,
          this.statBufs, this.splitParams, this.targetTex, numToSplit);
      }

      // Clear removal_delta if pruning; clear grad buffers if site count grew
      if (shouldPrune) {
        this.clearEnc.encode(encoder, this.removalDelta, this.clearParamsPost, postSites);
      }
      if (postSites > this.clearedSiteCount) {
        // Clear the grown portion of the consolidated grads buffer (one u32
        // slot per channel per site).
        this.clearI32Enc.encode(encoder, this.grads, this.clearParamsPost, postSites * 10);
        this.clearedSiteCount = postSites;
      }

      // Gradients
      this.gradsEnc.encode(encoder, this.sitesBuf, this.gradParamsGrad,
        this.cand0A, this.cand1A, this.targetTex, this.maskTex,
        this.grads, this.removalDelta, this.width, this.height);

      // Tau diffusion
      if (TAU_DIFFUSE_PASSES > 0 && TAU_DIFFUSE_LAMBDA > 0) {
        this.tauEnc.encodeExtract(encoder, this.grads, this.tauGradRaw,
          this.clearParamsPost, postSites);
        let curIn = this.tauGradRaw;
        let curOut = this.tauGradTmp;
        for (let p = 0; p < TAU_DIFFUSE_PASSES; p += 1) {
          this.tauEnc.encodeDiffuse(encoder, this.cand0A, this.cand1A,
            this.sitesBuf, this.tauGradRaw, curIn, curOut, this.tauParams, postSites);
          if (curOut === this.tauGradTmp) {
            [curIn, curOut] = [curOut, this.tauGradTmp2];
          } else {
            [curIn, curOut] = [curOut, this.tauGradTmp];
          }
        }
        this.tauEnc.encodeWriteback(encoder, curIn, this.grads,
          this.clearParamsPost, postSites);
      }

      // Adam
      this.adamEnc.encode(encoder, this.sitesBuf, this.adamBuf, this.grads,
        this.adamParams, postSites);

      // Prune: score → sort → write indices → prune
      if (shouldPrune) {
        this.scoreEnc.encodePrune(encoder, this.sitesBuf, this.removalDelta,
          this.pairsBuf, this.pruneScoreParams, this.plan.scorePairsCount);
        if (this.radixSort) this.radixSort.encode(encoder, this.pairsBuf, 0xffffffff);
        if (numToPrune > 0) {
          this.writeIdxEnc.encode(encoder, this.pairsBuf, this.pruneIndicesBuf,
            this.pruneIndicesParams, numToPrune);
          this.pruneEnc.encode(encoder, this.sitesBuf, this.pruneIndicesBuf,
            this.pruneIndicesParams, numToPrune);
        }
      }

      // Preview/log renders. The log path needs a synchronous readback so we
      // can compute PSNR and append the log line before the next iter. The
      // preview path (much more frequent) fires a readback asynchronously —
      // `mapAsync` resolves on a later microtask without blocking the training
      // loop. Only one in-flight preview readback at a time.
      if (shouldPreview) {
        this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
          this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
      }
      queue.submit([encoder.finish()]);

      if (shouldLog) {
        const rgba = await readRgba32FloatTexture(device, this.renderTex, this.width, this.height);
        const sitesData = await readSites(device, this.sitesBuf, postSites);
        this._lastSitesCpu = sitesData;
        const psnr = computePsnrSafe(rgba, this.target, this.width, this.height, this.maskCpu, this.maskSum);
        this.bestPsnr = Math.max(this.bestPsnr, psnr);
        const elapsed = (performance.now() - this.trainStartMs) / 1000;
        const itsPerSec = (it + 1) / Math.max(elapsed, 1e-6);
        if (this.onLog) {
          this.onLog({
            type: "iter",
            it, iters,
            psnr,
            activeSites: postActive,
            totalSites: postSites,
            itsPerSec,
            elapsedSec: elapsed,
            renderRgba: rgba,
            sitesData,
            width: this.width,
            height: this.height,
          });
        }
        await yieldToBrowser();
      } else if (shouldPreview) {
        this._schedulePreviewReadback(it, iters, postActive, postSites);
      } else if (((it + 1) & 0x1f) === 0) {
        await yieldToBrowser();
      }

      this.actualSites = postSites;
      this.activeEstimate = postActive;
    }

    // Final candidate refresh + render, mirroring the epilogue in train_wgpu.main.
    const queue2 = device.queue;
    const finalPasses = this.candUpdatePasses;

    if (finalPasses > 0) {
      if (this.usesHilbert) {
        const encoder = device.createCommandEncoder();
        // Force refresh by clearing cache.
        this.hilbertReady = false;
        this._maybeRefreshHilbert(encoder, this.actualSites);
        device.queue.submit([encoder.finish()]);
      }
      for (let p = 0; p < finalPasses; p += 1) {
        const stepIndex = this.jumpPassIndex + p;
        const step = packJumpStep(stepIndex, this.candWidth, this.candHeight);
        const seed = (stepIndex >>> 16) & 0xffff;
        updateParamsBuffer(queue2, this.candParams[p], this._paramsArgs(this.actualSites, step, seed));
      }
      updateCountParamsBuffer(queue2, this.packParams, this.actualSites);
      const encoder = device.createCommandEncoder();
      this.packEnc.encode(encoder, this.sitesBuf, this.packedSitesBuf, this.packParams, this.actualSites);
      for (let p = 0; p < finalPasses; p += 1) {
        this.candsEnc.encodeUpdate(encoder,
          this.packedSitesBuf, this.candParams[p],
          this.cand0A, this.cand1A, this.cand0B, this.cand1B,
          this._hilbertOrderBuf(), this._hilbertPosBuf(),
          this.candWidth, this.candHeight);
        [this.cand0A, this.cand0B] = [this.cand0B, this.cand0A];
        [this.cand1A, this.cand1B] = [this.cand1B, this.cand1A];
        this.jumpPassIndex += 1;
      }
      device.queue.submit([encoder.finish()]);
    }

    updateParamsBuffer(queue2, this.renderParams, this._paramsArgs(this.actualSites, 0, 0));
    const encoder = device.createCommandEncoder();
    this.renderEnc.encode(encoder, this.sitesBuf, this.renderParams,
      this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    device.queue.submit([encoder.finish()]);
  }

  // Fire a non-blocking readback of the render texture for the preview canvas.
  // Only one readback in flight at a time; subsequent viewer-freq ticks skip
  // the copy if the previous one hasn't resolved yet. This keeps the training
  // loop off the GPU sync critical path (mapAsync would otherwise force a full
  // queue flush every viewer-freq iters, costing ~4× throughput at 2K res).
  _schedulePreviewReadback(it, iters, postActive, postSites) {
    if (this._previewBusy) return;
    this._previewBusy = true;
    const device = this.device;
    const width = this.width;
    const height = this.height;
    const bytesPerPixel = 16;
    const bytesPerRow = width * bytesPerPixel;
    const aligned = Math.floor((bytesPerRow + 255) / 256) * 256;
    const size = aligned * height;

    // Reuse a staging buffer across preview ticks.
    if (!this._previewStaging || this._previewStagingSize !== size) {
      if (this._previewStaging) this._previewStaging.destroy();
      this._previewStaging = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this._previewStagingSize = size;
    }
    const staging = this._previewStaging;

    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: this.renderTex },
      { buffer: staging, bytesPerRow: aligned, rowsPerImage: height },
      [width, height, 1],
    );
    device.queue.submit([enc.finish()]);

    const startMs = this.trainStartMs;
    const totalIters = iters;
    staging.mapAsync(GPUMapMode.READ).then(() => {
      const mapped = new Uint8Array(staging.getMappedRange());
      const rgba = new Float32Array(width * height * 4);
      for (let y = 0; y < height; y += 1) {
        const row = mapped.subarray(y * aligned, y * aligned + bytesPerRow);
        rgba.set(new Float32Array(row.buffer, row.byteOffset, width * 4), y * width * 4);
      }
      staging.unmap();
      this._lastHashedRgba = null;
      this._lastHeatmapRgba = null;
      if (this.onLog) {
        const elapsed = (performance.now() - startMs) / 1000;
        const itsPerSec = (it + 1) / Math.max(elapsed, 1e-6);
        this.onLog({
          type: "preview",
          it, iters: totalIters,
          activeSites: postActive,
          totalSites: postSites,
          itsPerSec,
          elapsedSec: elapsed,
          renderRgba: rgba,
          width, height,
        });
      }
      this._previewBusy = false;
    }).catch((err) => {
      this._previewBusy = false;
      console.warn("Preview readback failed", err);
    });
  }

  // On-demand render using the current site + candidate state. `mode` is one
  // of "render" (color), "hashed" (cell-id hashColor blend) or "heatmap"
  // (tau centroid heatmap).
  async renderView(mode) {
    const device = this.device;
    const queue = device.queue;
    updateParamsBuffer(queue, this.renderParams, this._paramsArgs(this.actualSites, 0, 0));

    const encoder = device.createCommandEncoder();
    if (mode === "hashed") {
      this.renderHashedEnc.encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    } else if (mode === "heatmap") {
      const heat = this._taskHeatmapParamsBuffer();
      this.renderTauEnc.encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cand0A, this.cand1A, this.renderTex,
        heat, this.width, this.height);
    } else {
      this.renderEnc.encode(encoder,
        this.sitesBuf, this.renderParams,
        this.cand0A, this.cand1A, this.renderTex, this.width, this.height);
    }
    queue.submit([encoder.finish()]);
    return await readRgba32FloatTexture(device, this.renderTex, this.width, this.height);
  }

  _taskHeatmapParamsBuffer() {
    // Compute tau min / mean / max from the last-read sites — Metal does the
    // same scan before every heatmap render. Dot radius scales with resolution
    // like `backends/metal/sources/training/training.swift` (`centroidRadius = 2.0`).
    if (!this._tauHeatBuf) {
      this._tauHeatBuf = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    let minT = Infinity;
    let maxT = -Infinity;
    let sumT = 0;
    let count = 0;
    const data = this._lastSitesCpu;
    if (data) {
      for (let i = 0; i < this.actualSites; i += 1) {
        const base = i * SITE_FLOATS;
        if (data[base] < 0) continue;
        const t = Math.exp(data[base + 2]);
        if (!Number.isFinite(t)) continue;
        if (t < minT) minT = t;
        if (t > maxT) maxT = t;
        sumT += t;
        count += 1;
      }
    }
    if (!Number.isFinite(minT)) { minT = 0; maxT = 1; sumT = 0.5; count = 1; }
    const meanT = sumT / Math.max(count, 1);
    const bytes = new ArrayBuffer(16);
    const f = new Float32Array(bytes);
    f[0] = minT;
    f[1] = meanT;
    f[2] = maxT;
    f[3] = 2.5; // dot radius in pixels; matches Metal's centroid_radius
    this.device.queue.writeBuffer(this._tauHeatBuf, 0, bytes);
    return this._tauHeatBuf;
  }

  setLastSites(data) { this._lastSitesCpu = data; }

  async finalize() {
    const rgba = await readRgba32FloatTexture(this.device, this.renderTex, this.width, this.height);
    const sitesArray = await readSites(this.device, this.sitesBuf, this.actualSites);
    return {
      renderRgba: rgba,
      sites: sitesArray,
      count: this.actualSites,
      width: this.width,
      height: this.height,
      bestPsnr: this.bestPsnr,
      trainSeconds: (performance.now() - this.trainStartMs) / 1000,
    };
  }
}

// Mirrors main()'s PSNR computation: only pixels inside the mask count, values
// are sanitized (NaN → 0, +Inf → 1).
function computePsnrSafe(render, target, width, height, maskCpu, maskSum) {
  const n = width * height;
  let acc = 0;
  let counted = 0;
  for (let i = 0; i < n; i += 1) {
    if (maskCpu && maskCpu[i] <= 0) continue;
    for (let c = 0; c < 3; c += 1) {
      let r = render[i * 4 + c];
      if (!Number.isFinite(r)) r = r === Number.POSITIVE_INFINITY ? 1 : 0;
      const d = r - target[i * 4 + c];
      acc += d * d;
    }
    counted += 3;
  }
  if (maskSum && maskSum > 0 && counted === 0) return 0;
  if (counted === 0) counted = 1;
  const mse = acc / counted;
  if (mse <= 0) return 100.0;
  return 20.0 * Math.log10(1.0 / Math.sqrt(mse));
}
