// Direct port of backends/webgpu_py/encoders.py.
// One class per WGSL kernel entry point; bindings match the WGSL @group(0)
// declarations one-for-one.

function makePipeline(device, code, entryPoint = "main") {
  const module = device.createShaderModule({ code });
  if (module.getCompilationInfo) {
    module.getCompilationInfo().then((info) => {
      for (const m of info.messages) {
        if (m.type === "error") {
          console.error(`WGSL ${entryPoint} ${m.lineNum}:${m.linePos}: ${m.message}`);
        }
      }
    });
  }
  return device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint },
  });
}

function getTextureView(viewCache, texture) {
  let view = viewCache.get(texture);
  if (!view) {
    view = texture.createView();
    viewCache.set(texture, view);
  }
  return view;
}

function _unusedBindGroupKey(layout, entries) {
  const parts = [layout];
  for (const entry of entries) {
    const binding = entry.binding;
    const res = entry.resource;
    if (res && typeof res === "object" && "buffer" in res) {
      parts.push("b", binding, res.buffer);
    } else {
      parts.push("t", binding, res);
    }
  }
  return parts.join("");
}

// Naive stringified cache keys (what the Python encoders do via id()) collide
// in JS because all objects stringify the same. Fresh bind groups per dispatch.
function getBindGroup(device, _cache, layout, entries) {
  return device.createBindGroup({ layout, entries });
}

function dispatch1D(encoder, pipeline, bg, workgroups) {
  const p = encoder.beginComputePass();
  p.setPipeline(pipeline);
  p.setBindGroup(0, bg);
  p.dispatchWorkgroups(workgroups, 1, 1);
  p.end();
}

function dispatch2D(encoder, pipeline, bg, gx, gy) {
  const p = encoder.beginComputePass();
  p.setPipeline(pipeline);
  p.setBindGroup(0, bg);
  p.dispatchWorkgroups(gx, gy, 1);
  p.end();
}

// ---------------------------------------------------------------------------

export class InitGradientEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.INIT_GRADIENT);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, seedCounter, initParams, targetTex, maskTex, numSites) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: seedCounter } },
      { binding: 2, resource: { buffer: initParams } },
      { binding: 3, resource: getTextureView(this._views, targetTex) },
      { binding: 4, resource: getTextureView(this._views, maskTex) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(numSites / 64));
  }
}

export class CandidatesEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.initPipeline = makePipeline(device, shaders.INIT_CAND);
    this.updatePipeline = makePipeline(device, shaders.VPT);
    this._views = new Map();
    this._bgs = new Map();
  }
  encodeInit(encoder, paramsBuf, cand0, cand1, candWidth, candHeight) {
    const entries = [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: getTextureView(this._views, cand0) },
      { binding: 2, resource: getTextureView(this._views, cand1) },
    ];
    const layout = this.initPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.initPipeline, bg,
      Math.ceil(candWidth / 16), Math.ceil(candHeight / 16));
  }
  encodeUpdate(encoder, packedSitesBuffer, paramsBuf, cand0In, cand1In, cand0Out, cand1Out,
                hilbertOrder, hilbertPos, candWidth, candHeight) {
    const entries = [
      { binding: 0, resource: { buffer: packedSitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0In) },
      { binding: 3, resource: getTextureView(this._views, cand1In) },
      { binding: 4, resource: getTextureView(this._views, cand0Out) },
      { binding: 5, resource: getTextureView(this._views, cand1Out) },
      { binding: 6, resource: { buffer: hilbertOrder } },
      { binding: 7, resource: { buffer: hilbertPos } },
    ];
    const layout = this.updatePipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.updatePipeline, bg,
      Math.ceil(candWidth / 16), Math.ceil(candHeight / 16));
  }
}

export class SeedCandidatesEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.SEED_CAND);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, paramsBuf, cand0In, cand0Out, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0In) },
      { binding: 3, resource: getTextureView(this._views, cand0Out) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(siteCount / 64));
  }
}

export class JFAClearEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.JFA_CLEAR);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, paramsBuf, cand0, cand1, candWidth, candHeight) {
    const entries = [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: getTextureView(this._views, cand0) },
      { binding: 2, resource: getTextureView(this._views, cand1) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(candWidth / 16), Math.ceil(candHeight / 16));
  }
}

export class JFAFloodEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.JFA_FLOOD);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, paramsBuf, candIn, candOut, candWidth, candHeight) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, candIn) },
      { binding: 3, resource: getTextureView(this._views, candOut) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(candWidth / 16), Math.ceil(candHeight / 16));
  }
}

export class CandidatePackEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.CANDIDATE_PACK);
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, packedBuffer, paramsBuf, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: packedBuffer } },
      { binding: 2, resource: { buffer: paramsBuf } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(siteCount / 256));
  }
}

export class RenderEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.RENDER);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, paramsBuf, cand0, cand1, outputTex, width, height) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0) },
      { binding: 3, resource: getTextureView(this._views, cand1) },
      { binding: 4, resource: getTextureView(this._views, outputTex) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

export class GradientsEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.GRADIENTS_TILED);
    this._views = new Map();
    this._bgs = new Map();
  }
  // gradsBuffer is the consolidated interleaved atomic<i32> buffer of size
  // capacity * 10; index with `idx * 10 + channel`.
  encode(encoder, sitesBuffer, gradParams, cand0, cand1, targetTex, maskTex,
         gradsBuffer, removalDelta, width, height) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: gradParams } },
      { binding: 2, resource: getTextureView(this._views, cand0) },
      { binding: 3, resource: getTextureView(this._views, cand1) },
      { binding: 4, resource: getTextureView(this._views, targetTex) },
      { binding: 5, resource: { buffer: gradsBuffer } },
      { binding: 6, resource: { buffer: removalDelta } },
      { binding: 7, resource: getTextureView(this._views, maskTex) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

export class AdamEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.ADAM);
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, adamBuffer, gradsBuffer, adamParams, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: adamBuffer } },
      { binding: 2, resource: { buffer: gradsBuffer } },
      { binding: 3, resource: { buffer: adamParams } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(siteCount / 256));
  }
}

export class TauEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.extractPipeline = makePipeline(device, shaders.TAU_EXTRACT);
    this.diffusePipeline = makePipeline(device, shaders.TAU_DIFFUSE);
    this.writebackPipeline = makePipeline(device, shaders.TAU_WRITEBACK);
    this._views = new Map();
    this._bgs = new Map();
  }
  // gradsBuffer is the consolidated gradient buffer; extract reads channel 2
  // (log_tau) into a plain f32 buffer.
  encodeExtract(encoder, gradsBuffer, gradOut, clearParams, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: gradsBuffer } },
      { binding: 1, resource: { buffer: gradOut } },
      { binding: 2, resource: { buffer: clearParams } },
    ];
    const layout = this.extractPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.extractPipeline, bg, Math.ceil(siteCount / 256));
  }
  encodeDiffuse(encoder, cand0, cand1, sitesBuffer, gradRaw, gradIn, gradOut, tauParams, siteCount) {
    const entries = [
      { binding: 0, resource: getTextureView(this._views, cand0) },
      { binding: 1, resource: getTextureView(this._views, cand1) },
      { binding: 2, resource: { buffer: sitesBuffer } },
      { binding: 3, resource: { buffer: gradRaw } },
      { binding: 4, resource: { buffer: gradIn } },
      { binding: 5, resource: { buffer: gradOut } },
      { binding: 6, resource: { buffer: tauParams } },
    ];
    const layout = this.diffusePipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.diffusePipeline, bg, Math.ceil(siteCount / 256));
  }
  // gradsBuffer is the consolidated gradient buffer; writeback writes diffused
  // tau values back to channel 2.
  encodeWriteback(encoder, gradIn, gradsBuffer, clearParams, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: gradIn } },
      { binding: 1, resource: { buffer: gradsBuffer } },
      { binding: 2, resource: { buffer: clearParams } },
    ];
    const layout = this.writebackPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.writebackPipeline, bg, Math.ceil(siteCount / 256));
  }
}

export class StatsEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.STATS);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, gradParams, cand0, cand1, targetTex, maskTex,
         statBuffers, width, height) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: gradParams } },
      { binding: 2, resource: getTextureView(this._views, cand0) },
      { binding: 3, resource: getTextureView(this._views, cand1) },
      { binding: 4, resource: getTextureView(this._views, targetTex) },
      { binding: 5, resource: { buffer: statBuffers[0] } },
      { binding: 6, resource: { buffer: statBuffers[1] } },
      { binding: 7, resource: { buffer: statBuffers[2] } },
      { binding: 8, resource: { buffer: statBuffers[3] } },
      { binding: 9, resource: { buffer: statBuffers[4] } },
      { binding: 10, resource: { buffer: statBuffers[5] } },
      { binding: 11, resource: { buffer: statBuffers[6] } },
      { binding: 12, resource: { buffer: statBuffers[7] } },
      { binding: 13, resource: getTextureView(this._views, maskTex) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

export class SplitEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.SPLIT);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, adamBuffer, splitIndices, statBuffers, splitParams,
         targetTex, numToSplit) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: adamBuffer } },
      { binding: 2, resource: { buffer: splitIndices } },
      { binding: 3, resource: { buffer: statBuffers[0] } },
      { binding: 4, resource: { buffer: statBuffers[2] } },
      { binding: 5, resource: { buffer: statBuffers[3] } },
      { binding: 6, resource: { buffer: statBuffers[4] } },
      { binding: 7, resource: { buffer: statBuffers[5] } },
      { binding: 8, resource: { buffer: statBuffers[6] } },
      { binding: 9, resource: { buffer: statBuffers[7] } },
      { binding: 10, resource: { buffer: splitParams } },
      { binding: 11, resource: getTextureView(this._views, targetTex) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(numToSplit / 256));
  }
}

export class PruneEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.PRUNE);
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, indices, pruneParams, count) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: indices } },
      { binding: 2, resource: { buffer: pruneParams } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(count / 256));
  }
}

export class ClearBufferEncoder {
  constructor(device, shaders, { useI32 = false } = {}) {
    this.device = device;
    const code = useI32 ? shaders.CLEAR_BUFFER_I32 : shaders.CLEAR_BUFFER_U32;
    this.pipeline = makePipeline(device, code);
    this._bgs = new Map();
  }
  encode(encoder, buffer, clearParams, count) {
    const entries = [
      { binding: 0, resource: { buffer } },
      { binding: 1, resource: { buffer: clearParams } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(count / 256));
  }
}

export class ScorePairsEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.densifyPipeline = makePipeline(device, shaders.SCORE_PAIRS_DENSIFY);
    this.prunePipeline = makePipeline(device, shaders.SCORE_PAIRS_PRUNE);
    this._bgs = new Map();
  }
  encodeDensify(encoder, sitesBuffer, mass, energy, pairs, paramsBuf, pairCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: mass } },
      { binding: 2, resource: { buffer: energy } },
      { binding: 3, resource: { buffer: pairs } },
      { binding: 4, resource: { buffer: paramsBuf } },
    ];
    const layout = this.densifyPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.densifyPipeline, bg, Math.ceil(pairCount / 256));
  }
  encodePrune(encoder, sitesBuffer, removalDelta, pairs, paramsBuf, pairCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: removalDelta } },
      { binding: 2, resource: { buffer: pairs } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ];
    const layout = this.prunePipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.prunePipeline, bg, Math.ceil(pairCount / 256));
  }
}

export class WriteIndicesEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.WRITE_INDICES);
    this._bgs = new Map();
  }
  encode(encoder, sortedPairs, indices, paramsBuf, count) {
    const entries = [
      { binding: 0, resource: { buffer: sortedPairs } },
      { binding: 1, resource: { buffer: indices } },
      { binding: 2, resource: { buffer: paramsBuf } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.pipeline, bg, Math.ceil(count / 256));
  }
}

// Softmax-blended hashColor rendering — same bindings as the normal render
// kernel (sites, params, cand0, cand1, outTex).
export class RenderHashedEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.RENDER_HASHED);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, paramsBuf, cand0, cand1, outputTex, width, height) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0) },
      { binding: 3, resource: getTextureView(this._views, cand1) },
      { binding: 4, resource: getTextureView(this._views, outputTex) },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

// Tau heatmap overlay — adds a TauHeatmapParams uniform (bind 5) on top of the
// normal render bindings.
export class RenderTauHeatmapEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.pipeline = makePipeline(device, shaders.RENDER_TAU_HEATMAP);
    this._views = new Map();
    this._bgs = new Map();
  }
  encode(encoder, sitesBuffer, paramsBuf, cand0, cand1, outputTex, heatParams, width, height) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: getTextureView(this._views, cand0) },
      { binding: 3, resource: getTextureView(this._views, cand1) },
      { binding: 4, resource: getTextureView(this._views, outputTex) },
      { binding: 5, resource: { buffer: heatParams } },
    ];
    const layout = this.pipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch2D(encoder, this.pipeline, bg,
      Math.ceil(width / 16), Math.ceil(height / 16));
  }
}

export class HilbertEncoder {
  constructor(device, shaders) {
    this.device = device;
    this.buildPipeline = makePipeline(device, shaders.HILBERT, "buildHilbertPairs");
    this.writePipeline = makePipeline(device, shaders.HILBERT, "writeHilbertOrder");
    this._bgs = new Map();
  }
  encodePairs(encoder, sitesBuffer, pairsBuffer, paramsBuf, paddedCount) {
    const entries = [
      { binding: 0, resource: { buffer: sitesBuffer } },
      { binding: 1, resource: { buffer: pairsBuffer } },
      { binding: 2, resource: { buffer: paramsBuf } },
    ];
    const layout = this.buildPipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.buildPipeline, bg, Math.ceil(paddedCount / 256));
  }
  encodeOrder(encoder, pairsBuffer, orderBuffer, posBuffer, paramsBuf, siteCount) {
    const entries = [
      { binding: 0, resource: { buffer: pairsBuffer } },
      { binding: 1, resource: { buffer: orderBuffer } },
      { binding: 2, resource: { buffer: posBuffer } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ];
    const layout = this.writePipeline.getBindGroupLayout(0);
    const bg = getBindGroup(this.device, this._bgs, layout, entries);
    dispatch1D(encoder, this.writePipeline, bg, Math.ceil(siteCount / 256));
  }
}
