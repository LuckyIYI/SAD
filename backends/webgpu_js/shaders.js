// Generated from backends/shared/sad_shared.wgsl.
window.SAD_SHADER_CODE = `
// BEGIN COMMON


struct Site {
  position : vec2<f32>,
  log_tau : f32,
  radius_sq : f32,
  color_r : f32,
  color_g : f32,
  color_b : f32,
  aniso_dir_x : f32,
  aniso_dir_y : f32,
  log_aniso : f32,
}

// Packed site format for candidate search (four half2 values).
struct PackedCandidateSite {
  data : vec4<u32>,
}

fn site_color(site : Site) -> vec3<f32> {
  return vec3<f32>(site.color_r, site.color_g, site.color_b);
}

fn site_aniso_dir(site : Site) -> vec2<f32> {
  return vec2<f32>(site.aniso_dir_x, site.aniso_dir_y);
}

struct Params {
  width : u32,
  height : u32,
  siteCount : u32,
  step : u32,
  invScaleSq : f32,
  seed : u32,
  radiusScale : f32,
  radiusProbes : u32,
  injectCount : u32,
  hilbertProbes : u32,
  hilbertWindow : u32,
  candDownscale : u32,
  candWidth : u32,
  candHeight : u32,
  _pad0 : u32,
  _pad1 : u32,
}

struct ViewParams {
  imageSize : vec2<f32>,
  canvasSize : vec2<f32>,
  pan : vec2<f32>,
  zoom : f32,
  renderMode : u32,
  showDots : u32,
  _pad0 : u32,
  minTau : f32,
  maxTau : f32,
  minAniso : f32,
  maxAniso : f32,
  minRadius : f32,
  maxRadius : f32,
  _pad1 : f32,
}

struct InitParams {
  numSites : u32,
  gradThreshold : f32,
  maxAttempts : u32,
  _pad0 : u32,
  initLogTau : f32,
  initRadius : f32,
  _pad1 : u32,
  _pad2 : u32,
}

struct GradParams {
  siteCount : u32,
  computeRemoval : u32,
  _pad0 : u32,
  _pad1 : u32,
  invScaleSq : f32,
  _pad2 : f32,
  _pad3 : f32,
  _pad4 : f32,
}

struct AdamParams {
  lr_pos : f32,
  lr_tau : f32,
  lr_radius : f32,
  lr_color : f32,
  lr_dir : f32,
  lr_aniso : f32,
  beta1 : f32,
  beta2 : f32,
  eps : f32,
  t : u32,
  width : u32,
  height : u32,
  _pad0 : u32,
}

struct SplitParams {
  numToSplit : u32,
  currentSiteCount : u32,
  _pad0 : u32,
  _pad1 : u32,
}

struct PruneParams {
  count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

struct ClearParams {
  count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

struct TauDiffuseParams {
  siteCount : u32,
  candDownscale : u32,
  _pad0 : u32,
  _pad1 : u32,
  lambda : f32,
  _pad2 : f32,
  _pad3 : f32,
  _pad4 : f32,
}

fn viewToImage(viewPx : vec2<f32>, view : ViewParams) -> vec2<f32> {
  let centered = (viewPx - view.canvasSize * 0.5) / max(view.zoom, 1e-6);
  return centered + view.imageSize * 0.5 + view.pan;
}

fn imageToView(imagePx : vec2<f32>, view : ViewParams) -> vec2<f32> {
  let centered = (imagePx - view.imageSize * 0.5 - view.pan) * view.zoom;
  return centered + view.canvasSize * 0.5;
}

// RNG
fn xorshift32(x : u32) -> u32 {
  var s = x;
  s = s ^ (s << 13u);
  s = s ^ (s >> 17u);
  s = s ^ (s << 5u);
  return s;
}

fn rand01(state : ptr<function, u32>) -> f32 {
  *state = xorshift32(*state);
  return f32(*state) * (1.0 / 4294967296.0);
}

fn is_bad(x : f32) -> bool {
  return (x != x) || abs(x) > 1.0e30;
}

fn is_bad2(v : vec2<f32>) -> bool {
  return is_bad(v.x) || is_bad(v.y);
}

fn is_bad3(v : vec3<f32>) -> bool {
  return is_bad(v.x) || is_bad(v.y) || is_bad(v.z);
}

fn safe_dir(v : vec2<f32>) -> vec2<f32> {
  if (is_bad2(v)) { return vec2<f32>(1.0, 0.0); }
  let len2 = dot(v, v);
  if (len2 < 1.0e-12) { return vec2<f32>(1.0, 0.0); }
  return v * inverseSqrt(len2);
}

fn candidateCoord(gid : vec2<u32>, outputSize : vec2<u32>, candSize : vec2<u32>) -> vec2<u32> {
  let outW = max(outputSize.x, 1u);
  let outH = max(outputSize.y, 1u);
  let candW = max(candSize.x, 1u);
  let candH = max(candSize.y, 1u);
  let cx = min((gid.x * candW) / outW, candW - 1u);
  let cy = min((gid.y * candH) / outH, candH - 1u);
  return vec2<u32>(cx, cy);
}

fn candidateUvFromCoord(coord : vec2<u32>, outputSize : vec2<u32>, candSize : vec2<u32>) -> vec2<f32> {
  let outW = max(outputSize.x, 1u);
  let outH = max(outputSize.y, 1u);
  let candW = max(candSize.x, 1u);
  let candH = max(candSize.y, 1u);
  let outSizeF = vec2<f32>(f32(outW), f32(outH));
  let candSizeF = vec2<f32>(f32(candW), f32(candH));
  var uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5)) * (outSizeF / candSizeF);
  uv = min(uv, outSizeF - vec2<f32>(1.0, 1.0));
  return uv;
}

fn hashColor(siteId : u32) -> vec3<f32> {
  var h = siteId;
  h = (h ^ 61u) ^ (h >> 16u);
  h = h + (h << 3u);
  h = h ^ (h >> 4u);
  let r = h * 0x27d4eb2du;

  h = siteId * 2654435761u;
  h = (h ^ 61u) ^ (h >> 16u);
  h = h + (h << 3u);
  let g = h * 0x27d4eb2du;

  h = siteId * 1103515245u;
  h = (h ^ 61u) ^ (h >> 16u);
  h = h ^ (h >> 4u);
  let b = h * 0x27d4eb2du;

  return vec3<f32>(
    f32(r & 0xFFFFFFu) / f32(0xFFFFFFu),
    f32(g & 0xFFFFFFu) / f32(0xFFFFFFu),
    f32(b & 0xFFFFFFu) / f32(0xFFFFFFu)
  );
}

fn removalDeltaForSite(pred : vec3<f32>, tgt : vec3<f32>, siteColor : vec3<f32>, w_total : f32) -> f32 {
  let keep = 1.0 - w_total;
  if (keep < 1.0e-4) { return 1.0e6; }
  let pred_keep = (pred - w_total * siteColor) / keep;
  let diff = pred - tgt;
  let diff_keep = pred_keep - tgt;
  return dot(diff_keep, diff_keep) - dot(diff, diff);
}

const K_GRAD_QUANT_SCALE : f32 = 1000000.0;
const K_GRAD_QUANT_SCALE_INV : f32 = 1.0 / 1000000.0;

// Compute soft Voronoi distance (same as Metal)
fn voronoi_dmix2(site : Site, uv : vec2<f32>, invScaleSq : f32) -> f32 {
  let diff = uv - site.position;
  let diff2 = dot(diff, diff);
  let proj = dot(site_aniso_dir(site), diff);
  let proj2 = proj * proj;
  let perp2 = max(diff2 - proj2, 0.0);
  let l1 = exp(site.log_aniso);
  let l2 = 1.0 / l1;
  let d2_aniso = l1 * proj2 + l2 * perp2;
  let d2_norm = d2_aniso * invScaleSq;
  let d2_safe = max(d2_norm, 1.0e-8);
  let inv_scale = sqrt(invScaleSq);
  let r_norm = site.radius_sq * inv_scale;
  return (sqrt(d2_safe) - r_norm);
}

fn voronoi_dmix2_fast(site : Site, uv : vec2<f32>, invScaleSq : f32, inv_scale : f32) -> f32 {
  let diff = uv - site.position;
  let diff2 = dot(diff, diff);
  let proj = dot(site_aniso_dir(site), diff);
  let proj2 = proj * proj;
  let perp2 = max(diff2 - proj2, 0.0);
  let l1 = exp(site.log_aniso);
  let l2 = 1.0 / l1;
  let d2_aniso = l1 * proj2 + l2 * perp2;
  let d2_norm = d2_aniso * invScaleSq;
  let d2_safe = max(d2_norm, 1.0e-8);
  let r_norm = site.radius_sq * inv_scale;
  return (sqrt(d2_safe) - r_norm);
}

// Insert candidate into sorted list of 4 (for JFA flood - same as Metal insertClosest4)
fn insertClosest4(
  bestIdx : ptr<function, array<u32, 4>>,
  bestD2 : ptr<function, array<f32, 4>>,
  candIdx : u32,
  imgUv : vec2<f32>,
  site : Site,
  invScaleSq : f32
) {
  if (candIdx == 0xffffffffu) { return; }

  // Check if already in list
  for (var i = 0u; i < 4u; i = i + 1u) {
    if ((*bestIdx)[i] == candIdx) { return; }
  }

  if (site.position.x < 0.0) { return; }

  let dmix2 = voronoi_dmix2(site, imgUv, invScaleSq);
  let tau = max(exp(site.log_tau), 1e-4);
  let d2 = tau * dmix2;

  // Insertion sort into 4-element list (unrolled like Metal)
  if (d2 < (*bestD2)[3]) {
    if (d2 < (*bestD2)[1]) {
      if (d2 < (*bestD2)[0]) {
        // Insert at 0
        (*bestD2)[3] = (*bestD2)[2]; (*bestIdx)[3] = (*bestIdx)[2];
        (*bestD2)[2] = (*bestD2)[1]; (*bestIdx)[2] = (*bestIdx)[1];
        (*bestD2)[1] = (*bestD2)[0]; (*bestIdx)[1] = (*bestIdx)[0];
        (*bestD2)[0] = d2; (*bestIdx)[0] = candIdx;
      } else {
        // Insert at 1
        (*bestD2)[3] = (*bestD2)[2]; (*bestIdx)[3] = (*bestIdx)[2];
        (*bestD2)[2] = (*bestD2)[1]; (*bestIdx)[2] = (*bestIdx)[1];
        (*bestD2)[1] = d2; (*bestIdx)[1] = candIdx;
      }
    } else {
      if (d2 < (*bestD2)[2]) {
        // Insert at 2
        (*bestD2)[3] = (*bestD2)[2]; (*bestIdx)[3] = (*bestIdx)[2];
        (*bestD2)[2] = d2; (*bestIdx)[2] = candIdx;
      } else {
        // Insert at 3
        (*bestD2)[3] = d2; (*bestIdx)[3] = candIdx;
      }
    }
  }
}

// Insert candidate into sorted list of 8 (same as Metal insertClosest8)
fn insertClosest8(
  bestIdx : ptr<function, array<u32, 8>>,
  bestD2 : ptr<function, array<f32, 8>>,
  candIdx : u32,
  imgUv : vec2<f32>,
  site : Site,
  invScaleSq : f32
) {
  if (candIdx == 0xffffffffu) { return; }

  // Check if already in list
  for (var i = 0u; i < 8u; i = i + 1u) {
    if ((*bestIdx)[i] == candIdx) { return; }
  }

  var d2 : f32;
  if (site.position.x < 0.0) {
    d2 = 1e20;
  } else {
    let dmix2 = voronoi_dmix2(site, imgUv, invScaleSq);
    let tau = max(exp(site.log_tau), 1e-4);
    d2 = tau * dmix2;
  }

  // Insertion sort
  if (d2 >= (*bestD2)[7]) { return; }

  var insertPos = 7u;
  for (var i = 0u; i < 8u; i = i + 1u) {
    if (d2 < (*bestD2)[i]) {
      insertPos = i;
      break;
    }
  }

  // Shift elements down
  for (var i = 7u; i > insertPos; i = i - 1u) {
    (*bestD2)[i] = (*bestD2)[i - 1u];
    (*bestIdx)[i] = (*bestIdx)[i - 1u];
  }
  (*bestD2)[insertPos] = d2;
  (*bestIdx)[insertPos] = candIdx;
}

// ==================== CLEAR CANDIDATES ====================
@group(0) @binding(0) var<uniform> clearParams : Params;
@group(0) @binding(1) var clearCand0 : texture_storage_2d<rgba32uint, write>;


// END COMMON
@compute @workgroup_size(8, 8)
fn clearCandidates(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= clearParams.candWidth || gid.y >= clearParams.candHeight) { return; }
  textureStore(clearCand0, vec2<i32>(gid.xy), vec4<u32>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu));
}

// ==================== COPY CANDIDATES ====================
@group(0) @binding(0) var<uniform> copyParams : Params;
@group(0) @binding(1) var copyInCand0 : texture_2d<u32>;
@group(0) @binding(2) var copyOutCand0 : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(8, 8)
fn copyCandidates(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= copyParams.candWidth || gid.y >= copyParams.candHeight) { return; }
  let pixel = vec2<i32>(gid.xy);
  textureStore(copyOutCand0, pixel, textureLoad(copyInCand0, pixel, 0));
}

// ==================== INIT CANDIDATES (Random) ====================
@group(0) @binding(0) var<uniform> initParams : Params;
@group(0) @binding(1) var initCand0 : texture_storage_2d<rgba32uint, write>;
@group(0) @binding(2) var initCand1 : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(8, 8)
fn initCandidates(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= initParams.candWidth || gid.y >= initParams.candHeight) { return; }

  // Random initialization (matching Metal's initCandidates with perPixelMode=false)
  var state = initParams.seed ^ (gid.x * 1973u) ^ (gid.y * 9277u);
  var idx : array<u32, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    state = xorshift32(state + i);
    idx[i] = state % initParams.siteCount;
  }

  textureStore(initCand0, vec2<i32>(gid.xy), vec4<u32>(idx[0], idx[1], idx[2], idx[3]));
  textureStore(initCand1, vec2<i32>(gid.xy), vec4<u32>(idx[4], idx[5], idx[6], idx[7]));
}

// ==================== JFA SEED (Non-deterministic) ====================
@group(0) @binding(0) var<storage, read> seedSites : array<Site>;
@group(0) @binding(1) var<uniform> seedParams : Params;
@group(0) @binding(2) var<uniform> seedView : ViewParams;
@group(0) @binding(3) var seedInCand0 : texture_2d<u32>;
@group(0) @binding(4) var seedOut : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(64)
fn jfaSeed(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= seedParams.siteCount) { return; }

  let site = seedSites[gid.x];
  if (site.position.x < 0.0) { return; }

  let outW = max(seedParams.width, 1u);
  let outH = max(seedParams.height, 1u);
  let candW = max(seedParams.candWidth, 1u);
  let candH = max(seedParams.candHeight, 1u);
  let fx = clamp(site.position.x, 0.0, f32(outW - 1u));
  let fy = clamp(site.position.y, 0.0, f32(outH - 1u));
  let candX = min(u32(fx * f32(candW) / f32(outW)), candW - 1u);
  let candY = min(u32(fy * f32(candH) / f32(outH)), candH - 1u);
  let homePixel = vec2<i32>(i32(candX), i32(candY));

  let existing = textureLoad(seedInCand0, homePixel, 0);
  textureStore(seedOut, homePixel, vec4<u32>(gid.x, existing.y, existing.z, existing.w));
}

// ==================== JFA FLOOD ====================
// One pass with given step size (same as Metal jfaFlood)
@group(0) @binding(0) var<storage, read> floodSites : array<Site>;
@group(0) @binding(1) var<uniform> floodParams : Params;
@group(0) @binding(2) var<uniform> floodView : ViewParams;
@group(0) @binding(3) var floodInCand0 : texture_2d<u32>;
@group(0) @binding(4) var floodOutCand0 : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(8, 8)
fn jfaFlood(@builtin(global_invocation_id) gid : vec3<u32>) {
  let w = floodParams.candWidth;
  let h = floodParams.candHeight;
  if (gid.x >= w || gid.y >= h) { return; }

  let uv = candidateUvFromCoord(vec2<u32>(gid.xy),
                                vec2<u32>(floodParams.width, floodParams.height),
                                vec2<u32>(floodParams.candWidth, floodParams.candHeight));
  let stepSize = i32(floodParams.step);

  // Start with empty list
  var bestIdx : array<u32, 4> = array<u32, 4>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  let inf = 1e20;
  var bestD2 : array<f32, 4> = array<f32, 4>(inf, inf, inf, inf);

  // Sample 3x3 grid at step offset (9 samples including self)
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      var samplePos = vec2<i32>(gid.xy) + vec2<i32>(dx, dy) * stepSize;
      samplePos = clamp(samplePos, vec2<i32>(0), vec2<i32>(i32(w) - 1, i32(h) - 1));

      let cand = textureLoad(floodInCand0, samplePos, 0);

      if (cand.x < floodParams.siteCount) {
        let site = floodSites[cand.x];
        insertClosest4(&bestIdx, &bestD2, cand.x, uv, site, floodParams.invScaleSq);
      }
      if (cand.y < floodParams.siteCount) {
        let site = floodSites[cand.y];
        insertClosest4(&bestIdx, &bestD2, cand.y, uv, site, floodParams.invScaleSq);
      }
      if (cand.z < floodParams.siteCount) {
        let site = floodSites[cand.z];
        insertClosest4(&bestIdx, &bestD2, cand.z, uv, site, floodParams.invScaleSq);
      }
      if (cand.w < floodParams.siteCount) {
        let site = floodSites[cand.w];
        insertClosest4(&bestIdx, &bestD2, cand.w, uv, site, floodParams.invScaleSq);
      }
    }
  }

  textureStore(floodOutCand0, vec2<i32>(gid.xy), vec4<u32>(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]));
}

// ==================== UPDATE CANDIDATES (VPT) ====================
// Matches Metal updateCandidates structure, but uses view-space pixel mapping.
@group(1) @binding(0) var<storage, read> updateSites : array<Site>;
@group(1) @binding(1) var<uniform> updateParams : Params;
@group(1) @binding(2) var<uniform> updateView : ViewParams;
@group(1) @binding(3) var updateInCand0 : texture_2d<u32>;
@group(1) @binding(4) var updateInCand1 : texture_2d<u32>;
@group(1) @binding(5) var updateOutCand0 : texture_storage_2d<rgba32uint, write>;
@group(1) @binding(6) var updateOutCand1 : texture_storage_2d<rgba32uint, write>;

fn insertCandidates4(
  bestIdx : ptr<function, array<u32, 8>>,
  bestD2 : ptr<function, array<f32, 8>>,
  c : vec4<u32>,
  imgUv : vec2<f32>,
  siteCount : u32,
  invScaleSq : f32
) {
  if (c.x < siteCount) {
    let site = updateSites[c.x];
    insertClosest8(bestIdx, bestD2, c.x, imgUv, site, invScaleSq);
  }
  if (c.y < siteCount) {
    let site = updateSites[c.y];
    insertClosest8(bestIdx, bestD2, c.y, imgUv, site, invScaleSq);
  }
  if (c.z < siteCount) {
    let site = updateSites[c.z];
    insertClosest8(bestIdx, bestD2, c.z, imgUv, site, invScaleSq);
  }
  if (c.w < siteCount) {
    let site = updateSites[c.w];
    insertClosest8(bestIdx, bestD2, c.w, imgUv, site, invScaleSq);
  }
}

@compute @workgroup_size(8, 8)
fn updateCandidates(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= updateParams.candWidth || gid.y >= updateParams.candHeight) { return; }

  let stepIndex = updateParams.step & 0xffffu;
  let jumpStep = max(1u, updateParams.step >> 16u);
  let fullStepIndex = (updateParams.seed << 16u) | stepIndex;

  let uv = candidateUvFromCoord(vec2<u32>(gid.xy),
                                vec2<u32>(updateParams.width, updateParams.height),
                                vec2<u32>(updateParams.candWidth, updateParams.candHeight));
  let gi = vec2<i32>(gid.xy);
  let w = i32(updateParams.candWidth);
  let h = i32(updateParams.candHeight);

  // Initialize best list
  var bestIdx : array<u32, 8>;
  let inf = 1e20;
  var bestD2 : array<f32, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    bestIdx[i] = 0xffffffffu;
    bestD2[i] = inf;
  }

  // Read and merge current candidates (temporal reinsertion)
  let self0 = textureLoad(updateInCand0, gi, 0);
  let self1 = textureLoad(updateInCand1, gi, 0);
  insertCandidates4(&bestIdx, &bestD2, self0, uv, updateParams.siteCount, updateParams.invScaleSq);
  insertCandidates4(&bestIdx, &bestD2, self1, uv, updateParams.siteCount, updateParams.invScaleSq);

  // Random probes (radius scales with image width, baseline at 1024w)
  let rad = updateParams.radiusScale * (f32(updateParams.candWidth) / 1024.0);
  var state = (gid.x * 73856093u) ^ (gid.y * 19349663u) ^ ((stepIndex + jumpStep) * 83492791u);
  // 4-connected neighbors (jump schedule)
  let offsets = array<vec2<i32>, 4>(
    vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1)
  );

  for (var i = 0u; i < 4u; i = i + 1u) {
    let p = clamp(gi + offsets[i] * i32(jumpStep), vec2<i32>(0), vec2<i32>(w - 1, h - 1));
    let n0 = textureLoad(updateInCand0, p, 0);
    let n1 = textureLoad(updateInCand1, p, 0);
    insertCandidates4(&bestIdx, &bestD2, n0, uv, updateParams.siteCount, updateParams.invScaleSq);
    insertCandidates4(&bestIdx, &bestD2, n1, uv, updateParams.siteCount, updateParams.invScaleSq);
  }

  for (var r = 0u; r < updateParams.radiusProbes; r = r + 1u) {
    let a = rand01(&state) * 6.2831853;
    let dx = i32(cos(a) * rad);
    let dy = i32(sin(a) * rad);
    let p = clamp(gi + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(w - 1, h - 1));
    if (gid.x == u32(p.x) && gid.y == u32(p.y)) { continue; }
    let n0 = textureLoad(updateInCand0, p, 0);
    let n1 = textureLoad(updateInCand1, p, 0);
    insertCandidates4(&bestIdx, &bestD2, n0, uv, updateParams.siteCount, updateParams.invScaleSq);
    insertCandidates4(&bestIdx, &bestD2, n1, uv, updateParams.siteCount, updateParams.invScaleSq);
  }

  var injectState = (gid.x * 1664525u) ^ (gid.y * 1013904223u) ^ (fullStepIndex * 374761393u);
  for (var i = 0u; i < updateParams.injectCount; i = i + 1u) {
    injectState = xorshift32(injectState + i);
    let cand = injectState % updateParams.siteCount;
    let site = updateSites[cand];
    insertClosest8(&bestIdx, &bestD2, cand, uv, site, updateParams.invScaleSq);
  }

  textureStore(updateOutCand0, gi, vec4<u32>(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]));
  textureStore(updateOutCand1, gi, vec4<u32>(bestIdx[4], bestIdx[5], bestIdx[6], bestIdx[7]));
}

// ==================== PACK CANDIDATE SITES ====================
@group(0) @binding(0) var<storage, read> packSites : array<Site>;
@group(0) @binding(1) var<storage, read_write> packedSites : array<PackedCandidateSite>;
@group(0) @binding(2) var<uniform> packParams : ClearParams;

@compute @workgroup_size(256)
fn packCandidateSites(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= packParams.count) { return; }

  let site = packSites[idx];
  var out : PackedCandidateSite;
  if (site.position.x < 0.0) {
    out.data = vec4<u32>(
      pack2x16float(vec2<f32>(-1.0, -1.0)),
      pack2x16float(vec2<f32>(0.0, 0.0)),
      pack2x16float(vec2<f32>(0.0, 0.0)),
      pack2x16float(vec2<f32>(0.0, 0.0))
    );
  } else {
    out.data = vec4<u32>(
      pack2x16float(site.position),
      pack2x16float(vec2<f32>(site.log_tau, site.radius_sq)),
      pack2x16float(site_aniso_dir(site)),
      pack2x16float(vec2<f32>(site.log_aniso, 0.0))
    );
  }
  packedSites[idx] = out;
}

// ==================== UPDATE CANDIDATES (PACKED VPT) ====================
@group(0) @binding(0) var<storage, read> vptSites : array<PackedCandidateSite>;
@group(0) @binding(1) var<uniform> vptParams : Params;
@group(0) @binding(2) var vptInCand0 : texture_2d<u32>;
@group(0) @binding(3) var vptInCand1 : texture_2d<u32>;
@group(0) @binding(4) var vptOutCand0 : texture_storage_2d<rgba32uint, write>;
@group(0) @binding(5) var vptOutCand1 : texture_storage_2d<rgba32uint, write>;
@group(0) @binding(6) var<storage, read> vptHilbertOrder : array<u32>;
@group(0) @binding(7) var<storage, read> vptHilbertPos : array<u32>;

fn tryInsertPacked8(bestIdx : ptr<function, array<u32, 8>>, bestD2 : ptr<function, array<f32, 8>>,
                    candIdx : u32, uv : vec2<f32>, siteCount : u32,
                    invScaleSq : f32, inv_scale : f32) {
  if (candIdx >= siteCount) { return; }
  for (var i = 0u; i < 8u; i = i + 1u) {
    if ((*bestIdx)[i] == candIdx) { return; }
  }
  let packed = vptSites[candIdx];
  let pos = unpack2x16float(packed.data.x);
  let tau_rad = unpack2x16float(packed.data.y);
  let dir = unpack2x16float(packed.data.z);
  let log_aniso = unpack2x16float(packed.data.w).x;
  var d2 : f32;
  if (pos.x < 0.0 || is_bad2(pos) || is_bad(tau_rad.x) || is_bad(log_aniso) || is_bad2(dir)) {
    return;
  } else {
    let diff = uv - pos;
    let diff2 = dot(diff, diff);
    let proj = dot(dir, diff);
    let proj2 = proj * proj;
    let perp2 = max(diff2 - proj2, 0.0);

    let l1 = exp(log_aniso);
    let l2 = 1.0 / max(l1, 1e-8);
    let d2_aniso = l1 * proj2 + l2 * perp2;
    let d2_norm = d2_aniso * invScaleSq;
    let d2_safe = max(d2_norm, 1e-8);
    let r_norm = tau_rad.y * inv_scale;
    let dmix2 = sqrt(d2_safe) - r_norm;
    if (is_bad(dmix2)) { return; }
    let tau = max(exp(tau_rad.x), 1e-4);
    if (is_bad(tau)) { return; }
    d2 = tau * dmix2;
    if (is_bad(d2)) { return; }
  }
  if (d2 >= (*bestD2)[7]) { return; }
  var insertPos = 7u;
  for (var i = 0u; i < 8u; i = i + 1u) {
    if (d2 < (*bestD2)[i]) { insertPos = i; break; }
  }
  for (var i = 7u; i > insertPos; i = i - 1u) {
    (*bestD2)[i] = (*bestD2)[i - 1u];
    (*bestIdx)[i] = (*bestIdx)[i - 1u];
  }
  (*bestD2)[insertPos] = d2;
  (*bestIdx)[insertPos] = candIdx;
}

@compute @workgroup_size(16, 16)
fn updateCandidatesPacked(@builtin(global_invocation_id) gid : vec3<u32>) {
  let candW = max(vptParams.candWidth, 1u);
  let candH = max(vptParams.candHeight, 1u);
  if (gid.x >= candW || gid.y >= candH) { return; }

  let stepIndex = vptParams.step & 0xffffu;
  let jumpStep = max(1u, vptParams.step >> 16u);
  let fullStepIndex = (vptParams.seed << 16u) | stepIndex;

  let uv = candidateUvFromCoord(vec2<u32>(gid.xy), vec2<u32>(vptParams.width, vptParams.height),
                                vec2<u32>(candW, candH));
  let gi = vec2<i32>(gid.xy);
  let w = i32(candW);
  let h = i32(candH);

  let inv_scale = sqrt(vptParams.invScaleSq);

  var bestIdx : array<u32, 8>;
  let inf = 1e20;
  var bestD2 : array<f32, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    bestIdx[i] = 0xffffffffu;
    bestD2[i] = inf;
  }

  let self0 = textureLoad(vptInCand0, gi, 0);
  let self1 = textureLoad(vptInCand1, gi, 0);
  tryInsertPacked8(&bestIdx, &bestD2, self0.x, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  tryInsertPacked8(&bestIdx, &bestD2, self0.y, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  tryInsertPacked8(&bestIdx, &bestD2, self0.z, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  tryInsertPacked8(&bestIdx, &bestD2, self0.w, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  tryInsertPacked8(&bestIdx, &bestD2, self1.x, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  tryInsertPacked8(&bestIdx, &bestD2, self1.y, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  tryInsertPacked8(&bestIdx, &bestD2, self1.z, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  tryInsertPacked8(&bestIdx, &bestD2, self1.w, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);

  let offsets = array<vec2<i32>, 4>(
    vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1)
  );

  for (var i = 0u; i < 4u; i = i + 1u) {
    let p = clamp(gi + offsets[i] * i32(jumpStep), vec2<i32>(0), vec2<i32>(w - 1, h - 1));
    let n0 = textureLoad(vptInCand0, p, 0);
    let n1 = textureLoad(vptInCand1, p, 0);
    tryInsertPacked8(&bestIdx, &bestD2, n0.x, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n0.y, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n0.z, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n0.w, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n1.x, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n1.y, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n1.z, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n1.w, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  }

  let rad = vptParams.radiusScale * (f32(candW) / 1024.0);
  var state = (gid.x * 73856093u) ^ (gid.y * 19349663u) ^ ((stepIndex + jumpStep) * 83492791u);
  for (var r = 0u; r < vptParams.radiusProbes; r = r + 1u) {
    let a = rand01(&state) * 6.2831853;
    let dx = i32(cos(a) * rad);
    let dy = i32(sin(a) * rad);
    let p = clamp(gi + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(w - 1, h - 1));
    if (gid.x == u32(p.x) && gid.y == u32(p.y)) { continue; }
    let n0 = textureLoad(vptInCand0, p, 0);
    let n1 = textureLoad(vptInCand1, p, 0);
    tryInsertPacked8(&bestIdx, &bestD2, n0.x, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n0.y, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n0.z, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n0.w, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n1.x, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n1.y, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n1.z, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
    tryInsertPacked8(&bestIdx, &bestD2, n1.w, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  }

  if (vptParams.hilbertProbes > 0u && vptParams.hilbertWindow > 0u) {
    let bestCand = bestIdx[0];
    if (bestCand < vptParams.siteCount) {
      let pos = vptHilbertPos[bestCand];
      let span = vptParams.hilbertWindow * 2u + 1u;
      var hState = (gid.x * 2654435761u) ^ (gid.y * 1597334677u) ^ (fullStepIndex * 374761393u);
      for (var i = 0u; i < vptParams.hilbertProbes; i = i + 1u) {
        hState = xorshift32(hState + i);
        let offset = i32(hState % span) - i32(vptParams.hilbertWindow);
        let idx = clamp(i32(pos) + offset, 0, i32(vptParams.siteCount - 1u));
        let cand = vptHilbertOrder[u32(idx)];
        tryInsertPacked8(&bestIdx, &bestD2, cand, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
      }
    }
  }

  var injectState = (gid.x * 1664525u) ^ (gid.y * 1013904223u) ^ (fullStepIndex * 374761393u);
  for (var i = 0u; i < vptParams.injectCount; i = i + 1u) {
    injectState = xorshift32(injectState + i);
    let cand = injectState % vptParams.siteCount;
    tryInsertPacked8(&bestIdx, &bestD2, cand, uv, vptParams.siteCount, vptParams.invScaleSq, inv_scale);
  }

  textureStore(vptOutCand0, gi, vec4<u32>(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]));
  textureStore(vptOutCand1, gi, vec4<u32>(bestIdx[4], bestIdx[5], bestIdx[6], bestIdx[7]));
}

// ==================== RENDER ====================
struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@group(2) @binding(0) var<storage, read> renderSites : array<Site>;
@group(2) @binding(1) var<uniform> renderParams : Params;
@group(2) @binding(2) var<uniform> renderView : ViewParams;
@group(2) @binding(3) var renderCand0 : texture_2d<u32>;
@group(2) @binding(4) var renderCand1 : texture_2d<u32>;

@vertex
fn vsMain(@location(0) position : vec2<f32>) -> VertexOut {
  var out : VertexOut;
  out.position = vec4<f32>(position, 0.0, 1.0);
  var uv = (position + vec2<f32>(1.0)) * 0.5;
  uv.y = 1.0 - uv.y;  // Flip Y to match texture coordinates
  out.uv = uv;
  return out;
}

@fragment
fn fsMain(in : VertexOut) -> @location(0) vec4<f32> {
  // Convert normalized UV to canvas pixel coordinates
  let canvasPx = in.uv * renderView.canvasSize;

  // Transform to image space (centered pan/zoom)
  let centered = (canvasPx - renderView.canvasSize * 0.5) / renderView.zoom;
  let imgPxF = centered + renderView.imageSize * 0.5 + renderView.pan;

  // Out of bounds check
  if (imgPxF.x < 0.0 || imgPxF.y < 0.0 || imgPxF.x >= renderView.imageSize.x || imgPxF.y >= renderView.imageSize.y) {
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
  }

  let ix = i32(imgPxF.x);
  let iy = i32(imgPxF.y);
  let w = i32(renderParams.width);
  let h = i32(renderParams.height);

  if (ix < 0 || iy < 0 || ix >= w || iy >= h) {
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
  }

  // Use continuous image-space coordinate for softmax.
  let uv = imgPxF;

  // Read candidates from the downscaled grid that covers this image pixel.
  let outW = max(renderParams.width, 1u);
  let outH = max(renderParams.height, 1u);
  let candW = max(renderParams.candWidth, 1u);
  let candH = max(renderParams.candHeight, 1u);
  let fx = clamp(imgPxF.x, 0.0, f32(outW - 1u));
  let fy = clamp(imgPxF.y, 0.0, f32(outH - 1u));
  let candX = min(u32(fx * f32(candW) / f32(outW)), candW - 1u);
  let candY = min(u32(fy * f32(candH) / f32(outH)), candH - 1u);
  let candXY = vec2<i32>(i32(candX), i32(candY));
  let c0 = textureLoad(renderCand0, candXY, 0);
  let c1 = textureLoad(renderCand1, candXY, 0);
  let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

  // Compute logits at integer pixel coordinate (same as Metal)
  var logits : array<f32, 8>;
  let inf = 1e20;
  let negInf = -1e20;
  var maxLogit = negInf;
  for (var i = 0u; i < 8u; i = i + 1u) {
    let idx = candIds[i];
    if (idx >= renderParams.siteCount) {
      logits[i] = negInf;
      continue;
    }
    let site = renderSites[idx];
    if (site.position.x < 0.0) {
      logits[i] = negInf;
      continue;
    }
    let tau = exp(site.log_tau);
    let dmix2 = voronoi_dmix2(site, uv, renderParams.invScaleSq);
    logits[i] = -tau * dmix2;
    maxLogit = max(maxLogit, logits[i]);
  }

  // Early exit if all candidates invalid
  if (maxLogit == negInf) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  // Softmax
  var sumW = 0.0;
  var weights : array<f32, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    weights[i] = exp(logits[i] - maxLogit);
    sumW = sumW + weights[i];
  }
  let invSum = 1.0 / max(sumW, 1e-8);

  // Blend colors based on render mode
  var color = vec3<f32>(0.0);
  var bestId = 0u;
  var bestW = 0.0;

  // Find best site for dots overlay
  for (var i = 0u; i < 8u; i = i + 1u) {
    let w = weights[i] * invSum;
    if (w > bestW) {
      bestW = w;
      bestId = candIds[i];
    }
  }

  // Render mode: 0=Color, 1=IDs, 2=Tau, 3=Anisotropy, 4=Radius
  if (renderView.renderMode == 0u) {
    // Color mode
    for (var i = 0u; i < 8u; i = i + 1u) {
      let w = weights[i] * invSum;
      let idx = candIds[i];
      if (idx < renderParams.siteCount) {
        let site = renderSites[idx];
        if (site.position.x >= 0.0 && w == w && abs(w) < inf) {
          color = color + w * site_color(site);
        }
      }
    }
  } else if (renderView.renderMode == 1u) {
    // IDs mode
    for (var i = 0u; i < 8u; i = i + 1u) {
      let w = weights[i] * invSum;
      let idx = candIds[i];
      if (idx < renderParams.siteCount) {
        let site = renderSites[idx];
        if (site.position.x >= 0.0 && w == w && abs(w) < inf) {
          color = color + w * hashColor(idx);
        }
      }
    }
  } else if (renderView.renderMode == 2u) {
    // Tau mode
    for (var i = 0u; i < 8u; i = i + 1u) {
      let w = weights[i] * invSum;
      let idx = candIds[i];
      if (idx < renderParams.siteCount) {
        let site = renderSites[idx];
        if (site.position.x >= 0.0 && w == w && abs(w) < inf) {
          let normalized = (site.log_tau - renderView.minTau) / (renderView.maxTau - renderView.minTau);
          color = color + w * vec3<f32>(normalized);
        }
      }
    }
  } else if (renderView.renderMode == 3u) {
    // Anisotropy mode
    for (var i = 0u; i < 8u; i = i + 1u) {
      let w = weights[i] * invSum;
      let idx = candIds[i];
      if (idx < renderParams.siteCount) {
        let site = renderSites[idx];
        if (site.position.x >= 0.0 && w == w && abs(w) < inf) {
          let normalized = (site.log_aniso - renderView.minAniso) / (renderView.maxAniso - renderView.minAniso);
          color = color + w * vec3<f32>(normalized);
        }
      }
    }
  } else if (renderView.renderMode == 4u) {
    // Radius mode
    for (var i = 0u; i < 8u; i = i + 1u) {
      let w = weights[i] * invSum;
      let idx = candIds[i];
      if (idx < renderParams.siteCount) {
        let site = renderSites[idx];
        if (site.position.x >= 0.0 && w == w && abs(w) < inf) {
          let normalized = (site.radius_sq - renderView.minRadius) / (renderView.maxRadius - renderView.minRadius);
          color = color + w * vec3<f32>(normalized);
        }
      }
    }
  }

  if (renderView.showDots != 0u && bestId < renderParams.siteCount) {
    let site = renderSites[bestId];
    if (site.position.x >= 0.0) {
      let d = distance(imgPxF, site.position);
      if (d <= 2.0) {
        color = mix(color, vec3<f32>(1.0), 0.9);
      }
    }
  }
  return vec4<f32>(color, 1.0);
}

`;
