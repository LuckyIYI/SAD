@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : GradParams;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var targetTex : texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> mass : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> energy : array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> err_w : array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> err_wx : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> err_wy : array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> err_wxx : array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> err_wxy : array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> err_wyy : array<atomic<u32>>;
@group(0) @binding(13) var maskTex : texture_2d<f32>;

fn atomicAddFloatMass(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&mass[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&mass[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

fn atomicAddFloatEnergy(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&energy[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&energy[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

fn atomicAddFloatErrW(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&err_w[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&err_w[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

fn atomicAddFloatErrWx(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&err_wx[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&err_wx[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

fn atomicAddFloatErrWy(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&err_wy[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&err_wy[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

fn atomicAddFloatErrWxx(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&err_wxx[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&err_wxx[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

fn atomicAddFloatErrWxy(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&err_wxy[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&err_wxy[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

fn atomicAddFloatErrWyy(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&err_wyy[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&err_wyy[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(targetTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let mask_val = textureLoad(maskTex, vec2<i32>(gid.xy), 0).r;
  if (mask_val <= 0.0) { return; }

  let uv = vec2<f32>(f32(gid.x), f32(gid.y));
  let inv_scale = sqrt(params.invScaleSq);
  let candSize = textureDimensions(inCand0);
  let candCoord = candidateCoord(vec2<u32>(gid.xy), dims, candSize);
  let c0 = textureLoad(inCand0, vec2<i32>(candCoord), 0);
  let c1 = textureLoad(inCand1, vec2<i32>(candCoord), 0);
  let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

  let neg_inf = -1e30;
  var logits : array<f32, 8>;
  var max_logit : f32 = neg_inf;
  for (var i = 0u; i < 8u; i = i + 1u) {
    let idx = candIds[i];
    if (idx >= params.siteCount) {
      logits[i] = neg_inf;
      continue;
    }
    let site = sites[idx];
    if (site.position.x < 0.0 || is_bad(site.position.x) || is_bad(site.position.y) ||
        is_bad(site.log_tau) || is_bad(site.log_aniso)) {
      logits[i] = neg_inf;
      continue;
    }
    let tau = exp(site.log_tau);
    if (is_bad(tau)) {
      logits[i] = neg_inf;
      continue;
    }
    let dmix2 = voronoi_dmix2_fast(site, uv, params.invScaleSq, inv_scale);
    if (is_bad(dmix2)) {
      logits[i] = neg_inf;
      continue;
    }
    logits[i] = -tau * dmix2;
    if (is_bad(logits[i])) {
      logits[i] = neg_inf;
      continue;
    }
    max_logit = max(max_logit, logits[i]);
  }

  if (max_logit == neg_inf || is_bad(max_logit)) { return; }

  var weights : array<f32, 8>;
  var sum_w : f32 = 0.0;
  for (var i = 0u; i < 8u; i = i + 1u) {
    weights[i] = exp(logits[i] - max_logit);
    if (is_bad(weights[i])) { weights[i] = 0.0; }
    sum_w = sum_w + weights[i];
  }
  if (is_bad(sum_w)) { return; }
  let inv_sum = 1.0 / max(sum_w, 1e-8);

  var pred = vec3<f32>(0.0, 0.0, 0.0);
  for (var i = 0u; i < 8u; i = i + 1u) {
    weights[i] = weights[i] * inv_sum;
    let idx = candIds[i];
    if (idx < params.siteCount && sites[idx].position.x >= 0.0 && !is_bad3(site_color(sites[idx]))) {
      pred = pred + weights[i] * site_color(sites[idx]);
    }
  }

  let tgt = textureLoad(targetTex, vec2<i32>(gid.xy), 0).rgb;
  if (is_bad3(tgt)) { return; }
  let diff = pred - tgt;
  let err = dot(diff, diff);
  if (is_bad(err)) { return; }

  for (var i = 0u; i < 8u; i = i + 1u) {
    let idx = candIds[i];
    if (idx >= params.siteCount || sites[idx].position.x < 0.0) { continue; }

    var seen = false;
    for (var j = 0u; j < i; j = j + 1u) {
      if (candIds[j] == idx) { seen = true; break; }
    }
    if (seen) { continue; }

    var w_total = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
      if (candIds[j] == idx) { w_total = w_total + weights[j]; }
    }

    if (w_total > 0.0) {
      atomicAddFloatMass(idx, w_total);
      atomicAddFloatEnergy(idx, w_total * err);
      let werr = w_total * min(err, 1.0);
      atomicAddFloatErrW(idx, werr);
      atomicAddFloatErrWx(idx, werr * uv.x);
      atomicAddFloatErrWy(idx, werr * uv.y);
      atomicAddFloatErrWxx(idx, werr * uv.x * uv.x);
      atomicAddFloatErrWxy(idx, werr * uv.x * uv.y);
      atomicAddFloatErrWyy(idx, werr * uv.y * uv.y);
    }
  }
}
