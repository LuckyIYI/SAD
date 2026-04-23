@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let w = params.width;
  let h = params.height;
  if (gid.x >= w || gid.y >= h) { return; }

  let uv = vec2<f32>(f32(gid.x), f32(gid.y));
  let inv_scale = sqrt(params.invScaleSq);
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let candCoord = candidateCoord(vec2<u32>(gid.xy), vec2<u32>(w, h), vec2<u32>(params.candWidth, params.candHeight));
  let candXY = vec2<i32>(i32(candCoord.x), i32(candCoord.y));
  let c0 = textureLoad(inCand0, candXY, 0);
  let c1 = textureLoad(inCand1, candXY, 0);
  let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

  let inf = 1e30;
  let neg_inf = -1e30;
  var logits : array<f32, 8>;
  var maxLogit : f32 = neg_inf;
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
  maxLogit = max(maxLogit, logits[i]);
}

if (maxLogit == neg_inf || is_bad(maxLogit)) {
  textureStore(outTex, vec2<i32>(ix, iy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
  return;
}

var sumW : f32 = 0.0;
var weights : array<f32, 8>;
for (var i = 0u; i < 8u; i = i + 1u) {
  weights[i] = exp(logits[i] - maxLogit);
  if (is_bad(weights[i])) { weights[i] = 0.0; }
  sumW = sumW + weights[i];
}
if (is_bad(sumW)) {
  textureStore(outTex, vec2<i32>(ix, iy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
  return;
}
let invSum = 1.0 / max(sumW, 1e-8);

  var color = vec3<f32>(0.0);
  for (var i = 0u; i < 8u; i = i + 1u) {
    let wt = weights[i] * invSum;
    let idx = candIds[i];
    if (wt == wt && !is_bad(wt) && idx < params.siteCount && sites[idx].position.x >= 0.0 && !is_bad3(site_color(sites[idx]))) {
      color = color + wt * site_color(sites[idx]);
    }
}

if (is_bad3(color)) {
  color = vec3<f32>(0.0, 0.0, 0.0);
}

textureStore(outTex, vec2<i32>(ix, iy), vec4<f32>(color, 1.0));
}
