@group(0) @binding(0) var<storage, read_write> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> adam : array<f32>;
@group(0) @binding(2) var<storage, read> splitIndices : array<u32>;
@group(0) @binding(3) var<storage, read_write> mass : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> err_w : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> err_wx : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> err_wy : array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> err_wxx : array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> err_wxy : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> err_wyy : array<atomic<u32>>;
@group(0) @binding(10) var<uniform> params : SplitParams;
@group(0) @binding(11) var targetTex : texture_2d<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.numToSplit) { return; }

  let parentIdx = splitIndices[gid.x];
  let parent = sites[parentIdx];
  if (parent.position.x < 0.0) { return; }

  let dims = textureDimensions(targetTex);
  let width = i32(dims.x);
  let height = i32(dims.y);

  var ew = bitcast<f32>(atomicLoad(&err_w[parentIdx]));
  var axis = site_aniso_dir(parent);
  var center = parent.position;
  var logAniso = parent.log_aniso * 0.8;

  if (ew > 1e-3) {
    let mx = bitcast<f32>(atomicLoad(&err_wx[parentIdx])) / ew;
    let my = bitcast<f32>(atomicLoad(&err_wy[parentIdx])) / ew;
    center = mix(parent.position, vec2<f32>(mx, my), 0.6);

    var exx = bitcast<f32>(atomicLoad(&err_wxx[parentIdx])) / ew - mx * mx;
    var exy = bitcast<f32>(atomicLoad(&err_wxy[parentIdx])) / ew - mx * my;
    var eyy = bitcast<f32>(atomicLoad(&err_wyy[parentIdx])) / ew - my * my;

    exx = max(exx, 1e-4);
    eyy = max(eyy, 1e-4);

    let theta = 0.5 * atan2(2.0 * exy, exx - eyy);
    axis = vec2<f32>(cos(theta), sin(theta));

    let trace = exx + eyy;
    let disc = sqrt(max(0.0, 0.25 * (exx - eyy) * (exx - eyy) + exy * exy));
    let lambda1 = max(1e-4, 0.5 * trace + disc);
    let lambda2 = max(1e-4, 0.5 * trace - disc);
    logAniso = clamp(-0.5 * log(lambda1 / lambda2), -2.0, 2.0);
  } else {
    let ip = vec2<i32>(clamp(parent.position, vec2<f32>(0.0), vec2<f32>(f32(width - 1), f32(height - 1))));

    let pTL = clamp(ip + vec2<i32>(-1, -1), vec2<i32>(0), vec2<i32>(width - 1, height - 1));
    let pT  = clamp(ip + vec2<i32>( 0, -1), vec2<i32>(0), vec2<i32>(width - 1, height - 1));
    let pTR = clamp(ip + vec2<i32>( 1, -1), vec2<i32>(0), vec2<i32>(width - 1, height - 1));
    let pL  = clamp(ip + vec2<i32>(-1,  0), vec2<i32>(0), vec2<i32>(width - 1, height - 1));
    let pR  = clamp(ip + vec2<i32>( 1,  0), vec2<i32>(0), vec2<i32>(width - 1, height - 1));
    let pBL = clamp(ip + vec2<i32>(-1,  1), vec2<i32>(0), vec2<i32>(width - 1, height - 1));
    let pB  = clamp(ip + vec2<i32>( 0,  1), vec2<i32>(0), vec2<i32>(width - 1, height - 1));
    let pBR = clamp(ip + vec2<i32>( 1,  1), vec2<i32>(0), vec2<i32>(width - 1, height - 1));

    let TL = dot(textureLoad(targetTex, pTL, 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let T  = dot(textureLoad(targetTex, pT, 0).rgb,  vec3<f32>(0.299, 0.587, 0.114));
    let TR = dot(textureLoad(targetTex, pTR, 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let L  = dot(textureLoad(targetTex, pL, 0).rgb,  vec3<f32>(0.299, 0.587, 0.114));
    let R  = dot(textureLoad(targetTex, pR, 0).rgb,  vec3<f32>(0.299, 0.587, 0.114));
    let BL = dot(textureLoad(targetTex, pBL, 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let B  = dot(textureLoad(targetTex, pB, 0).rgb,  vec3<f32>(0.299, 0.587, 0.114));
    let BR = dot(textureLoad(targetTex, pBR, 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let gx = (-TL + TR - 2.0 * L + 2.0 * R - BL + BR) / 4.0;
    let gy = (-TL - 2.0 * T - TR + BL + 2.0 * B + BR) / 4.0;
    let grad = vec2<f32>(gx, gy);
    let g2 = dot(grad, grad);
    if (g2 > 1e-8) { axis = grad * inverseSqrt(g2); }
  }

  axis = safe_dir(axis);
  let m = max(bitcast<f32>(atomicLoad(&mass[parentIdx])), 1.0);
  let offsetDist = clamp(sqrt(m) * 0.5, 1.5, 48.0);
  let offset = axis * offsetDist;

  let posA = clamp(center + offset, vec2<f32>(0.0), vec2<f32>(f32(width - 1), f32(height - 1)));
  let posB = clamp(center - offset, vec2<f32>(0.0), vec2<f32>(f32(width - 1), f32(height - 1)));
  let colA = textureLoad(targetTex, vec2<i32>(posA), 0).rgb;
  let colB = textureLoad(targetTex, vec2<i32>(posB), 0).rgb;

  let childIdx = params.currentSiteCount + gid.x;
  if (childIdx == parentIdx) { return; }

  var child0 = parent;
  var child1 = parent;
  child0.position = posA;
  child1.position = posB;
  child0.color_r = colA.r;
  child0.color_g = colA.g;
  child0.color_b = colA.b;
  child1.color_r = colB.r;
  child1.color_g = colB.g;
  child1.color_b = colB.b;
  child0.log_tau = parent.log_tau - 0.25;
  child1.log_tau = parent.log_tau - 0.25;
  child0.radius_sq = parent.radius_sq * 0.85;
  child1.radius_sq = parent.radius_sq * 0.85;
  child0.aniso_dir_x = axis.x;
  child0.aniso_dir_y = axis.y;
  child1.aniso_dir_x = axis.x;
  child1.aniso_dir_y = axis.y;
  child0.log_aniso = logAniso;
  child1.log_aniso = logAniso;

  sites[parentIdx] = child0;
  sites[childIdx] = child1;

  let base0 = parentIdx * 24u;
  let base1 = childIdx * 24u;
  for (var i = 0u; i < 24u; i = i + 1u) {
    adam[base0 + i] = 0.0;
    adam[base1 + i] = 0.0;
  }
}
