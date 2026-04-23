// WebGPU-JS WGSL overrides for kernels whose original binding count exceeds
// the WebGPU per-shader-stage storage-buffer limit (10 on common adapters).
// The consolidated gradient buffer uses interleaved layout `grads[idx*10+ch]`:
//   0 pos_x, 1 pos_y, 2 log_tau, 3 radius_sq,
//   4 color_r, 5 color_g, 6 color_b,
//   7 dir_x, 8 dir_y, 9 log_aniso

export const GRAD_CHANNELS = 10;

// gradients_tiled: workgroup-tiled atomic accumulation (open-addressed hash
// per workgroup, 256 slots) that flushes to the consolidated `grads` buffer
// once at the end. This is the speed-critical path — direct global atomics per
// pixel was ~10× slower than Metal. To satisfy Dawn's strict uniformity
// analysis (barriers must be reachable from uniform control flow), the kernel
// is structured so NO thread takes an early `return` before either barrier:
// every skip is expressed as an `if (guard) { work }` with no return inside.
export const GRADIENTS_TILED_OVERRIDE = `
@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : GradParams;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var targetTex : texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> grads : array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> removal_delta : array<atomic<u32>>;
@group(0) @binding(7) var maskTex : texture_2d<f32>;

fn atomicAddFloatRemoval(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&removal_delta[idx]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&removal_delta[idx], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

const K_TILE_HASH_SIZE : u32 = 256u;
const K_TILE_HASH_MASK : u32 = 255u;
const K_TILE_MAX_PROBES : u32 = 8u;
const K_TILE_EMPTY_KEY : u32 = 0xffffffffu;

fn tile_hash(key : u32) -> u32 { return key * 2654435761u; }

var<workgroup> tg_keys : array<atomic<u32>, 256>;
// Ten grad channels laid out interleaved: tg_grads[slot * 10 + channel].
var<workgroup> tg_grads : array<atomic<i32>, 2560>;
var<workgroup> tg_delta : array<atomic<u32>, 256>;

fn wgAddGrad(slot : u32, ch : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grads[slot * 10u + ch], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddDelta(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  var old = atomicLoad(&tg_delta[slot]);
  loop {
    let new_f = bitcast<f32>(old) + val;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(&tg_delta[slot], old, new_u);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) tid : vec3<u32>) {
  let local_idx = tid.y * 16u + tid.x;
  let local_threads = 256u;

  // Init workgroup memory — uniform loop over local_idx.
  for (var i = local_idx; i < K_TILE_HASH_SIZE; i = i + local_threads) {
    atomicStore(&tg_keys[i], K_TILE_EMPTY_KEY);
    atomicStore(&tg_delta[i], 0u);
  }
  for (var i = local_idx; i < 2560u; i = i + local_threads) {
    atomicStore(&tg_grads[i], 0);
  }
  workgroupBarrier();

  // Per-pixel gradient contribution — every thread takes the same path to the
  // next barrier. All guards are plain if-blocks; no early returns inside.
  let dims = textureDimensions(targetTex);
  let in_bounds = gid.x < dims.x && gid.y < dims.y;
  var mask_val : f32 = 0.0;
  if (in_bounds) {
    mask_val = textureLoad(maskTex, vec2<i32>(gid.xy), 0).r;
  }
  let should_work = in_bounds && mask_val > 0.0;

  if (should_work) {
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
      var lv : f32 = neg_inf;
      if (idx < params.siteCount) {
        let site = sites[idx];
        if (site.position.x >= 0.0 && !is_bad(site.position.x) && !is_bad(site.position.y) &&
            !is_bad(site.log_tau) && !is_bad(site.log_aniso)) {
          let tau = exp(site.log_tau);
          if (!is_bad(tau)) {
            let dmix2 = voronoi_dmix2_fast(site, uv, params.invScaleSq, inv_scale);
            if (!is_bad(dmix2)) {
              let l = -tau * dmix2;
              if (!is_bad(l)) { lv = l; }
            }
          }
        }
      }
      logits[i] = lv;
      max_logit = max(max_logit, lv);
    }

    let have_max = (max_logit != neg_inf) && !is_bad(max_logit);
    if (have_max) {
      var weights : array<f32, 8>;
      var sum_w : f32 = 0.0;
      for (var i = 0u; i < 8u; i = i + 1u) {
        var w = exp(logits[i] - max_logit);
        if (is_bad(w)) { w = 0.0; }
        weights[i] = w;
        sum_w = sum_w + w;
      }
      let sum_ok = !is_bad(sum_w) && sum_w > 0.0;
      if (sum_ok) {
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
        let dL_dpred = 2.0 * (pred - tgt);
        let dL_ok = !is_bad3(dL_dpred);
        if (dL_ok) {
          if (params.computeRemoval != 0u) {
            var uniqueSites : array<u32, 8>;
            var uniqueWeights : array<f32, 8>;
            var numUnique = 0u;
            for (var i = 0u; i < 8u; i = i + 1u) {
              let idx = candIds[i];
              if (idx < params.siteCount && sites[idx].position.x >= 0.0) {
                let w = weights[i];
                var found = false;
                for (var j = 0u; j < numUnique; j = j + 1u) {
                  if (uniqueSites[j] == idx) { uniqueWeights[j] = uniqueWeights[j] + w; found = true; break; }
                }
                if (!found && numUnique < 8u) {
                  uniqueSites[numUnique] = idx;
                  uniqueWeights[numUnique] = w;
                  numUnique = numUnique + 1u;
                }
              }
            }
            for (var i = 0u; i < numUnique; i = i + 1u) {
              let idx = uniqueSites[i];
              let w_total = uniqueWeights[i];
              if (w_total > 0.0) {
                let delta = removalDeltaForSite(pred, tgt, site_color(sites[idx]), w_total);
                var inserted = false;
                let base = tile_hash(idx) & K_TILE_HASH_MASK;
                for (var probe = 0u; probe < K_TILE_MAX_PROBES; probe = probe + 1u) {
                  let slot = (base + probe) & K_TILE_HASH_MASK;
                  var expected = K_TILE_EMPTY_KEY;
                  let res = atomicCompareExchangeWeak(&tg_keys[slot], expected, idx);
                  if (res.exchanged || res.old_value == idx) { wgAddDelta(slot, delta); inserted = true; break; }
                }
                if (!inserted) { atomicAddFloatRemoval(idx, delta); }
              }
            }
          }

          for (var i = 0u; i < 8u; i = i + 1u) {
            let idx = candIds[i];
            if (idx < params.siteCount) {
              let site = sites[idx];
              if (site.position.x >= 0.0) {
                let w = weights[i];
                let dL_dcolor = w * dL_dpred;
                let site_col = site_color(site);
                let dL_dlogit = dot(dL_dpred, w * (site_col - pred));

                let tau = exp(site.log_tau);
                if (!is_bad(tau)) {
                  let diff = uv - site.position;
                  let diff2 = dot(diff, diff);
                  let dir = site_aniso_dir(site);
                  let proj = dot(dir, diff);
                  let proj2 = proj * proj;
                  let perp2 = max(diff2 - proj2, 0.0);
                  let l1 = exp(site.log_aniso);
                  let l2 = 1.0 / l1;

                  let d2_norm = (l1 * proj2 + l2 * perp2) * params.invScaleSq;
                  let d2_safe = max(d2_norm, 1.0e-8);
                  let inv_sqrt_d2 = inverseSqrt(d2_safe);
                  let r_norm = site.radius_sq * inv_scale;
                  let dmix2 = (sqrt(d2_safe) - r_norm);
                  if (!is_bad(dmix2)) {
                    let g_diff = l2 * diff + (l1 - l2) * dir * proj;
                    let dL_dpos = dL_dlogit * (tau * params.invScaleSq * inv_sqrt_d2) * g_diff;
                    let dL_dlog_tau = dL_dlogit * (-dmix2) * tau;
                    let dL_dradius_sq = dL_dlogit * tau * inv_scale;
                    let d2_dlog_aniso = (l1 * proj2 - l2 * perp2) * params.invScaleSq;
                    let dL_dlog_aniso = dL_dlogit * (-tau) * (0.5 * inv_sqrt_d2) * d2_dlog_aniso;
                    let d2_ddir = (2.0 * (l1 - l2) * proj * params.invScaleSq) * diff;
                    let dL_ddir = dL_dlogit * (-tau) * (0.5 * inv_sqrt_d2) * d2_ddir;

                    var inserted = false;
                    let base = tile_hash(idx) & K_TILE_HASH_MASK;
                    for (var probe = 0u; probe < K_TILE_MAX_PROBES; probe = probe + 1u) {
                      let slot = (base + probe) & K_TILE_HASH_MASK;
                      var expected = K_TILE_EMPTY_KEY;
                      let res = atomicCompareExchangeWeak(&tg_keys[slot], expected, idx);
                      if (res.exchanged || res.old_value == idx) {
                        wgAddGrad(slot, 0u, dL_dpos.x);
                        wgAddGrad(slot, 1u, dL_dpos.y);
                        wgAddGrad(slot, 2u, dL_dlog_tau);
                        wgAddGrad(slot, 3u, dL_dradius_sq);
                        wgAddGrad(slot, 4u, dL_dcolor.x);
                        wgAddGrad(slot, 5u, dL_dcolor.y);
                        wgAddGrad(slot, 6u, dL_dcolor.z);
                        wgAddGrad(slot, 7u, dL_ddir.x);
                        wgAddGrad(slot, 8u, dL_ddir.y);
                        wgAddGrad(slot, 9u, dL_dlog_aniso);
                        inserted = true; break;
                      }
                    }
                    if (!inserted) {
                      atomicAdd(&grads[idx * 10u + 0u], i32(dL_dpos.x * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 1u], i32(dL_dpos.y * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 2u], i32(dL_dlog_tau * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 3u], i32(dL_dradius_sq * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 4u], i32(dL_dcolor.x * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 5u], i32(dL_dcolor.y * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 6u], i32(dL_dcolor.z * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 7u], i32(dL_ddir.x * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 8u], i32(dL_ddir.y * K_GRAD_QUANT_SCALE));
                      atomicAdd(&grads[idx * 10u + 9u], i32(dL_dlog_aniso * K_GRAD_QUANT_SCALE));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  workgroupBarrier();

  // Flush workgroup-local accumulators to global. Uniform loop over slot.
  for (var i = local_idx; i < K_TILE_HASH_SIZE; i = i + local_threads) {
    let siteID = atomicLoad(&tg_keys[i]);
    if (siteID != K_TILE_EMPTY_KEY) {
      for (var ch = 0u; ch < 10u; ch = ch + 1u) {
        let g = atomicLoad(&tg_grads[i * 10u + ch]);
        if (g != 0) { atomicAdd(&grads[siteID * 10u + ch], g); }
      }
      if (params.computeRemoval != 0u) {
        let d = bitcast<f32>(atomicLoad(&tg_delta[i]));
        if (d != 0.0 && !is_bad(d)) { atomicAddFloatRemoval(siteID, d); }
      }
    }
  }
}
`;

export const ADAM_OVERRIDE = `
@group(0) @binding(0) var<storage, read_write> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> adam : array<f32>;
@group(0) @binding(2) var<storage, read_write> grads : array<atomic<i32>>;
@group(0) @binding(3) var<uniform> params : AdamParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let base_g = idx * 10u;
  let norm = 1.0 / f32(params.width * params.height);
  let scale = K_GRAD_QUANT_SCALE_INV * norm;
  let g_pos = vec2<f32>(f32(atomicLoad(&grads[base_g + 0u])),
                        f32(atomicLoad(&grads[base_g + 1u]))) * scale;
  let g_log_tau = f32(atomicLoad(&grads[base_g + 2u])) * scale;
  let g_radius_sq = f32(atomicLoad(&grads[base_g + 3u])) * scale;
  let g_color = vec3<f32>(f32(atomicLoad(&grads[base_g + 4u])),
                          f32(atomicLoad(&grads[base_g + 5u])),
                          f32(atomicLoad(&grads[base_g + 6u]))) * scale;
  let g_dir = vec2<f32>(f32(atomicLoad(&grads[base_g + 7u])),
                        f32(atomicLoad(&grads[base_g + 8u]))) * scale;
  var g_log_aniso = f32(atomicLoad(&grads[base_g + 9u])) * scale;
  var safe_g_dir = g_dir;
  if (is_bad2(safe_g_dir)) { safe_g_dir = vec2<f32>(0.0, 0.0); }
  if (is_bad(g_log_aniso)) { g_log_aniso = 0.0; }

  if (sites[idx].position.x < 0.0) { return; }
  if (is_bad(sites[idx].position.x) || is_bad(sites[idx].position.y) || is_bad(sites[idx].log_tau) ||
      is_bad(sites[idx].radius_sq) || is_bad3(site_color(sites[idx])) || is_bad(sites[idx].log_aniso) ||
      is_bad(g_pos.x) || is_bad(g_pos.y) || is_bad(g_log_tau) || is_bad(g_radius_sq) ||
      is_bad3(g_color) || is_bad(safe_g_dir.x) || is_bad(safe_g_dir.y) || is_bad(g_log_aniso)) {
    for (var c = 0u; c < 10u; c = c + 1u) { atomicStore(&grads[base_g + c], 0); }
    return;
  }

  let base = idx * 24u;
  var m_pos = vec2<f32>(adam[base + 0u], adam[base + 1u]);
  var v_pos = vec2<f32>(adam[base + 2u], adam[base + 3u]);
  var m_log_tau = adam[base + 4u];
  var v_log_tau = adam[base + 5u];
  var m_radius_sq = adam[base + 6u];
  var v_radius_sq = adam[base + 7u];
  var m_color = vec3<f32>(adam[base + 8u], adam[base + 9u], adam[base + 10u]);
  var v_color = vec3<f32>(adam[base + 11u], adam[base + 12u], adam[base + 13u]);
  var m_dir = vec2<f32>(adam[base + 14u], adam[base + 15u]);
  var v_dir = vec2<f32>(adam[base + 16u], adam[base + 17u]);
  var m_log_aniso = adam[base + 18u];
  var v_log_aniso = adam[base + 19u];

  let tt = f32(params.t);
  let b1t_corr = 1.0 / (1.0 - pow(params.beta1, tt));
  let b2t_corr = 1.0 / (1.0 - pow(params.beta2, tt));

  m_pos = params.beta1 * m_pos + (1.0 - params.beta1) * g_pos;
  let g2_pos = dot(g_pos, g_pos);
  let v_pos_scalar = params.beta2 * v_pos.x + (1.0 - params.beta2) * g2_pos;
  let m_hat_pos = m_pos * b1t_corr;
  let v_hat_pos = v_pos_scalar * b2t_corr;
  let step_pos = params.lr_pos * m_hat_pos / (sqrt(v_hat_pos) + params.eps);
  sites[idx].position = clamp(sites[idx].position - step_pos,
    vec2<f32>(0.0), vec2<f32>(f32(params.width - 1u), f32(params.height - 1u)));
  v_pos = vec2<f32>(v_pos_scalar, v_pos_scalar);

  m_log_tau = params.beta1 * m_log_tau + (1.0 - params.beta1) * g_log_tau;
  v_log_tau = params.beta2 * v_log_tau + (1.0 - params.beta2) * (g_log_tau * g_log_tau);
  let m_hat_tau = m_log_tau * b1t_corr;
  let v_hat_tau = v_log_tau * b2t_corr;
  let step_tau = params.lr_tau * m_hat_tau / (sqrt(v_hat_tau) + params.eps);
  sites[idx].log_tau = sites[idx].log_tau - step_tau;

  m_radius_sq = params.beta1 * m_radius_sq + (1.0 - params.beta1) * g_radius_sq;
  v_radius_sq = params.beta2 * v_radius_sq + (1.0 - params.beta2) * (g_radius_sq * g_radius_sq);
  let m_hat_rad = m_radius_sq * b1t_corr;
  let v_hat_rad = v_radius_sq * b2t_corr;
  let step_rad = params.lr_radius * m_hat_rad / (sqrt(v_hat_rad) + params.eps);
  sites[idx].radius_sq = sites[idx].radius_sq - step_rad;

  m_color = params.beta1 * m_color + (1.0 - params.beta1) * g_color;
  v_color = params.beta2 * v_color + (1.0 - params.beta2) * (g_color * g_color);
  let m_hat_col = m_color * b1t_corr;
  let v_hat_col = v_color * b2t_corr;
  let step_col = params.lr_color * m_hat_col / (sqrt(v_hat_col) + params.eps);
  let new_col = site_color(sites[idx]) - step_col;
  sites[idx].color_r = new_col.x;
  sites[idx].color_g = new_col.y;
  sites[idx].color_b = new_col.z;

  let curr_dir = safe_dir(site_aniso_dir(sites[idx]));
  let g_tan = safe_g_dir - curr_dir * dot(safe_g_dir, curr_dir);
  m_dir = params.beta1 * m_dir + (1.0 - params.beta1) * g_tan;
  let g2_dir = dot(g_tan, g_tan);
  let v_dir_scalar = params.beta2 * v_dir.x + (1.0 - params.beta2) * g2_dir;
  let m_hat_dir = m_dir * b1t_corr;
  let v_hat_dir = v_dir_scalar * b2t_corr;
  let step_dir = params.lr_dir * m_hat_dir / (sqrt(v_hat_dir) + params.eps);
  var new_dir = curr_dir - step_dir;
  new_dir = safe_dir(new_dir);
  sites[idx].aniso_dir_x = new_dir.x;
  sites[idx].aniso_dir_y = new_dir.y;
  v_dir = vec2<f32>(v_dir_scalar, v_dir_scalar);

  m_log_aniso = params.beta1 * m_log_aniso + (1.0 - params.beta1) * g_log_aniso;
  v_log_aniso = params.beta2 * v_log_aniso + (1.0 - params.beta2) * (g_log_aniso * g_log_aniso);
  let m_hat_aniso = m_log_aniso * b1t_corr;
  let v_hat_aniso = v_log_aniso * b2t_corr;
  let step_aniso = params.lr_aniso * m_hat_aniso / (sqrt(v_hat_aniso) + params.eps);
  sites[idx].log_aniso = sites[idx].log_aniso - step_aniso;
  sites[idx].log_aniso = clamp(sites[idx].log_aniso, -2.0, 2.0);

  adam[base + 0u] = m_pos.x;
  adam[base + 1u] = m_pos.y;
  adam[base + 2u] = v_pos.x;
  adam[base + 3u] = v_pos.y;
  adam[base + 4u] = m_log_tau;
  adam[base + 5u] = v_log_tau;
  adam[base + 6u] = m_radius_sq;
  adam[base + 7u] = v_radius_sq;
  adam[base + 8u] = m_color.x;
  adam[base + 9u] = m_color.y;
  adam[base + 10u] = m_color.z;
  adam[base + 11u] = v_color.x;
  adam[base + 12u] = v_color.y;
  adam[base + 13u] = v_color.z;
  adam[base + 14u] = m_dir.x;
  adam[base + 15u] = m_dir.y;
  adam[base + 16u] = v_dir.x;
  adam[base + 17u] = v_dir.y;
  adam[base + 18u] = m_log_aniso;
  adam[base + 19u] = v_log_aniso;

  for (var c = 0u; c < 10u; c = c + 1u) { atomicStore(&grads[base_g + c], 0); }
}
`;

export const TAU_EXTRACT_OVERRIDE = `
@group(0) @binding(0) var<storage, read_write> grads : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> grad_out : array<f32>;
@group(0) @binding(2) var<uniform> params : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  grad_out[idx] = f32(atomicLoad(&grads[idx * 10u + 2u])) * K_GRAD_QUANT_SCALE_INV;
}
`;

// Voronoi-diagram visualization: same softmax blending as render_compute, but
// substitutes each candidate site's color with hashColor(siteId). Mirrors the
// Metal `renderVoronoiHashed` kernel (backends/metal/shaders/encoders/render_encoder.metal).
export const RENDER_HASHED_OVERRIDE = `
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

  let candCoord = candidateCoord(vec2<u32>(gid.xy), vec2<u32>(w, h),
    vec2<u32>(params.candWidth, params.candHeight));
  let candXY = vec2<i32>(i32(candCoord.x), i32(candCoord.y));
  let c0 = textureLoad(inCand0, candXY, 0);
  let c1 = textureLoad(inCand1, candXY, 0);
  let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

  let neg_inf = -1e30;
  var logits : array<f32, 8>;
  var maxLogit : f32 = neg_inf;
  for (var i = 0u; i < 8u; i = i + 1u) {
    let id = candIds[i];
    if (id >= params.siteCount) { logits[i] = neg_inf; continue; }
    let site = sites[id];
    if (site.position.x < 0.0 || is_bad(site.position.x) || is_bad(site.position.y) ||
        is_bad(site.log_tau) || is_bad(site.log_aniso)) { logits[i] = neg_inf; continue; }
    let tau = exp(site.log_tau);
    if (is_bad(tau)) { logits[i] = neg_inf; continue; }
    let dmix2 = voronoi_dmix2_fast(site, uv, params.invScaleSq, inv_scale);
    if (is_bad(dmix2)) { logits[i] = neg_inf; continue; }
    logits[i] = -tau * dmix2;
    maxLogit = max(maxLogit, logits[i]);
  }
  if (maxLogit == neg_inf || is_bad(maxLogit)) {
    textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }

  var weights : array<f32, 8>;
  var sumW : f32 = 0.0;
  for (var i = 0u; i < 8u; i = i + 1u) {
    weights[i] = exp(logits[i] - maxLogit);
    if (is_bad(weights[i])) { weights[i] = 0.0; }
    sumW = sumW + weights[i];
  }
  let invSum = 1.0 / max(sumW, 1e-8);

  var color = vec3<f32>(0.0);
  for (var i = 0u; i < 8u; i = i + 1u) {
    let w = weights[i] * invSum;
    let id = candIds[i];
    if (w == w && !is_bad(w) && id < params.siteCount && sites[id].position.x >= 0.0) {
      color = color + w * hashColor(id);
    }
  }
  textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
`;

// Tau heatmap: centroid dots blended over a dark background, colored on a
// blue→white→red scale from min→mean→max tau. Mirrors Metal's
// renderCentroidsTauHeatmap. The candidate textures give us the 8 nearest
// sites per pixel, which is all the overlap we need for dot rendering.
// Layout for params (TauHeatmapParams, 16 bytes):
//   f32 minTau, f32 meanTau, f32 maxTau, f32 dotRadius.
export const RENDER_TAU_HEATMAP_OVERRIDE = `
struct TauHeatmapParams {
  minTau : f32,
  meanTau : f32,
  maxTau : f32,
  dotRadius : f32,
}

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<uniform> heat : TauHeatmapParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let w = params.width;
  let h = params.height;
  if (gid.x >= w || gid.y >= h) { return; }
  let uv = vec2<f32>(f32(gid.x), f32(gid.y));

  let candCoord = candidateCoord(vec2<u32>(gid.xy), vec2<u32>(w, h),
    vec2<u32>(params.candWidth, params.candHeight));
  let candXY = vec2<i32>(i32(candCoord.x), i32(candCoord.y));
  let c0 = textureLoad(inCand0, candXY, 0);
  let c1 = textureLoad(inCand1, candXY, 0);
  let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

  let radius = heat.dotRadius;
  var accColor = vec3<f32>(0.0);
  var accAlpha : f32 = 0.0;

  for (var i = 0u; i < 8u; i = i + 1u) {
    let id = candIds[i];
    if (id >= params.siteCount) { continue; }
    let site = sites[id];
    if (site.position.x < 0.0) { continue; }

    let dist = length(uv - site.position);
    let alpha = 1.0 - smoothstep(radius - 1.0, radius + 1.0, dist);
    if (alpha <= 0.0) { continue; }

    let tau = exp(site.log_tau);
    var color : vec3<f32>;
    if (tau <= heat.meanTau) {
      var t : f32 = 0.5;
      if (heat.meanTau > heat.minTau) { t = (tau - heat.minTau) / (heat.meanTau - heat.minTau); }
      color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), pow(clamp(t, 0.0, 1.0), 0.2));
    } else {
      var t : f32 = 0.5;
      if (heat.maxTau > heat.meanTau) { t = (tau - heat.meanTau) / (heat.maxTau - heat.meanTau); }
      color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), pow(clamp(t, 0.0, 1.0), 0.2));
    }

    let blend = alpha * (1.0 - accAlpha);
    accColor = accColor + color * blend;
    accAlpha = accAlpha + blend;
  }

  let bg = vec3<f32>(0.05, 0.05, 0.05);
  let finalColor = accColor + bg * (1.0 - accAlpha);
  textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(finalColor, 1.0));
}
`;

export const TAU_WRITEBACK_OVERRIDE = `
@group(0) @binding(0) var<storage, read> grad_in : array<f32>;
@group(0) @binding(1) var<storage, read_write> grads : array<atomic<i32>>;
@group(0) @binding(2) var<uniform> params : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  let val = grad_in[idx];
  if (is_bad(val)) {
    atomicStore(&grads[idx * 10u + 2u], 0);
  } else {
    atomicStore(&grads[idx * 10u + 2u], i32(val * K_GRAD_QUANT_SCALE));
  }
}
`;
