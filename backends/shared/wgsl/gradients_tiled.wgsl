@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : GradParams;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var targetTex : texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> grad_pos_x : array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> grad_pos_y : array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> grad_log_tau : array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> grad_radius_sq : array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> grad_color_r : array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> grad_color_g : array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> grad_color_b : array<atomic<i32>>;
@group(0) @binding(12) var<storage, read_write> grad_dir_x : array<atomic<i32>>;
@group(0) @binding(13) var<storage, read_write> grad_dir_y : array<atomic<i32>>;
@group(0) @binding(14) var<storage, read_write> grad_log_aniso : array<atomic<i32>>;
@group(0) @binding(15) var<storage, read_write> removal_delta : array<atomic<u32>>;
@group(0) @binding(16) var maskTex : texture_2d<f32>;

fn atomicAddGradPosX(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_pos_x[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradPosY(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_pos_y[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradLogTau(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_log_tau[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradRadius(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_radius_sq[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradColorR(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_color_r[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradColorG(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_color_g[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradColorB(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_color_b[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradDirX(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_dir_x[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradDirY(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_dir_y[idx], i32(val * K_GRAD_QUANT_SCALE));
}

fn atomicAddGradLogAniso(idx : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&grad_log_aniso[idx], i32(val * K_GRAD_QUANT_SCALE));
}

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

fn tile_hash(key : u32) -> u32 {
  return key * 2654435761u;
}

var<workgroup> tg_keys : array<atomic<u32>, 256>;
var<workgroup> tg_grad_pos_x : array<atomic<i32>, 256>;
var<workgroup> tg_grad_pos_y : array<atomic<i32>, 256>;
var<workgroup> tg_grad_log_tau : array<atomic<i32>, 256>;
var<workgroup> tg_grad_radius_sq : array<atomic<i32>, 256>;
var<workgroup> tg_grad_color_r : array<atomic<i32>, 256>;
var<workgroup> tg_grad_color_g : array<atomic<i32>, 256>;
var<workgroup> tg_grad_color_b : array<atomic<i32>, 256>;
var<workgroup> tg_grad_dir_x : array<atomic<i32>, 256>;
var<workgroup> tg_grad_dir_y : array<atomic<i32>, 256>;
var<workgroup> tg_grad_log_aniso : array<atomic<i32>, 256>;
var<workgroup> tg_delta : array<atomic<u32>, 256>;

fn wgAddGradPosX(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_pos_x[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradPosY(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_pos_y[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradLogTau(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_log_tau[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradRadius(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_radius_sq[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradColorR(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_color_r[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradColorG(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_color_g[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradColorB(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_color_b[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradDirX(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_dir_x[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradDirY(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_dir_y[slot], i32(val * K_GRAD_QUANT_SCALE));
}

fn wgAddGradLogAniso(slot : u32, val : f32) {
  if (val == 0.0 || is_bad(val)) { return; }
  atomicAdd(&tg_grad_log_aniso[slot], i32(val * K_GRAD_QUANT_SCALE));
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

fn wgLoadDelta(slot : u32) -> f32 {
  return bitcast<f32>(atomicLoad(&tg_delta[slot]));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) tid : vec3<u32>) {
  let local_idx = tid.y * 16u + tid.x;
  let local_threads = 256u;

  for (var i = local_idx; i < K_TILE_HASH_SIZE; i = i + local_threads) {
    atomicStore(&tg_keys[i], K_TILE_EMPTY_KEY);
    atomicStore(&tg_grad_pos_x[i], 0);
    atomicStore(&tg_grad_pos_y[i], 0);
    atomicStore(&tg_grad_log_tau[i], 0);
    atomicStore(&tg_grad_radius_sq[i], 0);
    atomicStore(&tg_grad_color_r[i], 0);
    atomicStore(&tg_grad_color_g[i], 0);
    atomicStore(&tg_grad_color_b[i], 0);
    atomicStore(&tg_grad_dir_x[i], 0);
    atomicStore(&tg_grad_dir_y[i], 0);
    atomicStore(&tg_grad_log_aniso[i], 0);
    atomicStore(&tg_delta[i], 0u);
  }
  workgroupBarrier();

  let dims = textureDimensions(targetTex);
  if (gid.x < dims.x && gid.y < dims.y) {
    let mask_val = textureLoad(maskTex, vec2<i32>(gid.xy), 0).r;
    if (mask_val <= 0.0) { return; }
    let uv = vec2<f32>(f32(gid.x), f32(gid.y));
    let inv_scale = sqrt(params.invScaleSq);
    let candSize = textureDimensions(inCand0);
    let candCoord = candidateCoord(vec2<u32>(gid.xy), dims, candSize);
    let c0 = textureLoad(inCand0, vec2<i32>(candCoord), 0);
    let c1 = textureLoad(inCand1, vec2<i32>(candCoord), 0);
    let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

    let neg_inf = bitcast<f32>(0xff800000u);
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

  if (max_logit != neg_inf && !is_bad(max_logit)) {
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
      let dL_dpred = 2.0 * (pred - tgt);
      if (is_bad3(dL_dpred)) { return; }

      if (params.computeRemoval != 0u) {
        var uniqueSites : array<u32, 8>;
        var uniqueWeights : array<f32, 8>;
        var numUnique = 0u;

        for (var i = 0u; i < 8u; i = i + 1u) {
          let idx = candIds[i];
          if (idx >= params.siteCount) { continue; }
          if (sites[idx].position.x < 0.0) { continue; }
          let w = weights[i];

          var found = false;
          for (var j = 0u; j < numUnique; j = j + 1u) {
            if (uniqueSites[j] == idx) {
              uniqueWeights[j] = uniqueWeights[j] + w;
              found = true;
              break;
            }
          }

          if (!found && numUnique < 8u) {
            uniqueSites[numUnique] = idx;
            uniqueWeights[numUnique] = w;
            numUnique = numUnique + 1u;
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
              if (res.exchanged || res.old_value == idx) {
                wgAddDelta(slot, delta);
                inserted = true;
                break;
              }
            }
            if (!inserted) {
              atomicAddFloatRemoval(idx, delta);
            }
          }
        }
      }

      for (var i = 0u; i < 8u; i = i + 1u) {
        let idx = candIds[i];
        if (idx >= params.siteCount) { continue; }
        let site = sites[idx];
        if (site.position.x < 0.0) { continue; }

        let w = weights[i];
        let dL_dcolor = w * dL_dpred;
        let site_col = site_color(site);
        let dL_dlogit = dot(dL_dpred, w * (site_col - pred));

        let tau = exp(site.log_tau);
        if (is_bad(tau)) { continue; }
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
        let inv_scale = sqrt(params.invScaleSq);
        let r_norm = site.radius_sq * inv_scale;
        let dmix2 = (sqrt(d2_safe) - r_norm);
        if (is_bad(dmix2)) { continue; }

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
            wgAddGradPosX(slot, dL_dpos.x);
            wgAddGradPosY(slot, dL_dpos.y);
            wgAddGradLogTau(slot, dL_dlog_tau);
            wgAddGradRadius(slot, dL_dradius_sq);
            wgAddGradColorR(slot, dL_dcolor.x);
            wgAddGradColorG(slot, dL_dcolor.y);
            wgAddGradColorB(slot, dL_dcolor.z);
            wgAddGradDirX(slot, dL_ddir.x);
            wgAddGradDirY(slot, dL_ddir.y);
            wgAddGradLogAniso(slot, dL_dlog_aniso);
            inserted = true;
            break;
          }
        }

        if (!inserted) {
          atomicAddGradPosX(idx, dL_dpos.x);
          atomicAddGradPosY(idx, dL_dpos.y);
          atomicAddGradLogTau(idx, dL_dlog_tau);
          atomicAddGradRadius(idx, dL_dradius_sq);
          atomicAddGradColorR(idx, dL_dcolor.x);
          atomicAddGradColorG(idx, dL_dcolor.y);
          atomicAddGradColorB(idx, dL_dcolor.z);
          atomicAddGradDirX(idx, dL_ddir.x);
          atomicAddGradDirY(idx, dL_ddir.y);
          atomicAddGradLogAniso(idx, dL_dlog_aniso);
        }
      }
    }
  }

  workgroupBarrier();

  for (var i = local_idx; i < K_TILE_HASH_SIZE; i = i + local_threads) {
    let siteID = atomicLoad(&tg_keys[i]);
    if (siteID == K_TILE_EMPTY_KEY) { continue; }

    let g_pos_x = atomicLoad(&tg_grad_pos_x[i]);
    let g_pos_y = atomicLoad(&tg_grad_pos_y[i]);
    let g_log_tau = atomicLoad(&tg_grad_log_tau[i]);
    let g_radius_sq = atomicLoad(&tg_grad_radius_sq[i]);
    let g_color_r = atomicLoad(&tg_grad_color_r[i]);
    let g_color_g = atomicLoad(&tg_grad_color_g[i]);
    let g_color_b = atomicLoad(&tg_grad_color_b[i]);
    let g_dir_x = atomicLoad(&tg_grad_dir_x[i]);
    let g_dir_y = atomicLoad(&tg_grad_dir_y[i]);
    let g_log_aniso = atomicLoad(&tg_grad_log_aniso[i]);
    let g_delta = wgLoadDelta(i);

    atomicAdd(&grad_pos_x[siteID], g_pos_x);
    atomicAdd(&grad_pos_y[siteID], g_pos_y);
    atomicAdd(&grad_log_tau[siteID], g_log_tau);
    atomicAdd(&grad_radius_sq[siteID], g_radius_sq);
    atomicAdd(&grad_color_r[siteID], g_color_r);
    atomicAdd(&grad_color_g[siteID], g_color_g);
    atomicAdd(&grad_color_b[siteID], g_color_b);
    atomicAdd(&grad_dir_x[siteID], g_dir_x);
    atomicAdd(&grad_dir_y[siteID], g_dir_y);
    atomicAdd(&grad_log_aniso[siteID], g_log_aniso);
    if (params.computeRemoval != 0u && g_delta != 0.0) {
      atomicAddFloatRemoval(siteID, g_delta);
    }
  }
}
