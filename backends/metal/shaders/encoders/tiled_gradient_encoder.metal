#include "../sad_common.metal"

kernel void computeGradientsTiled(
    texture2d<uint, access::read>  candidates0 [[texture(0)]],
    texture2d<uint, access::read>  candidates1 [[texture(1)]],
    texture2d<float, access::read> target [[texture(2)]],
    texture2d<float, access::read> rendered [[texture(3)]],
    texture2d<float, access::read> mask [[texture(4)]],
    device atomic_float *grad_pos_x [[buffer(0)]],
    device atomic_float *grad_pos_y [[buffer(1)]],
    device atomic_float *grad_log_tau [[buffer(2)]],
    device atomic_float *grad_radius [[buffer(3)]],
    device atomic_float *grad_color_r [[buffer(4)]],
    device atomic_float *grad_color_g [[buffer(5)]],
    device atomic_float *grad_color_b [[buffer(6)]],
    device atomic_float *grad_dir_x [[buffer(7)]],
    device atomic_float *grad_dir_y [[buffer(8)]],
    device atomic_float *grad_log_aniso [[buffer(9)]],
    constant VoronoiSite *sites [[buffer(10)]],
    constant float &inv_scale_sq [[buffer(11)]],
    constant uint &siteCount [[buffer(12)]],
    device float *removal_delta [[buffer(13)]],
    constant uint &computeRemoval [[buffer(14)]],
    constant float &ssim_weight [[buffer(15)]],
    threadgroup atomic_uint *tg_keys [[threadgroup(0)]],
    threadgroup atomic_int *tg_grad_pos_x [[threadgroup(1)]],
    threadgroup atomic_int *tg_grad_pos_y [[threadgroup(2)]],
    threadgroup atomic_int *tg_grad_log_tau [[threadgroup(3)]],
    threadgroup atomic_int *tg_grad_radius [[threadgroup(4)]],
    threadgroup atomic_int *tg_grad_color_r [[threadgroup(5)]],
    threadgroup atomic_int *tg_grad_color_g [[threadgroup(6)]],
    threadgroup atomic_int *tg_grad_color_b [[threadgroup(7)]],
    threadgroup atomic_int *tg_grad_dir_x [[threadgroup(8)]],
    threadgroup atomic_int *tg_grad_dir_y [[threadgroup(9)]],
    threadgroup atomic_int *tg_grad_log_aniso [[threadgroup(10)]],
    threadgroup atomic_int *tg_delta [[threadgroup(11)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]])
{
    uint local_idx = tid.y * tgs.x + tid.x;
    uint local_threads = tgs.x * tgs.y;

    for (uint i = local_idx; i < kTileHashSize; i += local_threads) {
        atomic_store_explicit(&tg_keys[i], kTileEmptyKey, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_pos_x[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_pos_y[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_log_tau[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_radius[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_color_r[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_color_g[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_color_b[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_dir_x[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_dir_y[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_grad_log_aniso[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_delta[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint width = target.get_width();
    uint height = target.get_height();
    bool in_bounds = (gid.x < width && gid.y < height);

    if (in_bounds) {
        float mask_val = mask.read(gid).r;
        if (mask_val <= 0.0f) {
            return;
        }
        float2 uv = float2(gid);
        uint candIds[8];
        uint2 outSize = uint2(width, height);
        loadCandidateIds(candidates0, candidates1, gid, outSize, candIds);

        float weights[8];
        float3 pred;
        if (computeCandidateWeights(candIds, sites, siteCount, uv, inv_scale_sq, weights, pred)) {

            float3 tgt = target.read(gid).rgb;
            float3 dL_dpred = 2.0f * (pred - tgt);
            if (ssim_weight > 0.0f) {
                float mu_x, mu_y, ex2, ey2, exy;
                ssim_window_stats(rendered, target, gid, mu_x, mu_y, ex2, ey2, exy);

                float sigma_x2 = max(ex2 - mu_x * mu_x, 0.0f);
                float sigma_y2 = max(ey2 - mu_y * mu_y, 0.0f);
                float sigma_xy = exy - mu_x * mu_y;

                float A = 2.0f * mu_x * mu_y + kSSIMC1;
                float B = 2.0f * sigma_xy + kSSIMC2;
                float C = mu_x * mu_x + mu_y * mu_y + kSSIMC1;
                float D = sigma_x2 + sigma_y2 + kSSIMC2;

                float denom = max(C * D, 1e-8f);
                float num = A * B;

                float xk = dot(rendered.read(gid).rgb, kLumaWeights);
                float yk = dot(target.read(gid).rgb, kLumaWeights);
                float dmu_x = kSSIMInvN;
                float dsig_x2 = (2.0f * kSSIMInvN) * (xk - mu_x);
                float dcov = kSSIMInvN * (yk - mu_y);

                float dA = 2.0f * mu_y * dmu_x;
                float dB = 2.0f * dcov;
                float dC = 2.0f * mu_x * dmu_x;
                float dD = dsig_x2;

                float dSSIM = (dA * B + A * dB) / denom
                    - num * (dC * D + C * dD) / (denom * denom);
                dL_dpred += (-ssim_weight * dSSIM) * kLumaWeights;
            }

            if (computeRemoval != 0) {
                uint uniqueSites[8];
                float uniqueWeights[8];
                uint numUnique = accumulateUniqueSites(candIds, weights, sites, siteCount,
                                                       uniqueSites, uniqueWeights);

                // Compute removal delta for each unique site with accumulated weight
                for (uint i = 0; i < numUnique; ++i) {
                    uint idx = uniqueSites[i];
                    float w_total = uniqueWeights[i];
                    if (w_total > 0.0f) {
                        float delta = removalDeltaForSite(pred, tgt, sites[idx].color, w_total);
                        bool inserted = false;
                        uint base = tile_hash(idx) & kTileHashMask;

                        for (uint probe = 0; probe < kTileMaxProbes; ++probe) {
                            uint slot = (base + probe) & kTileHashMask;
                            uint expected = kTileEmptyKey;
                            if (atomic_compare_exchange_weak_explicit(&tg_keys[slot], &expected, idx,
                                                                      memory_order_relaxed, memory_order_relaxed) ||
                                expected == idx) {
                                atomic_fetch_add_tg_float(&tg_delta[slot], delta);
                                inserted = true;
                                break;
                            }
                        }

                        if (!inserted) {
                            atomic_fetch_add_explicit_float(&removal_delta[idx], delta);
                        }
                    }
                }
            }

            for (uint i = 0; i < 8; ++i) {
                uint idx = candIds[i];
                if (idx >= siteCount) continue;
                VoronoiSite site = sites[idx];
                if (site.position.x < 0.0f) continue;

                float w = weights[i];
                float3 dL_dcolor = w * dL_dpred;
                float dL_dlogit = dot(dL_dpred, w * (site.color - pred));

                float tau = exp(site.log_tau);
                float2 diff = uv - site.position;

                float diff2 = dot(diff, diff);
                float proj = dot(site.aniso_dir, diff);
                float proj2 = proj * proj;
                float perp2 = max(diff2 - proj2, 0.0f);
                float l1 = exp(site.log_aniso);
                float l2 = 1.0f / l1;

                float d2_norm = (l1 * proj2 + l2 * perp2) * inv_scale_sq;
                float d2_safe = max(d2_norm, 1e-8f);
                float inv_sqrt_d2 = rsqrt(d2_safe);
                float inv_scale = sqrt(inv_scale_sq);
                float r_norm = site.radius * inv_scale;
                float dmix2 = (sqrt(d2_safe) - r_norm);

                float2 g_diff = l2 * diff + (l1 - l2) * site.aniso_dir * proj;
                float2 dL_dpos = dL_dlogit * (tau * inv_scale_sq * inv_sqrt_d2) * g_diff;

                float dL_dlog_tau = dL_dlogit * (-dmix2) * tau;
                float dL_dradius = dL_dlogit * tau * inv_scale;

                float d2_dlog_aniso = (l1 * proj2 - l2 * perp2) * inv_scale_sq;
                float dL_dlog_aniso = dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_dlog_aniso;

                float2 d2_ddir = (2.0f * (l1 - l2) * proj * inv_scale_sq) * diff;
                float2 dL_ddir = dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_ddir;

                bool inserted = false;
                uint base = tile_hash(idx) & kTileHashMask;

                for (uint probe = 0; probe < kTileMaxProbes; ++probe) {
                    uint slot = (base + probe) & kTileHashMask;
                    uint expected = kTileEmptyKey;
                    if (atomic_compare_exchange_weak_explicit(&tg_keys[slot], &expected, idx,
                                                              memory_order_relaxed, memory_order_relaxed) ||
                        expected == idx) {
                        atomic_fetch_add_float(&tg_grad_pos_x[slot], dL_dpos.x);
                        atomic_fetch_add_float(&tg_grad_pos_y[slot], dL_dpos.y);
                        atomic_fetch_add_float(&tg_grad_log_tau[slot], dL_dlog_tau);
                        atomic_fetch_add_float(&tg_grad_radius[slot], dL_dradius);
                        atomic_fetch_add_float(&tg_grad_color_r[slot], dL_dcolor.r);
                        atomic_fetch_add_float(&tg_grad_color_g[slot], dL_dcolor.g);
                        atomic_fetch_add_float(&tg_grad_color_b[slot], dL_dcolor.b);
                        atomic_fetch_add_float(&tg_grad_dir_x[slot], dL_ddir.x);
                        atomic_fetch_add_float(&tg_grad_dir_y[slot], dL_ddir.y);
                        atomic_fetch_add_float(&tg_grad_log_aniso[slot], dL_dlog_aniso);
                        inserted = true;
                        break;
                    }
                }

                if (!inserted) {
                    atomic_fetch_add_explicit(&grad_pos_x[idx], dL_dpos.x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_pos_y[idx], dL_dpos.y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_log_tau[idx], dL_dlog_tau, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_radius[idx], dL_dradius, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_color_r[idx], dL_dcolor.r, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_color_g[idx], dL_dcolor.g, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_color_b[idx], dL_dcolor.b, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_dir_x[idx], dL_ddir.x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_dir_y[idx], dL_ddir.y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grad_log_aniso[idx], dL_dlog_aniso, memory_order_relaxed);
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = local_idx; i < kTileHashSize; i += local_threads) {
        uint siteID = atomic_load_explicit(&tg_keys[i], memory_order_relaxed);
        if (siteID == kTileEmptyKey) continue;

        float g_pos_x = atomic_load_float(&tg_grad_pos_x[i]);
        float g_pos_y = atomic_load_float(&tg_grad_pos_y[i]);
        float g_log_tau = atomic_load_float(&tg_grad_log_tau[i]);
        float g_radius = atomic_load_float(&tg_grad_radius[i]);
        float g_color_r = atomic_load_float(&tg_grad_color_r[i]);
        float g_color_g = atomic_load_float(&tg_grad_color_g[i]);
        float g_color_b = atomic_load_float(&tg_grad_color_b[i]);
        float g_dir_x = atomic_load_float(&tg_grad_dir_x[i]);
        float g_dir_y = atomic_load_float(&tg_grad_dir_y[i]);
        float g_log_aniso = atomic_load_float(&tg_grad_log_aniso[i]);
        float delta = 0.0f;
        if (computeRemoval != 0) {
            delta = atomic_load_tg_float(&tg_delta[i]);
        }

        atomic_fetch_add_explicit(&grad_pos_x[siteID], g_pos_x, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_pos_y[siteID], g_pos_y, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_log_tau[siteID], g_log_tau, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_radius[siteID], g_radius, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_color_r[siteID], g_color_r, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_color_g[siteID], g_color_g, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_color_b[siteID], g_color_b, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_dir_x[siteID], g_dir_x, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_dir_y[siteID], g_dir_y, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_log_aniso[siteID], g_log_aniso, memory_order_relaxed);
        if (computeRemoval != 0 && delta != 0.0f) {
            atomic_fetch_add_explicit_float(&removal_delta[siteID], delta);
        }
    }
}
