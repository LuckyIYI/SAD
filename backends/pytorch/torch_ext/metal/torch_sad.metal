#include <metal_stdlib>
using namespace metal;

#include "sad_common.metal"

inline VoronoiSite load_site(device const float *sites, uint idx) {
    uint base = idx * 10u;
    VoronoiSite site;
    site.position = float2(sites[base + 0], sites[base + 1]);
    site.log_tau = sites[base + 2];
    site.radius = sites[base + 3];
    site.color = float3(sites[base + 4], sites[base + 5], sites[base + 6]);
    site.aniso_dir = float2(sites[base + 7], sites[base + 8]);
    site.log_aniso = sites[base + 9];
    return site;
}

inline bool computeCandidateWeightsRaw(
    thread uint candIds[8],
    device const float *sitesRaw,
    uint siteCount,
    float2 uv,
    float inv_scale_sq,
    thread float weights[8],
    thread float3 &pred)
{
    float logits[8];
    float max_logit = -INFINITY;

    for (uint i = 0; i < 8; ++i) {
        uint idx = candIds[i];
        if (idx >= siteCount) {
            logits[i] = -INFINITY;
            continue;
        }
        VoronoiSite site = load_site(sitesRaw, idx);
        if (site.position.x < 0.0f) {
            logits[i] = -INFINITY;
            continue;
        }
        float tau = exp(site.log_tau);
        float dmix2 = voronoi_dmix2(site, uv, inv_scale_sq);
        logits[i] = -tau * dmix2;
        max_logit = max(max_logit, logits[i]);
    }

    if (max_logit == -INFINITY) {
        return false;
    }

    float sum_w = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        weights[i] = exp(logits[i] - max_logit);
        sum_w += weights[i];
    }

    float inv_sum = 1.0f / max(sum_w, 1e-8f);
    pred = float3(0.0f);
    for (uint i = 0; i < 8; ++i) {
        weights[i] *= inv_sum;
        uint idx = candIds[i];
        if (idx < siteCount) {
            VoronoiSite site = load_site(sitesRaw, idx);
            if (site.position.x >= 0.0f) {
                pred += weights[i] * site.color;
            }
        }
    }

    return true;
}

kernel void packCandidateSitesRaw(
    device const float *sitesRaw [[buffer(0)]],
    device PackedCandidateSite *packed [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= siteCount) return;
    VoronoiSite site = load_site(sitesRaw, gid);

    PackedCandidateSite out;
    if (site.position.x < 0.0f) {
        out.a = half4(half(-1.0f), half(-1.0f), half(0.0f), half(0.0f));
        out.b = half4(half(0.0f), half(0.0f), half(0.0f), half(0.0f));
    } else {
        out.a = half4(half(site.position.x), half(site.position.y),
                      half(site.log_tau), half(site.radius));
        out.b = half4(half(site.aniso_dir.x), half(site.aniso_dir.y),
                      half(site.log_aniso), half(0.0f));
    }

    packed[gid] = out;
}

kernel void renderVoronoiBuffer(
    device const uint4 *candidates0 [[buffer(0)]],
    device const uint4 *candidates1 [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *sitesRaw [[buffer(3)]],
    constant float &inv_scale_sq [[buffer(4)]],
    constant uint &siteCount [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &height [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= height) { return; }

    uint pixelIdx = gid.y * width + gid.x;
    float2 uv = float2(gid);

    uint4 c0 = candidates0[pixelIdx];
    uint4 c1 = candidates1[pixelIdx];
    uint candIds[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};

    float logits[8];
    float max_logit = -INFINITY;

    for (uint i = 0; i < 8; ++i) {
        uint idx = candIds[i];
        if (idx >= siteCount) {
            logits[i] = -INFINITY;
            continue;
        }
        VoronoiSite site = load_site(sitesRaw, idx);
        if (site.position.x < 0.0f) {
            logits[i] = -INFINITY;
            continue;
        }
        float tau = exp(site.log_tau);
        float dmix2 = voronoi_dmix2(site, uv, inv_scale_sq);
        logits[i] = -tau * dmix2;
        max_logit = max(max_logit, logits[i]);
    }

    if (max_logit == -INFINITY) {
        output[pixelIdx * 3 + 0] = 0.0f;
        output[pixelIdx * 3 + 1] = 0.0f;
        output[pixelIdx * 3 + 2] = 0.0f;
        return;
    }

    float weights[8];
    float sum_w = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        weights[i] = exp(logits[i] - max_logit);
        sum_w += weights[i];
    }
    float inv_sum = 1.0f / max(sum_w, 1e-8f);

    float3 color = float3(0.0f);
    for (uint i = 0; i < 8; ++i) {
        float w = weights[i] * inv_sum;
        uint idx = candIds[i];
        if (!isnan(w) && !isinf(w) && idx < siteCount) {
            float pos_x = sitesRaw[idx * 10u + 0u];
            if (pos_x >= 0.0f) {
                float3 siteColor = float3(
                    sitesRaw[idx * 10u + 4u],
                    sitesRaw[idx * 10u + 5u],
                    sitesRaw[idx * 10u + 6u]
                );
                color += w * siteColor;
            }
        }
    }

    output[pixelIdx * 3 + 0] = color.r;
    output[pixelIdx * 3 + 1] = color.g;
    output[pixelIdx * 3 + 2] = color.b;
}

kernel void renderVoronoiBackwardBuffer(
    device const uint4 *candidates0 [[buffer(0)]],
    device const uint4 *candidates1 [[buffer(1)]],
    device const float *grad_output [[buffer(2)]],
    device atomic_float *grad_sites [[buffer(3)]],
    device const float *sitesRaw [[buffer(4)]],
    constant float &inv_scale_sq [[buffer(5)]],
    constant uint &siteCount [[buffer(6)]],
    constant uint &width [[buffer(7)]],
    constant uint &height [[buffer(8)]],
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid.x < width && gid.y < height) {
        uint pixelIdx = gid.y * width + gid.x;
        float2 uv = float2(gid);

        uint4 c0 = candidates0[pixelIdx];
        uint4 c1 = candidates1[pixelIdx];
        uint candIds[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};

        float weights[8];
        float3 pred;
        if (computeCandidateWeightsRaw(candIds, sitesRaw, siteCount, uv, inv_scale_sq, weights, pred)) {
            float3 dL_dpred = float3(
                grad_output[pixelIdx * 3 + 0],
                grad_output[pixelIdx * 3 + 1],
                grad_output[pixelIdx * 3 + 2]);

            for (uint i = 0; i < 8; ++i) {
                uint idx = candIds[i];
                if (idx >= siteCount) continue;
                VoronoiSite site = load_site(sitesRaw, idx);
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
                    device atomic_float *g = grad_sites + idx * 10u;
                    atomic_fetch_add_explicit(&g[0], dL_dpos.x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[1], dL_dpos.y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[2], dL_dlog_tau, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[3], dL_dradius, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[4], dL_dcolor.r, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[5], dL_dcolor.g, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[6], dL_dcolor.b, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[7], dL_ddir.x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[8], dL_ddir.y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g[9], dL_dlog_aniso, memory_order_relaxed);
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

        device atomic_float *g = grad_sites + siteID * 10u;
        atomic_fetch_add_explicit(&g[0], g_pos_x, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[1], g_pos_y, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[2], g_log_tau, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[3], g_radius, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[4], g_color_r, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[5], g_color_g, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[6], g_color_b, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[7], g_dir_x, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[8], g_dir_y, memory_order_relaxed);
        atomic_fetch_add_explicit(&g[9], g_log_aniso, memory_order_relaxed);
    }
}
