#include "../sad_common.metal"

inline float densify_sample_luma(texture2d<float, access::read> target,
                         uint2 ipos,
                         int width,
                         int height,
                         int dx,
                         int dy) {
    int sx = clamp(int(ipos.x) + dx, 0, width - 1);
    int sy = clamp(int(ipos.y) + dy, 0, height - 1);
    float3 rgb = target.read(uint2(sx, sy)).rgb;
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

kernel void splitSites(
    device VoronoiSite *sites [[buffer(0)]],
    device AdamState *adam [[buffer(1)]],
    constant uint *splitIndices [[buffer(2)]],  // Indices of sites to split
    constant uint &numToSplit [[buffer(3)]],
    const device float *mass [[buffer(4)]],
    const device float *err_w [[buffer(5)]],
    const device float *err_wx [[buffer(6)]],
    const device float *err_wy [[buffer(7)]],
    const device float *err_wxx [[buffer(8)]],
    const device float *err_wxy [[buffer(9)]],
    const device float *err_wyy [[buffer(10)]],
    constant uint &currentSiteCount [[buffer(11)]],
    texture2d<float, access::read> target [[texture(0)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numToSplit) return;

    uint parentIdx = splitIndices[gid];
    VoronoiSite parent = sites[parentIdx];

    if (parent.position.x < 0.0) return;  // Skip inactive sites

    uint width = target.get_width();
    uint height = target.get_height();

    // Compute residual-weighted centroid/covariance (from stats pass) for a robust split direction.
    float ew = err_w[parentIdx];
    float2 axis = parent.aniso_dir;
    float2 center = parent.position;
    float logAniso = parent.log_aniso * 0.8f;

    if (ew > 1e-3f) {
        float mx = err_wx[parentIdx] / ew;
        float my = err_wy[parentIdx] / ew;
        center = mix(parent.position, float2(mx, my), 0.6f);

        float exx = err_wxx[parentIdx] / ew - mx * mx;
        float exy = err_wxy[parentIdx] / ew - mx * my;
        float eyy = err_wyy[parentIdx] / ew - my * my;

        // Add a small diagonal to keep it well-conditioned.
        exx = max(exx, 1e-4f);
        eyy = max(eyy, 1e-4f);

        // Principal axis of 2x2 covariance.
        float theta = 0.5f * atan2(2.0f * exy, exx - eyy);
        axis = float2(cos(theta), sin(theta));

        float trace = exx + eyy;
        float disc = sqrt(max(0.0f, 0.25f * (exx - eyy) * (exx - eyy) + exy * exy));
        float lambda1 = max(1e-4f, 0.5f * trace + disc);
        float lambda2 = max(1e-4f, 0.5f * trace - disc);

        // Map variance ratio -> anisotropy (det=1 metric): elongate along axis => smaller metric eigenvalue along axis.
        logAniso = clamp(-0.5f * log(lambda1 / lambda2), -2.0f, 2.0f);
    } else {
        // Fallback: use local image gradient as a split axis.
        uint2 ip = uint2(clamp(parent.position, float2(0.0), float2(width - 1, height - 1)));
        float TL = densify_sample_luma(target, ip, int(width), int(height), -1, -1);
        float T  = densify_sample_luma(target, ip, int(width), int(height),  0, -1);
        float TR = densify_sample_luma(target, ip, int(width), int(height),  1, -1);
        float L  = densify_sample_luma(target, ip, int(width), int(height), -1,  0);
        float R  = densify_sample_luma(target, ip, int(width), int(height),  1,  0);
        float BL = densify_sample_luma(target, ip, int(width), int(height), -1,  1);
        float B  = densify_sample_luma(target, ip, int(width), int(height),  0,  1);
        float BR = densify_sample_luma(target, ip, int(width), int(height),  1,  1);
        float Gx = (-TL + TR - 2.0*L + 2.0*R - BL + BR) / 4.0;
        float Gy = (-TL - 2.0*T - TR + BL + 2.0*B + BR) / 4.0;
        float2 grad = float2(Gx, Gy);
        float g2 = dot(grad, grad);
        if (g2 > 1e-8f) axis = grad * rsqrt(g2);
    }

    // Split offset along axis.
    // Use site mass (approx covered pixels) as a geometric scale proxy.
    float m = max(mass[parentIdx], 1.0f);
    float offsetDist = clamp(sqrt(m) * 0.5f, 1.5f, 48.0f);
    float2 offset = axis * offsetDist;

    float2 posA = clamp(center + offset, float2(0.0), float2(width - 1, height - 1));
    float2 posB = clamp(center - offset, float2(0.0), float2(width - 1, height - 1));
    float3 colA = target.read(uint2(posA)).rgb;
    float3 colB = target.read(uint2(posB)).rgb;

    // In-place split: parent becomes one child, and the other child appends at the end.
    uint childIdx = currentSiteCount + gid;
    if (childIdx == parentIdx) return;

    VoronoiSite child0 = parent;
    VoronoiSite child1 = parent;

    child0.position = posA;
    child1.position = posB;
    child0.color = colA;
    child1.color = colB;

    child0.log_tau = parent.log_tau - 0.25f;
    child1.log_tau = parent.log_tau - 0.25f;
    child0.radius = parent.radius * 0.85f;
    child1.radius = parent.radius * 0.85f;

    child0.aniso_dir = axis;
    child1.aniso_dir = axis;
    child0.log_aniso = logAniso;
    child1.log_aniso = logAniso;

    sites[parentIdx] = child0;
    sites[childIdx] = child1;

    // Reset optimizer state for the modified parent and new child.
    adam[parentIdx] = AdamState();
    adam[childIdx] = AdamState();
}

kernel void computeSiteStatsTiled(
    texture2d<uint, access::read> candidates0 [[texture(0)]],
    texture2d<uint, access::read> candidates1 [[texture(1)]],
    texture2d<float, access::read> target [[texture(2)]],
    texture2d<float, access::read> mask [[texture(3)]],
    constant VoronoiSite *sites [[buffer(0)]],
    constant float &inv_scale_sq [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    device float *mass [[buffer(3)]],
    device float *energy [[buffer(4)]],
    device float *err_w [[buffer(5)]],
    device float *err_wx [[buffer(6)]],
    device float *err_wy [[buffer(7)]],
    device float *err_wxx [[buffer(8)]],
    device float *err_wxy [[buffer(9)]],
    device float *err_wyy [[buffer(10)]],
    threadgroup atomic_uint *tg_keys [[threadgroup(0)]],
    threadgroup atomic_int *tg_mass [[threadgroup(1)]],
    threadgroup atomic_int *tg_energy [[threadgroup(2)]],
    threadgroup atomic_int *tg_ew [[threadgroup(3)]],
    threadgroup atomic_int *tg_ewx [[threadgroup(4)]],
    threadgroup atomic_int *tg_ewy [[threadgroup(5)]],
    threadgroup atomic_int *tg_ewxx [[threadgroup(6)]],
    threadgroup atomic_int *tg_ewxy [[threadgroup(7)]],
    threadgroup atomic_int *tg_ewyy [[threadgroup(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]])
{
    uint local_idx = tid.y * tgs.x + tid.x;
    uint local_threads = tgs.x * tgs.y;

    for (uint i = local_idx; i < kTileHashSize; i += local_threads) {
        atomic_store_explicit(&tg_keys[i], kTileEmptyKey, memory_order_relaxed);
        atomic_store_explicit(&tg_mass[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_energy[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_ew[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_ewx[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_ewy[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_ewxx[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_ewxy[i], 0, memory_order_relaxed);
        atomic_store_explicit(&tg_ewyy[i], 0, memory_order_relaxed);
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
            float3 diff = pred - tgt;
            float err = dot(diff, diff);

            uint uniqueSites[8];
            float uniqueWeights[8];
            uint numUnique = accumulateUniqueSites(candIds, weights, sites, siteCount,
                                                   uniqueSites, uniqueWeights);

            for (uint i = 0; i < numUnique; ++i) {
                uint idx = uniqueSites[i];
                float w_total = uniqueWeights[i];
                if (w_total <= 0.0f) continue;

                if (w_total <= 0.0f) {
                    continue;
                }
                float werr = w_total * min(err, 1.0f);
                bool inserted = false;
                uint base = tile_hash(idx) & kTileHashMask;

                for (uint probe = 0; probe < kTileMaxProbes; ++probe) {
                    uint slot = (base + probe) & kTileHashMask;
                    uint expected = kTileEmptyKey;
                    if (atomic_compare_exchange_weak_explicit(&tg_keys[slot], &expected, idx,
                                                              memory_order_relaxed, memory_order_relaxed) ||
                        expected == idx) {
                        atomic_fetch_add_tg_float(&tg_mass[slot], w_total);
                        atomic_fetch_add_tg_float(&tg_energy[slot], w_total * err);
                        atomic_fetch_add_tg_float(&tg_ew[slot], werr);
                        atomic_fetch_add_tg_float(&tg_ewx[slot], werr * uv.x);
                        atomic_fetch_add_tg_float(&tg_ewy[slot], werr * uv.y);
                        atomic_fetch_add_tg_float(&tg_ewxx[slot], werr * uv.x * uv.x);
                        atomic_fetch_add_tg_float(&tg_ewxy[slot], werr * uv.x * uv.y);
                        atomic_fetch_add_tg_float(&tg_ewyy[slot], werr * uv.y * uv.y);
                        inserted = true;
                        break;
                    }
                }

                if (!inserted) {
                    atomic_fetch_add_explicit_float(&mass[idx], w_total);
                    atomic_fetch_add_explicit_float(&energy[idx], w_total * err);
                    atomic_fetch_add_explicit_float(&err_w[idx], werr);
                    atomic_fetch_add_explicit_float(&err_wx[idx], werr * uv.x);
                    atomic_fetch_add_explicit_float(&err_wy[idx], werr * uv.y);
                    atomic_fetch_add_explicit_float(&err_wxx[idx], werr * uv.x * uv.x);
                    atomic_fetch_add_explicit_float(&err_wxy[idx], werr * uv.x * uv.y);
                    atomic_fetch_add_explicit_float(&err_wyy[idx], werr * uv.y * uv.y);
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = local_idx; i < kTileHashSize; i += local_threads) {
        uint siteID = atomic_load_explicit(&tg_keys[i], memory_order_relaxed);
        if (siteID == kTileEmptyKey) continue;

        float m = atomic_load_tg_float(&tg_mass[i]);
        float e = atomic_load_tg_float(&tg_energy[i]);
        float ew = atomic_load_tg_float(&tg_ew[i]);
        float ewx = atomic_load_tg_float(&tg_ewx[i]);
        float ewy = atomic_load_tg_float(&tg_ewy[i]);
        float ewxx = atomic_load_tg_float(&tg_ewxx[i]);
        float ewxy = atomic_load_tg_float(&tg_ewxy[i]);
        float ewyy = atomic_load_tg_float(&tg_ewyy[i]);

        atomic_fetch_add_explicit_float(&mass[siteID], m);
        atomic_fetch_add_explicit_float(&energy[siteID], e);
        atomic_fetch_add_explicit_float(&err_w[siteID], ew);
        atomic_fetch_add_explicit_float(&err_wx[siteID], ewx);
        atomic_fetch_add_explicit_float(&err_wy[siteID], ewy);
        atomic_fetch_add_explicit_float(&err_wxx[siteID], ewxx);
        atomic_fetch_add_explicit_float(&err_wxy[siteID], ewxy);
        atomic_fetch_add_explicit_float(&err_wyy[siteID], ewyy);
    }
}

kernel void computeDensifyScorePairs(
    const device VoronoiSite *sites [[buffer(0)]],
    const device float *mass [[buffer(1)]],
    const device float *energy [[buffer(2)]],
    device uint2 *pairs [[buffer(3)]],              // (key, siteID)
    constant uint &siteCount [[buffer(4)]],
    constant float &minMass [[buffer(5)]],
    constant float &scoreAlpha [[buffer(6)]],
    constant uint &pairCount [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= pairCount) return;

    uint2 outPair = uint2(0xffffffffu, 0xffffffffu);
    if (gid < siteCount) {
        float m = mass[gid];
        float e = energy[gid];
        bool active = (sites[gid].position.x >= 0.0f);
        if (active && m > minMass && isfinite(e)) {
            float denom = pow(max(m, 1e-8f), scoreAlpha);
            float score = max(e, 0.0f) / denom;  // normalized error (alpha in [0..1])
            // Scores are non-negative; for positive floats, uint bit order is monotonic.
            uint key = 0xffffffffu - as_type<uint>(score);
            outPair = uint2(key, gid);
        }
    }
    pairs[gid] = outPair;
}
