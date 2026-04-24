#ifndef VORONOI_COMMON_METAL
#define VORONOI_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

// Site parameters structure
struct VoronoiSite {
    float2 position;    // Position in pixels
    float log_tau;      // log(temperature)
    float radius;      // Radius (pixels)
    float3 color;       // RGB color
    float2 aniso_dir;   // Unit direction for anisotropy (image-space)
    float log_aniso;    // log(aspect), det(G)=1 via (e^a, e^-a)
};

// Packed site format for inference-only rendering.
struct PackedInferenceSite {
    uint4 data;
};

// Packed site format for candidate search (two half4 = 16 bytes).
struct PackedCandidateSite {
    half4 a;
    half4 b;
};

struct PackedSiteQuant {
    float logTauMin;
    float logTauScale;
    float radiusMin;
    float radiusScale;
    float colorRMin;
    float colorRScale;
    float colorGMin;
    float colorGScale;
    float colorBMin;
    float colorBScale;
};

// Adam optimizer state
struct AdamState {
    float2 m_pos;
    float v_pos;
    float m_log_tau;
    float v_log_tau;
    float m_radius;
    float v_radius;
    float3 m_color;
    float3 v_color;
    float2 m_dir;
    float v_dir;
    float m_log_aniso;
    float v_log_aniso;
    float2 _pad;
};

// RNG utilities
inline uint xorshift32(uint x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline float rand01(thread uint &state) {
    state = xorshift32(state);
    return float(state) * (1.0f / 4294967296.0f);
}

constant float kPackedInvUnorm16 = 1.0f / 65535.0f;
constant float kPackedInvUnorm15 = 1.0f / 32767.0f;
constant float kPackedInvUnorm11 = 1.0f / 2047.0f;
constant float kPackedInvUnorm10 = 1.0f / 1023.0f;
constant float kPackedPi = 3.14159265f;
constant float kPackedTwoPi = 6.2831853f;
inline float unpackUnorm16(uint v, float minVal, float scale) {
    return minVal + float(v) * kPackedInvUnorm16 * max(scale, 0.0f);
}

inline float unpackUnorm10(uint v, float minVal, float scale) {
    return minVal + float(v) * kPackedInvUnorm10 * max(scale, 0.0f);
}

inline bool packedActive(PackedInferenceSite packed) {
    return ((packed.data.x >> 30) & 1u) == 0u;
}

inline VoronoiSite decodePackedSite(PackedInferenceSite packed, float2 dims,
                                    constant PackedSiteQuant &quant) {
    VoronoiSite site;
    uint w0 = packed.data.x;
    uint w1 = packed.data.y;
    uint w2 = packed.data.z;
    uint w3 = packed.data.w;

    uint px = w0 & 0x7fffu;
    uint py = (w0 >> 15) & 0x7fffu;
    site.position = float2(float(px) * kPackedInvUnorm15 * max(dims.x, 0.0f),
                           float(py) * kPackedInvUnorm15 * max(dims.y, 0.0f));

    site.color = float3(
        quant.colorRMin + float(w1 & 0x7ffu) * kPackedInvUnorm11 * max(quant.colorRScale, 0.0f),
        quant.colorGMin + float((w1 >> 11) & 0x7ffu) * kPackedInvUnorm11 * max(quant.colorGScale, 0.0f),
        quant.colorBMin + float((w1 >> 22) & 0x3ffu) * kPackedInvUnorm10 * max(quant.colorBScale, 0.0f)
    );

    uint lt = w2 & 0xffffu;
    uint rd = (w2 >> 16) & 0xffffu;
    site.log_tau = unpackUnorm16(lt, quant.logTauMin, quant.logTauScale);
    site.radius = unpackUnorm16(rd, quant.radiusMin, quant.radiusScale);

    uint angleBits = w3 & 0xffffu;
    ushort logAnisoBits = ushort((w3 >> 16) & 0xffffu);
    float angle = float(angleBits) * kPackedInvUnorm16 * kPackedTwoPi - kPackedPi;
    site.aniso_dir = float2(cos(angle), sin(angle));
    site.log_aniso = float(as_type<half>(logAnisoBits));

    return site;
}

// Compute soft Voronoi distance
inline float voronoi_dmix2(VoronoiSite site, float2 uv, float inv_scale_sq) {
    float2 diff = uv - site.position;

    // Anisotropic metric (SPD, det=1):
    // G = e^a * uu^T + e^-a * (I - uu^T)
    // diff^T G diff = e^a * (dot(u, diff))^2 + e^-a * (|diff|^2 - (dot(u, diff))^2)
    float diff2 = dot(diff, diff);
    float proj = dot(site.aniso_dir, diff);
    float proj2 = proj * proj;
    float perp2 = max(diff2 - proj2, 0.0f);

    float l1 = exp(site.log_aniso);
    float l2 = 1.0f / l1;
    float d2_aniso = l1 * proj2 + l2 * perp2;

    float d2_norm = d2_aniso * inv_scale_sq;
    float d2_safe = max(d2_norm, 1e-8f);
    float inv_scale = sqrt(inv_scale_sq);
    float r_norm = site.radius * inv_scale;
    return (sqrt(d2_safe) - r_norm);
}

inline float removalDeltaForSite(float3 pred, float3 tgt, float3 siteColor, float w_total) {
    float keep = 1.0f - w_total;
    if (keep < 1e-4f) {
        // Removing the site would leave no meaningful support; treat as very costly.
        return 1.0e6f;
    }
    float3 pred_keep = (pred - w_total * siteColor) / keep;
    float3 diff = pred - tgt;
    float3 diff_keep = pred_keep - tgt;
    return dot(diff_keep, diff_keep) - dot(diff, diff);
}

constant float3 kLumaWeights = float3(0.299f, 0.587f, 0.114f);
constant float kSSIMC1 = 0.0001f;  // (0.01)^2 for inputs in [0,1]
constant float kSSIMC2 = 0.0009f;  // (0.03)^2 for inputs in [0,1]
constant float kSSIMInvN = 1.0f / 9.0f;

inline void ssim_window_stats(
    texture2d<float, access::read> rendered,
    texture2d<float, access::read> target,
    uint2 gid,
    thread float &mu_x,
    thread float &mu_y,
    thread float &ex2,
    thread float &ey2,
    thread float &exy)
{
    uint width = rendered.get_width();
    uint height = rendered.get_height();
    mu_x = 0.0f;
    mu_y = 0.0f;
    ex2 = 0.0f;
    ey2 = 0.0f;
    exy = 0.0f;

    int max_x = int(width) - 1;
    int max_y = int(height) - 1;

    for (int dy = -1; dy <= 1; ++dy) {
        int yy = clamp(int(gid.y) + dy, 0, max_y);
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = clamp(int(gid.x) + dx, 0, max_x);
            uint2 p = uint2(xx, yy);
            float x = dot(rendered.read(p).rgb, kLumaWeights);
            float y = dot(target.read(p).rgb, kLumaWeights);
            mu_x += x;
            mu_y += y;
            ex2 += x * x;
            ey2 += y * y;
            exy += x * y;
        }
    }

    mu_x *= kSSIMInvN;
    mu_y *= kSSIMInvN;
    ex2 *= kSSIMInvN;
    ey2 *= kSSIMInvN;
    exy *= kSSIMInvN;
}

inline void atomic_fetch_add_explicit_float(device float *addr, float val);

inline void insertSorted8(
    thread uint  bestIdx[8],
    thread float bestD2[8],
    uint candIdx,
    float d2)
{
    // Condition-based insertion sort (faster than loop)
    // Split into binary search then conditional shift: 2 + 4 + 4 = 10 conditions vs 8*8=64 loop iterations

    // Binary search: which half?
    bool inFirstHalf = d2 < bestD2[3];

    if (inFirstHalf) {
        // First half [0-3]: check which position
        if (d2 < bestD2[1]) {
            if (d2 < bestD2[0]) {
                // Insert at 0: shift [0-6] -> [1-7]
                bestD2[7] = bestD2[6]; bestIdx[7] = bestIdx[6];
                bestD2[6] = bestD2[5]; bestIdx[6] = bestIdx[5];
                bestD2[5] = bestD2[4]; bestIdx[5] = bestIdx[4];
                bestD2[4] = bestD2[3]; bestIdx[4] = bestIdx[3];
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = bestD2[1]; bestIdx[2] = bestIdx[1];
                bestD2[1] = bestD2[0]; bestIdx[1] = bestIdx[0];
                bestD2[0] = d2; bestIdx[0] = candIdx;
            } else {
                // Insert at 1: shift [1-6] -> [2-7]
                bestD2[7] = bestD2[6]; bestIdx[7] = bestIdx[6];
                bestD2[6] = bestD2[5]; bestIdx[6] = bestIdx[5];
                bestD2[5] = bestD2[4]; bestIdx[5] = bestIdx[4];
                bestD2[4] = bestD2[3]; bestIdx[4] = bestIdx[3];
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = bestD2[1]; bestIdx[2] = bestIdx[1];
                bestD2[1] = d2; bestIdx[1] = candIdx;
            }
        } else {
            if (d2 < bestD2[2]) {
                // Insert at 2: shift [2-6] -> [3-7]
                bestD2[7] = bestD2[6]; bestIdx[7] = bestIdx[6];
                bestD2[6] = bestD2[5]; bestIdx[6] = bestIdx[5];
                bestD2[5] = bestD2[4]; bestIdx[5] = bestIdx[4];
                bestD2[4] = bestD2[3]; bestIdx[4] = bestIdx[3];
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = d2; bestIdx[2] = candIdx;
            } else if (d2 < bestD2[3]) {
                // Insert at 3: shift [3-6] -> [4-7]
                bestD2[7] = bestD2[6]; bestIdx[7] = bestIdx[6];
                bestD2[6] = bestD2[5]; bestIdx[6] = bestIdx[5];
                bestD2[5] = bestD2[4]; bestIdx[5] = bestIdx[4];
                bestD2[4] = bestD2[3]; bestIdx[4] = bestIdx[3];
                bestD2[3] = d2; bestIdx[3] = candIdx;
            }
        }
    } else {
        // Second half [4-7]: check which position
        if (d2 < bestD2[5]) {
            if (d2 < bestD2[4]) {
                // Insert at 4: shift [4-6] -> [5-7]
                bestD2[7] = bestD2[6]; bestIdx[7] = bestIdx[6];
                bestD2[6] = bestD2[5]; bestIdx[6] = bestIdx[5];
                bestD2[5] = bestD2[4]; bestIdx[5] = bestIdx[4];
                bestD2[4] = d2; bestIdx[4] = candIdx;
            } else {
                // Insert at 5: shift [5-6] -> [6-7]
                bestD2[7] = bestD2[6]; bestIdx[7] = bestIdx[6];
                bestD2[6] = bestD2[5]; bestIdx[6] = bestIdx[5];
                bestD2[5] = d2; bestIdx[5] = candIdx;
            }
        } else {
            if (d2 < bestD2[6]) {
                // Insert at 6: shift [6] -> [7]
                bestD2[7] = bestD2[6]; bestIdx[7] = bestIdx[6];
                bestD2[6] = d2; bestIdx[6] = candIdx;
            } else if (d2 < bestD2[7]) {
                // Insert at 7
                bestD2[7] = d2; bestIdx[7] = candIdx;
            }
        }
    }
}

inline void insertClosest8(
    thread uint  bestIdx[8],
    thread float bestD2[8],
    uint candIdx,
    float2 uv,
    constant VoronoiSite *sites,
    uint siteCount,
    float inv_scale_sq)
{
    if (candIdx >= siteCount) return;

    // Check if already in list
    for (uint i = 0; i < 8; ++i) {
        if (bestIdx[i] == candIdx) return;
    }

    if (sites[candIdx].position.x < 0.0f) {
        return;
    }
    VoronoiSite site = sites[candIdx];
    float dMix2 = voronoi_dmix2(site, uv, inv_scale_sq);
    float tau = max(exp(site.log_tau), 1e-4f);
    float d2 = tau * dMix2;

    insertSorted8(bestIdx, bestD2, candIdx, d2);
}

inline void mergeCandidates8(
    thread uint  bestIdx[8],
    thread float bestD2[8],
    uint4 c0,
    uint4 c1,
    float2 uv,
    constant VoronoiSite *sites,
    uint siteCount,
    float inv_scale_sq)
{
    insertClosest8(bestIdx, bestD2, c0.x, uv, sites, siteCount, inv_scale_sq);
    insertClosest8(bestIdx, bestD2, c0.y, uv, sites, siteCount, inv_scale_sq);
    insertClosest8(bestIdx, bestD2, c0.z, uv, sites, siteCount, inv_scale_sq);
    insertClosest8(bestIdx, bestD2, c0.w, uv, sites, siteCount, inv_scale_sq);
    insertClosest8(bestIdx, bestD2, c1.x, uv, sites, siteCount, inv_scale_sq);
    insertClosest8(bestIdx, bestD2, c1.y, uv, sites, siteCount, inv_scale_sq);
    insertClosest8(bestIdx, bestD2, c1.z, uv, sites, siteCount, inv_scale_sq);
    insertClosest8(bestIdx, bestD2, c1.w, uv, sites, siteCount, inv_scale_sq);
}

inline void insertClosest8Packed(
    thread uint  bestIdx[8],
    thread float bestD2[8],
    uint candIdx,
    float2 uv,
    constant PackedInferenceSite *sites,
    constant PackedSiteQuant &quant,
    uint siteCount,
    float inv_scale_sq,
    float2 dims)
{
    if (candIdx >= siteCount) return;

    for (uint i = 0; i < 8; ++i) {
        if (bestIdx[i] == candIdx) return;
    }

    PackedInferenceSite packed = sites[candIdx];
    if (!packedActive(packed)) return;

    VoronoiSite site = decodePackedSite(packed, dims, quant);
    float dMix2 = voronoi_dmix2(site, uv, inv_scale_sq);
    float tau = max(exp(site.log_tau), 1e-4f);
    float d2 = tau * dMix2;

    insertSorted8(bestIdx, bestD2, candIdx, d2);
}

inline void mergeCandidates8Packed(
    thread uint  bestIdx[8],
    thread float bestD2[8],
    uint4 c0,
    uint4 c1,
    float2 uv,
    constant PackedInferenceSite *sites,
    constant PackedSiteQuant &quant,
    uint siteCount,
    float inv_scale_sq,
    float2 dims)
{
    insertClosest8Packed(bestIdx, bestD2, c0.x, uv, sites, quant, siteCount, inv_scale_sq, dims);
    insertClosest8Packed(bestIdx, bestD2, c0.y, uv, sites, quant, siteCount, inv_scale_sq, dims);
    insertClosest8Packed(bestIdx, bestD2, c0.z, uv, sites, quant, siteCount, inv_scale_sq, dims);
    insertClosest8Packed(bestIdx, bestD2, c0.w, uv, sites, quant, siteCount, inv_scale_sq, dims);
    insertClosest8Packed(bestIdx, bestD2, c1.x, uv, sites, quant, siteCount, inv_scale_sq, dims);
    insertClosest8Packed(bestIdx, bestD2, c1.y, uv, sites, quant, siteCount, inv_scale_sq, dims);
    insertClosest8Packed(bestIdx, bestD2, c1.z, uv, sites, quant, siteCount, inv_scale_sq, dims);
    insertClosest8Packed(bestIdx, bestD2, c1.w, uv, sites, quant, siteCount, inv_scale_sq, dims);
}

inline float candidatePackedDistance(PackedCandidateSite packed, float2 uv, float inv_scale_sq) {
    float2 pos = float2(packed.a.xy);
    if (pos.x < 0.0f) {
        return INFINITY;
    }
    float log_tau = float(packed.a.z);
    float radius = float(packed.a.w);
    float2 dir = float2(packed.b.xy);
    float log_aniso = float(packed.b.z);

    VoronoiSite site;
    site.position = pos;
    site.log_tau = log_tau;
    site.radius = radius;
    site.aniso_dir = dir;
    site.log_aniso = log_aniso;

    float dMix2 = voronoi_dmix2(site, uv, inv_scale_sq);
    float tau = max(exp(log_tau), 1e-4f);
    return tau * dMix2;
}

inline void insertClosest8CandidatePacked(
    thread uint  bestIdx[8],
    thread float bestD2[8],
    uint candIdx,
    float2 uv,
    constant PackedCandidateSite *sites,
    uint siteCount,
    float inv_scale_sq)
{
    if (candIdx >= siteCount) return;

    for (uint i = 0; i < 8; ++i) {
        if (bestIdx[i] == candIdx) return;
    }

    PackedCandidateSite packed = sites[candIdx];
    float d2 = candidatePackedDistance(packed, uv, inv_scale_sq);
    insertSorted8(bestIdx, bestD2, candIdx, d2);
}

inline void mergeCandidates8CandidatePacked(
    thread uint  bestIdx[8],
    thread float bestD2[8],
    uint4 c0,
    uint4 c1,
    float2 uv,
    constant PackedCandidateSite *sites,
    uint siteCount,
    float inv_scale_sq)
{
    insertClosest8CandidatePacked(bestIdx, bestD2, c0.x, uv, sites, siteCount, inv_scale_sq);
    insertClosest8CandidatePacked(bestIdx, bestD2, c0.y, uv, sites, siteCount, inv_scale_sq);
    insertClosest8CandidatePacked(bestIdx, bestD2, c0.z, uv, sites, siteCount, inv_scale_sq);
    insertClosest8CandidatePacked(bestIdx, bestD2, c0.w, uv, sites, siteCount, inv_scale_sq);
    insertClosest8CandidatePacked(bestIdx, bestD2, c1.x, uv, sites, siteCount, inv_scale_sq);
    insertClosest8CandidatePacked(bestIdx, bestD2, c1.y, uv, sites, siteCount, inv_scale_sq);
    insertClosest8CandidatePacked(bestIdx, bestD2, c1.z, uv, sites, siteCount, inv_scale_sq);
    insertClosest8CandidatePacked(bestIdx, bestD2, c1.w, uv, sites, siteCount, inv_scale_sq);
}


// ============================================================================
// JFA (Jump Flood Algorithm) - True flood propagation for 4 closest sites
// ============================================================================

// Insert into sorted list of 4 closest
inline void insertSorted4(
    thread uint  bestIdx[4],
    thread float bestD2[4],
    uint candIdx,
    float d2)
{
    // Insertion sort into 4-element list
    if (d2 < bestD2[3]) {
        if (d2 < bestD2[1]) {
            if (d2 < bestD2[0]) {
                // Insert at 0
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = bestD2[1]; bestIdx[2] = bestIdx[1];
                bestD2[1] = bestD2[0]; bestIdx[1] = bestIdx[0];
                bestD2[0] = d2; bestIdx[0] = candIdx;
            } else {
                // Insert at 1
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = bestD2[1]; bestIdx[2] = bestIdx[1];
                bestD2[1] = d2; bestIdx[1] = candIdx;
            }
        } else {
            if (d2 < bestD2[2]) {
                // Insert at 2
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = d2; bestIdx[2] = candIdx;
            } else {
                // Insert at 3
                bestD2[3] = d2; bestIdx[3] = candIdx;
            }
        }
    }
}

inline void insertClosest4(
    thread uint  bestIdx[4],
    thread float bestD2[4],
    uint candIdx,
    float2 uv,
    constant VoronoiSite *sites,
    uint siteCount,
    float inv_scale_sq)
{
    if (candIdx >= siteCount) return;

    // Check if already in list
    for (uint i = 0; i < 4; ++i) {
        if (bestIdx[i] == candIdx) return;
    }

    // Skip inactive sites
    if (sites[candIdx].position.x < 0.0f) return;

    VoronoiSite site = sites[candIdx];
    float dMix2 = voronoi_dmix2(site, uv, inv_scale_sq);
    float tau = max(exp(site.log_tau), 1e-4f);
    float d2 = tau * dMix2;

    insertSorted4(bestIdx, bestD2, candIdx, d2);
}

inline void insertClosest4Packed(
    thread uint  bestIdx[4],
    thread float bestD2[4],
    uint candIdx,
    float2 uv,
    constant PackedInferenceSite *sites,
    constant PackedSiteQuant &quant,
    uint siteCount,
    float inv_scale_sq,
    float2 dims)
{
    if (candIdx >= siteCount) return;

    for (uint i = 0; i < 4; ++i) {
        if (bestIdx[i] == candIdx) return;
    }

    PackedInferenceSite packed = sites[candIdx];
    if (!packedActive(packed)) return;
    VoronoiSite site = decodePackedSite(packed, dims, quant);
    float dMix2 = voronoi_dmix2(site, uv, inv_scale_sq);
    float tau = max(exp(site.log_tau), 1e-4f);
    float d2 = tau * dMix2;

    insertSorted4(bestIdx, bestD2, candIdx, d2);
}

// Hash function for site ID -> color
inline float3 hashColor(uint siteId) {
    // Triple hash for RGB channels
    uint h = siteId;
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    h = h ^ (h >> 4u);
    uint r = h * 0x27d4eb2du;

    h = siteId * 2654435761u;
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    uint g = h * 0x27d4eb2du;

    h = siteId * 1103515245u;
    h = (h ^ 61u) ^ (h >> 16u);
    h = h ^ (h >> 4u);
    uint b = h * 0x27d4eb2du;

    return float3(
        float(r & 0xFFFFFFu) / float(0xFFFFFFu),
        float(g & 0xFFFFFFu) / float(0xFFFFFFu),
        float(b & 0xFFFFFFu) / float(0xFFFFFFu)
    );
}

// Tau heatmap parameters
struct TauHeatmapParams {
    float minTau;
    float meanTau;
    float maxTau;
    float dotRadius;
};

inline bool computePackedWeights(
    thread uint candIds[8],
    constant PackedInferenceSite *sites,
    constant PackedSiteQuant &quant,
    uint siteCount,
    float2 uv,
    float inv_scale_sq,
    float2 dims,
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
        PackedInferenceSite packed = sites[idx];
        if (!packedActive(packed)) {
            logits[i] = -INFINITY;
            continue;
        }
        VoronoiSite site = decodePackedSite(packed, dims, quant);
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
        if (idx >= siteCount) continue;
        PackedInferenceSite packed = sites[idx];
        if (!packedActive(packed)) continue;
        VoronoiSite site = decodePackedSite(packed, dims, quant);
        pred += weights[i] * site.color;
    }

    return true;
}

// Helper functions for atomic float operations using atomic_int storage
// Quantization scale for gradient accumulation (1e6 gives ~1e-6 precision)
constant float kGradQuantScale = 1000000.0f;
constant float kGradQuantScaleInv = 1.0f / 1000000.0f;

inline void atomic_fetch_add_float(threadgroup atomic_int *addr, float val) {
    // Quantize: convert float to scaled int for atomic add
    int quantized = int(val * kGradQuantScale);
    atomic_fetch_add_explicit(addr, quantized, memory_order_relaxed);
}

inline float atomic_load_float(threadgroup atomic_int *addr) {
    // Dequantize: convert scaled int back to float
    int quantized = atomic_load_explicit(addr, memory_order_relaxed);
    return float(quantized) * kGradQuantScaleInv;
}

inline void atomic_fetch_add_tg_float(threadgroup atomic_int *addr, float val) {
    int expected = atomic_load_explicit(addr, memory_order_relaxed);
    int desired;
    do {
        desired = as_type<int>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, desired,
                                                     memory_order_relaxed,
                                                     memory_order_relaxed));
}

inline float atomic_load_tg_float(threadgroup atomic_int *addr) {
    int bits = atomic_load_explicit(addr, memory_order_relaxed);
    return as_type<float>(bits);
}

inline void atomic_fetch_add_explicit_float(device float *addr, float val) {
    // For device memory, we can use regular atomic float operations or CAS loop
    // Metal supports direct float atomics on some hardware, but CAS is more portable
    device atomic_int *addr_as_int = (device atomic_int *)addr;
    int expected = atomic_load_explicit(addr_as_int, memory_order_relaxed);
    int desired;
    do {
        desired = as_type<int>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(addr_as_int, &expected, desired,
                                                     memory_order_relaxed,
                                                     memory_order_relaxed));
}

// ============================================================================

// TILED GRADIENT COMPUTATION: Threadgroup hash reduction (no global sort)
// ============================================================================

constant uint kTileHashSize = 256;
constant uint kTileHashMask = kTileHashSize - 1;
constant uint kTileMaxProbes = 8;
constant uint kTileEmptyKey = 0xffffffffu;

inline uint tile_hash(uint key) {
    return key * 2654435761u;
}

inline uint2 candidateCoord(uint2 gid, uint2 outputSize, uint2 candSize) {
    uint outW = max(outputSize.x, 1u);
    uint outH = max(outputSize.y, 1u);
    uint candX = min((gid.x * candSize.x) / outW, max(candSize.x, 1u) - 1u);
    uint candY = min((gid.y * candSize.y) / outH, max(candSize.y, 1u) - 1u);
    return uint2(candX, candY);
}

inline void loadCandidateIdsAtCoord(
    texture2d<uint, access::read> candidates0,
    texture2d<uint, access::read> candidates1,
    uint2 coord,
    thread uint outIds[8])
{
    uint4 c0 = candidates0.read(coord);
    uint4 c1 = candidates1.read(coord);
    outIds[0] = c0.x;
    outIds[1] = c0.y;
    outIds[2] = c0.z;
    outIds[3] = c0.w;
    outIds[4] = c1.x;
    outIds[5] = c1.y;
    outIds[6] = c1.z;
    outIds[7] = c1.w;
}

inline void loadCandidateIds(
    texture2d<uint, access::read> candidates0,
    texture2d<uint, access::read> candidates1,
    uint2 gid,
    uint2 outputSize,
    thread uint outIds[8])
{
    uint2 candSize = uint2(candidates0.get_width(), candidates0.get_height());
    uint2 coord = candidateCoord(gid, outputSize, candSize);
    loadCandidateIdsAtCoord(candidates0, candidates1, coord, outIds);
}

inline bool computeCandidateWeights(
    thread uint candIds[8],
    constant VoronoiSite *sites,
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
        VoronoiSite site = sites[idx];
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
        if (idx < siteCount && sites[idx].position.x >= 0.0f) {
            pred += weights[i] * sites[idx].color;
        }
    }

    return true;
}

inline uint accumulateUniqueSites(
    thread uint candIds[8],
    thread float weights[8],
    constant VoronoiSite *sites,
    uint siteCount,
    thread uint uniqueSites[8],
    thread float uniqueWeights[8])
{
    uint numUnique = 0;
    for (uint i = 0; i < 8; ++i) {
        uint idx = candIds[i];
        if (idx >= siteCount) continue;
        if (sites[idx].position.x < 0.0f) continue;
        float w = weights[i];

        bool found = false;
        for (uint j = 0; j < numUnique; ++j) {
            if (uniqueSites[j] == idx) {
                uniqueWeights[j] += w;
                found = true;
                break;
            }
        }

        if (!found && numUnique < 8) {
            uniqueSites[numUnique] = idx;
            uniqueWeights[numUnique] = w;
            numUnique++;
        }
    }
    return numUnique;
}

#endif  // VORONOI_COMMON_METAL
