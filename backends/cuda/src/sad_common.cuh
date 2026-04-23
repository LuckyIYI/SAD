#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// Constants
#define NUM_CANDIDATES 8
#define TILE_HASH_SIZE 256
#define TILE_HASH_MASK 255
#define TILE_MAX_PROBES 8
#define TILE_EMPTY_KEY 0xFFFFFFFFu

// SSIM constants
#define SSIM_C1 0.0001f
#define SSIM_C2 0.0009f
#define SSIM_INV_N 0.111111111f

// Site parameter structure (10 floats, padding handled implicitly by alignment).
struct Site {
    float2 position;
    float log_tau;
    float radius;
    float color_r;
    float color_g;
    float color_b;
    float aniso_dir_x;
    float aniso_dir_y;
    float log_aniso;
};

// Packed site format for candidate search (16 bytes).
struct PackedCandidateSite {
    __half2 a;
    __half2 b;
    __half2 c;
    __half2 d;
};

static_assert(sizeof(PackedCandidateSite) == 16, "PackedCandidateSite must be 16 bytes");

// Adam optimizer state (compatible with original AdamState struct)
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

// Helper functions for vector operations
__device__ __forceinline__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length(float2 v) {
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ float2 normalize(float2 v) {
    float len = length(v);
    return len > 1e-8f ? make_float2(v.x / len, v.y / len) : make_float2(1.0f, 0.0f);
}

__device__ __forceinline__ float clampf(float x, float a, float b) {
    return fmaxf(a, fminf(b, x));
}

__device__ __forceinline__ int clampi(int x, int a, int b) {
    return max(a, min(b, x));
}

__device__ __forceinline__ float2 clamp2(float2 v, float2 a, float2 b) {
    return make_float2(clampf(v.x, a.x, b.x), clampf(v.y, a.y, b.y));
}

__device__ __forceinline__ float3 clamp3(float3 v, float3 a, float3 b) {
    return make_float3(clampf(v.x, a.x, b.x), clampf(v.y, a.y, b.y), clampf(v.z, a.z, b.z));
}

__host__ __device__ __forceinline__ float3 site_color(const Site &s) {
    return make_float3(s.color_r, s.color_g, s.color_b);
}

__host__ __device__ __forceinline__ void site_set_color(Site &s, float3 c) {
    s.color_r = c.x;
    s.color_g = c.y;
    s.color_b = c.z;
}

__host__ __device__ __forceinline__ float2 site_aniso_dir(const Site &s) {
    return make_float2(s.aniso_dir_x, s.aniso_dir_y);
}

__host__ __device__ __forceinline__ void site_set_aniso_dir(Site &s, float2 v) {
    s.aniso_dir_x = v.x;
    s.aniso_dir_y = v.y;
}

__device__ __forceinline__ int candidate_index_for_pixel(
    int x, int y,
    int outW, int outH,
    int candW, int candH
) {
    int safeOutW = outW > 0 ? outW : 1;
    int safeOutH = outH > 0 ? outH : 1;
    int candX = (x * candW) / safeOutW;
    int candY = (y * candH) / safeOutH;
    candX = clampi(candX, 0, candW - 1);
    candY = clampi(candY, 0, candH - 1);
    return candY * candW + candX;
}

// RNG utilities
__device__ __forceinline__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__device__ __forceinline__ float rand01(uint32_t &state) {
    state = xorshift32(state);
    return float(state) * (1.0f / 4294967296.0f);
}

// Compute soft Voronoi distance
__device__ __forceinline__ float voronoi_dmix2(
    const Site &site,
    float2 uv,
    float inv_scale_sq
) {
    float2 diff = make_float2(uv.x - site.position.x, uv.y - site.position.y);

    // Anisotropic metric (SPD, det=1):
    // G = e^a * uu^T + e^-a * (I - uu^T)
    float diff2 = dot(diff, diff);
    float proj = dot(site_aniso_dir(site), diff);
    float proj2 = proj * proj;
    float perp2 = fmaxf(diff2 - proj2, 0.0f);

    float l1 = expf(site.log_aniso);
    float l2 = 1.0f / l1;
    float d2_aniso = l1 * proj2 + l2 * perp2;

    float d2_norm = d2_aniso * inv_scale_sq;
    float d2_safe = fmaxf(d2_norm, 1e-8f);
    float inv_scale = sqrtf(inv_scale_sq);
    float r_norm = site.radius * inv_scale;

    return (sqrtf(d2_safe) - r_norm);
}

// Hash function for tiled gradients
__device__ __forceinline__ uint32_t tile_hash(uint32_t key) {
    return key * 2654435761u;
}

// Safety checks
__device__ __forceinline__ bool is_bad(float v) {
    return isnan(v) || isinf(v) || fabsf(v) > 1.0e30f;
}

__device__ __forceinline__ bool is_bad2(float2 v) {
    return is_bad(v.x) || is_bad(v.y);
}

__device__ __forceinline__ bool is_bad3(float3 v) {
    return is_bad(v.x) || is_bad(v.y) || is_bad(v.z);
}

__device__ __forceinline__ float2 safe_dir(float2 v) {
    if (is_bad2(v)) {
        return make_float2(1.0f, 0.0f);
    }
    float len2 = v.x * v.x + v.y * v.y;
    if (len2 < 1.0e-12f) {
        return make_float2(1.0f, 0.0f);
    }
    float inv_len = rsqrtf(len2);
    return make_float2(v.x * inv_len, v.y * inv_len);
}
