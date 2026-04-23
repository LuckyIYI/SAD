#include "../sad_common.metal"

inline float init_sample_luma(texture2d<float, access::read> target,
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

kernel void initGradientWeighted(
    device VoronoiSite *sites [[buffer(0)]],
    constant uint &numSites [[buffer(1)]],
    device atomic_uint &seedCounter [[buffer(2)]],
    texture2d<float, access::read> target [[texture(0)]],
    texture2d<float, access::read> mask [[texture(1)]],
    constant float &gradThreshold [[buffer(3)]],
    constant uint &maxAttempts [[buffer(4)]],
    constant float &init_log_tau [[buffer(5)]],
    constant float &init_radius [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numSites) return;

    uint width = target.get_width();
    uint height = target.get_height();

    float2 acceptedPos = float2(-1.0, -1.0);
    float3 acceptedColor = float3(0.5);
    float acceptedGraident = 0.0;

    // Try quasi-random positions; accept the first that exceeds gradThreshold,
    // and leave the site inactive if none exceed it.
    for (uint attempt = 0; attempt < maxAttempts; ++attempt) {
        // R2 low-discrepancy sequence with atomic counter
        uint seed = atomic_fetch_add_explicit(&seedCounter, 1, memory_order_relaxed);

        const float g = 1.32471795724474602596;
        const float a1 = 1.0 / g;
        const float a2 = 1.0 / (g * g);

        float2 uv = fract(float2(0.5) + float2(seed) * float2(a1, a2));
        float2 pos = uv * float2(width - 1, height - 1);
        uint2 ipos = uint2(pos);

        float mask_val = mask.read(ipos).r;
        if (mask_val <= 0.0f) {
            continue;
        }

        // Compute gradient at this position (Sobel)
        float TL = init_sample_luma(target, ipos, int(width), int(height), -1, -1);
        float T  = init_sample_luma(target, ipos, int(width), int(height),  0, -1);
        float TR = init_sample_luma(target, ipos, int(width), int(height),  1, -1);
        float L  = init_sample_luma(target, ipos, int(width), int(height), -1,  0);
        float R  = init_sample_luma(target, ipos, int(width), int(height),  1,  0);
        float BL = init_sample_luma(target, ipos, int(width), int(height), -1,  1);
        float B  = init_sample_luma(target, ipos, int(width), int(height),  0,  1);
        float BR = init_sample_luma(target, ipos, int(width), int(height),  1,  1);

        float Gx = (-TL + TR - 2.0*L + 2.0*R - BL + BR) / 4.0;
        float Gy = (-TL - 2.0*T - TR + BL + 2.0*B + BR) / 4.0;
        float grad = sqrt(Gx*Gx + Gy*Gy) * mask_val;

        // Accept if gradient is high enough
        if (grad > gradThreshold) {
            acceptedPos = pos;
            acceptedColor = target.read(ipos).rgb;
            acceptedGraident = grad;
            break;
        }
    }

    // Initialize site
    sites[gid].position = acceptedPos;
    sites[gid].log_tau = init_log_tau + pow(acceptedGraident, 0.25) * 2.0;
    sites[gid].radius = init_radius;
    sites[gid].color = acceptedColor;
    sites[gid].aniso_dir = float2(1.0, 0.0);
    sites[gid].log_aniso = 0.0;
}
