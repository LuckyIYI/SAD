#include "../sad_common.metal"

kernel void computePSNR(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> target [[texture(1)]],
    texture2d<float, access::read> mask [[texture(2)]],
    device atomic_float *mse_accum [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = rendered.get_width();
    uint height = rendered.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float mask_val = mask.read(gid).r;
    if (mask_val <= 0.0f) return;

    float3 r = rendered.read(gid).rgb;
    float3 t = target.read(gid).rgb;
    float3 diff = r - t;
    float pixel_mse = dot(diff, diff);

    atomic_fetch_add_explicit(mse_accum, pixel_mse, memory_order_relaxed);
}
