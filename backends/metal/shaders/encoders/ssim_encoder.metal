#include "../sad_common.metal"

kernel void computeSSIM(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::read> target [[texture(1)]],
    texture2d<float, access::read> mask [[texture(2)]],
    device atomic_float *ssim_accum [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = rendered.get_width();
    uint height = rendered.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float mask_val = mask.read(gid).r;
    if (mask_val <= 0.0f) return;

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
    float ssim = (A * B) / denom;

    atomic_fetch_add_explicit(ssim_accum, ssim, memory_order_relaxed);
}
