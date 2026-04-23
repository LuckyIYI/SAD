#include "../sad_common.metal"

kernel void diffuseTauGradientsAtSite(
    texture2d<uint, access::read> candidates0 [[texture(0)]],
    texture2d<uint, access::read> candidates1 [[texture(1)]],
    const device VoronoiSite *sites [[buffer(0)]],
    const device float *grad_raw [[buffer(1)]],
    const device float *grad_in [[buffer(2)]],
    device float *grad_out [[buffer(3)]],
    constant uint &siteCount [[buffer(4)]],
    constant float &lambda [[buffer(5)]],
    constant uint &candDownscale [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= siteCount) return;
    if (lambda <= 0.0f) {
        grad_out[gid] = grad_in[gid];
        return;
    }

    VoronoiSite site = sites[gid];
    if (site.position.x < 0.0f) {
        grad_out[gid] = grad_in[gid];
        return;
    }

    uint width = candidates0.get_width();
    uint height = candidates0.get_height();
    float invDownscale = 1.0f / max(float(candDownscale), 1.0f);
    int2 p = int2(site.position * invDownscale + float2(0.5f));
    p = clamp(p, int2(0), int2(int(width) - 1, int(height) - 1));
    uint2 uv = uint2(p);

    uint4 c0 = candidates0.read(uv);
    uint4 c1 = candidates1.read(uv);
    uint candIds[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};

    float sum = 0.0f;
    uint count = 0;
    for (uint i = 0; i < 8; ++i) {
        uint n = candIds[i];
        if (n == 0xffffffffu || n >= siteCount || n == gid) continue;

        bool seen = false;
        for (uint j = 0; j < i; ++j) {
            if (candIds[j] == n) {
                seen = true;
                break;
            }
        }
        if (seen) continue;

        if (sites[n].position.x < 0.0f) continue;
        sum += grad_in[n];
        count += 1;
    }

    float diag = 1.0f + lambda * float(count);
    grad_out[gid] = (grad_raw[gid] + lambda * sum) / diag;
}
