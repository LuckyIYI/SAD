#include "../sad_common.metal"

kernel void jfaSeedPacked(
    texture2d<uint, access::read_write> cand0 [[texture(0)]],
    constant PackedInferenceSite *sites [[buffer(0)]],
    constant PackedSiteQuant &quant [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    constant uint &candDownscale [[buffer(3)]],
    constant uint &targetWidth [[buffer(4)]],
    constant uint &targetHeight [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= siteCount) return;

    uint width = cand0.get_width();
    uint height = cand0.get_height();
    float2 dims = float2(float(targetWidth - 1), float(targetHeight - 1));

    PackedInferenceSite packed = sites[gid];
    if (!packedActive(packed)) return;
    VoronoiSite site = decodePackedSite(packed, dims, quant);
    float invDownscale = 1.0f / max(float(candDownscale), 1.0f);
    int2 homePixel = int2(clamp(int(site.position.x * invDownscale), 0, int(width - 1)),
                          clamp(int(site.position.y * invDownscale), 0, int(height - 1)));

    uint4 existing = cand0.read(uint2(homePixel));
    existing.x = gid;
    cand0.write(existing, uint2(homePixel));
}

kernel void jfaFloodPacked(
    texture2d<uint, access::read>  inCand0 [[texture(0)]],
    texture2d<uint, access::write> outCand0 [[texture(1)]],
    constant PackedInferenceSite *sites [[buffer(0)]],
    constant PackedSiteQuant &quant [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    constant uint &stepSize [[buffer(3)]],
    constant float &inv_scale_sq [[buffer(4)]],
    constant uint &candDownscale [[buffer(5)]],
    constant uint &targetWidth [[buffer(6)]],
    constant uint &targetHeight [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = inCand0.get_width();
    uint height = inCand0.get_height();
    if (gid.x >= width || gid.y >= height) return;
    float2 dims = float2(float(targetWidth - 1), float(targetHeight - 1));

    float2 uv = (float2(gid) + 0.0f) * float(candDownscale);
    float2 maxUv = float2(float(targetWidth - 1), float(targetHeight - 1));
    uv = min(uv, maxUv);

    uint bestIdx[4] = { UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX };
    float bestD2[4] = { INFINITY, INFINITY, INFINITY, INFINITY };

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 samplePos = int2(gid) + int2(dx, dy) * int(stepSize);
            samplePos = clamp(samplePos, int2(0), int2(width - 1, height - 1));
            uint4 cand = inCand0.read(uint2(samplePos));

            insertClosest4Packed(bestIdx, bestD2, cand.x, uv, sites, quant, siteCount, inv_scale_sq, dims);
            insertClosest4Packed(bestIdx, bestD2, cand.y, uv, sites, quant, siteCount, inv_scale_sq, dims);
            insertClosest4Packed(bestIdx, bestD2, cand.z, uv, sites, quant, siteCount, inv_scale_sq, dims);
            insertClosest4Packed(bestIdx, bestD2, cand.w, uv, sites, quant, siteCount, inv_scale_sq, dims);
        }
    }

    outCand0.write(uint4(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]), gid);
}
