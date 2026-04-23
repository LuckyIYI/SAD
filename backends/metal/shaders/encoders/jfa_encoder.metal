#include "../sad_common.metal"

kernel void jfaSeed(
    texture2d<uint, access::read_write> cand0 [[texture(0)]],
    constant VoronoiSite *sites [[buffer(0)]],
    constant uint &siteCount [[buffer(1)]],
    constant uint &candDownscale [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= siteCount) return;

    VoronoiSite site = sites[gid];

    // Skip inactive sites
    if (site.position.x < 0.0f) return;

    uint width = cand0.get_width();
    uint height = cand0.get_height();

    // Clamp position to image bounds
    float invDownscale = 1.0f / max(float(candDownscale), 1.0f);
    int2 homePixel = int2(clamp(int(site.position.x * invDownscale), 0, int(width - 1)),
                          clamp(int(site.position.y * invDownscale), 0, int(height - 1)));

    // Overwrite only the first slot to preserve any existing candidates.
    uint4 existing = cand0.read(uint2(homePixel));
    existing.x = gid;
    cand0.write(existing, uint2(homePixel));
}

kernel void jfaClearCandidates(
    texture2d<uint, access::write> cand0 [[texture(0)]],
    texture2d<uint, access::write> cand1 [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = cand0.get_width();
    uint height = cand0.get_height();
    if (gid.x >= width || gid.y >= height) return;
    uint4 empty = uint4(UINT_MAX);
    cand0.write(empty, gid);
    cand1.write(empty, gid);
}

// JFA Flood: One pass with given step size (stateless - no pre-sort needed)
kernel void jfaFlood(
    texture2d<uint, access::read>  inCand0 [[texture(0)]],
    texture2d<uint, access::write> outCand0 [[texture(1)]],
    constant VoronoiSite *sites [[buffer(0)]],
    constant uint &siteCount [[buffer(1)]],
    constant uint &stepSize [[buffer(2)]],
    constant float &inv_scale_sq [[buffer(3)]],
    constant uint &candDownscale [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = inCand0.get_width();
    uint height = inCand0.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = (float2(gid) + 0.0f) * float(candDownscale);
    float2 maxUv = float2(float(width * candDownscale - 1), float(height * candDownscale - 1));
    uv = min(uv, maxUv);

    // Start with empty list
    uint bestIdx[4] = { UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX };
    float bestD2[4] = { INFINITY, INFINITY, INFINITY, INFINITY };

    // Sample 3x3 grid at step offset (9 samples including self)
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 samplePos = int2(gid) + int2(dx, dy) * int(stepSize);

            // Clamp to image bounds
            samplePos = clamp(samplePos, int2(0), int2(width - 1, height - 1));

            // Read candidates at this position
            uint4 cand = inCand0.read(uint2(samplePos));

            // Try to insert each candidate
            insertClosest4(bestIdx, bestD2, cand.x, uv, sites, siteCount, inv_scale_sq);
            insertClosest4(bestIdx, bestD2, cand.y, uv, sites, siteCount, inv_scale_sq);
            insertClosest4(bestIdx, bestD2, cand.z, uv, sites, siteCount, inv_scale_sq);
            insertClosest4(bestIdx, bestD2, cand.w, uv, sites, siteCount, inv_scale_sq);
        }
    }

    // Write result
    outCand0.write(uint4(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]), gid);
}
