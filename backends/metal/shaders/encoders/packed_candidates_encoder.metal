#include "../sad_common.metal"

kernel void updateCandidatesPacked(
    texture2d<uint, access::read>  inCand0 [[texture(0)]],
    texture2d<uint, access::read>  inCand1 [[texture(1)]],
    texture2d<uint, access::write> outCand0 [[texture(2)]],
    texture2d<uint, access::write> outCand1 [[texture(3)]],
    constant PackedInferenceSite *sites [[buffer(0)]],
    constant PackedSiteQuant &quant [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    constant uint &step [[buffer(3)]],
    constant float &inv_scale_sq [[buffer(4)]],
    constant uint &stepHigh [[buffer(5)]],
    constant float &radiusScale [[buffer(6)]],
    constant uint &radiusProbes [[buffer(7)]],
    constant uint &injectCount [[buffer(8)]],
    constant uint &candDownscale [[buffer(9)]],
    constant uint &targetWidth [[buffer(10)]],
    constant uint &targetHeight [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = inCand0.get_width();
    uint height = inCand0.get_height();
    if (gid.x >= width || gid.y >= height) return;
    float2 dims = float2(float(targetWidth - 1), float(targetHeight - 1));

    uint stepIndex = step & 0xffffu;
    uint jumpStep = step >> 16;
    if (jumpStep == 0u) {
        jumpStep = 1u;
    }
    uint fullStepIndex = (stepHigh << 16) | stepIndex;

    float2 uv = (float2(gid) + 0.0f) * float(candDownscale);
    float2 maxUv = float2(float(targetWidth - 1), float(targetHeight - 1));
    uv = min(uv, maxUv);

    uint4 self0 = inCand0.read(gid);
    uint4 self1 = inCand1.read(gid);

    int2 gi = int2(gid);
    int2 offsets[4] = {
        int2(-1,0), int2(1,0), int2(0,-1), int2(0,1)
    };

    thread uint bestIdx[8];
    thread float bestD2[8];
    for (uint i = 0; i < 8; ++i) {
        bestIdx[i] = 0xffffffffu;
        bestD2[i] = INFINITY;
    }

    mergeCandidates8Packed(bestIdx, bestD2, self0, self1, uv, sites, quant, siteCount,
                           inv_scale_sq, dims);

    for (uint i = 0; i < 4; ++i) {
        int2 p = clamp(gi + offsets[i] * int(jumpStep), int2(0), int2(width-1, height-1));
        uint2 pu = uint2(p);
        uint4 neigh1 = inCand1.read(pu);
        mergeCandidates8Packed(bestIdx, bestD2, inCand0.read(pu), neigh1, uv,
                               sites, quant, siteCount, inv_scale_sq, dims);
    }

    float rad = radiusScale * (float(width) / 1024.0f);
    uint state = (gid.x * 73856093u) ^ (gid.y * 19349663u) ^ ((stepIndex + jumpStep) * 83492791u);
    for (uint r = 0; r < radiusProbes; ++r) {
        float a = rand01(state) * 6.2831853f;
        int2 dp = int2(cos(a) * rad, sin(a) * rad);
        int2 p = clamp(gi + dp, int2(0), int2(width-1, height-1));
        uint2 pu = uint2(p);
        if (gid.x == pu.x && gid.y == pu.y) { continue; }
        uint4 probe1 = inCand1.read(pu);
        mergeCandidates8Packed(bestIdx, bestD2, inCand0.read(pu), probe1, uv,
                               sites, quant, siteCount, inv_scale_sq, dims);
    }

    uint injectState = (gid.x * 1664525u) ^ (gid.y * 1013904223u) ^ (fullStepIndex * 374761393u);
    for (uint i = 0; i < injectCount; ++i) {
        injectState = xorshift32(injectState + i);
        uint cand = injectState % siteCount;
        insertClosest8Packed(bestIdx, bestD2, cand, uv, sites, quant, siteCount,
                             inv_scale_sq, dims);
    }

    outCand0.write(uint4(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]), gid);
    outCand1.write(uint4(bestIdx[4], bestIdx[5], bestIdx[6], bestIdx[7]), gid);
}
