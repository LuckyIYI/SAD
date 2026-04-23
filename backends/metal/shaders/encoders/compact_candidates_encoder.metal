#include "../sad_common.metal"

kernel void updateCandidatesCompact(
    texture2d<uint, access::read>  inCand0 [[texture(0)]],
    texture2d<uint, access::read>  inCand1 [[texture(1)]],
    texture2d<uint, access::write> outCand0 [[texture(2)]],
    texture2d<uint, access::write> outCand1 [[texture(3)]],
    constant PackedCandidateSite *sites [[buffer(0)]],
    constant uint &siteCount [[buffer(1)]],
    constant uint &step [[buffer(2)]],
    constant float &inv_scale_sq [[buffer(3)]],
    constant uint &stepHigh [[buffer(4)]],
    constant float &radiusScale [[buffer(5)]],
    constant uint &radiusProbes [[buffer(6)]],
    constant uint &injectCount [[buffer(7)]],
    device const uint *hilbertOrder [[buffer(8)]],
    device const uint *hilbertPos [[buffer(9)]],
    constant uint &hilbertProbeCount [[buffer(10)]],
    constant uint &hilbertWindow [[buffer(11)]],
    constant uint &candDownscale [[buffer(12)]],
    constant uint &targetWidth [[buffer(13)]],
    constant uint &targetHeight [[buffer(14)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = inCand0.get_width();
    uint height = inCand0.get_height();
    if (gid.x >= width || gid.y >= height) return;

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

    mergeCandidates8CandidatePacked(bestIdx, bestD2, self0, self1, uv,
                                    sites, siteCount, inv_scale_sq);

    for (uint i = 0; i < 4; ++i) {
        int2 p = clamp(gi + offsets[i] * int(jumpStep), int2(0), int2(width-1, height-1));
        uint2 pu = uint2(p);
        uint4 neigh1 = inCand1.read(pu);
        mergeCandidates8CandidatePacked(bestIdx, bestD2, inCand0.read(pu), neigh1,
                                        uv, sites, siteCount, inv_scale_sq);
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
        mergeCandidates8CandidatePacked(bestIdx, bestD2, inCand0.read(pu), probe1,
                                        uv, sites, siteCount, inv_scale_sq);
    }

    if (hilbertProbeCount > 0u && hilbertWindow > 0u) {
        uint bestCand = bestIdx[0];
        if (bestCand < siteCount) {
            uint pos = hilbertPos[bestCand];
            uint span = hilbertWindow * 2u + 1u;
            uint hState = (gid.x * 2654435761u) ^ (gid.y * 1597334677u) ^ (fullStepIndex * 374761393u);
            for (uint i = 0; i < hilbertProbeCount; ++i) {
                hState = xorshift32(hState + i);
                int offset = int(hState % span) - int(hilbertWindow);
                int idx = clamp(int(pos) + offset, 0, int(siteCount - 1));
                uint cand = hilbertOrder[idx];
                insertClosest8CandidatePacked(bestIdx, bestD2, cand, uv, sites, siteCount, inv_scale_sq);
            }
        }
    }

    uint injectState = (gid.x * 1664525u) ^ (gid.y * 1013904223u) ^ (fullStepIndex * 374761393u);
    for (uint i = 0; i < injectCount; ++i) {
        injectState = xorshift32(injectState + i);
        uint cand = injectState % siteCount;
        insertClosest8CandidatePacked(bestIdx, bestD2, cand, uv, sites, siteCount, inv_scale_sq);
    }

    outCand0.write(uint4(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]), gid);
    outCand1.write(uint4(bestIdx[4], bestIdx[5], bestIdx[6], bestIdx[7]), gid);
}
