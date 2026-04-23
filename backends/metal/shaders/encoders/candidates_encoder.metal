#include "../sad_common.metal"

kernel void initCandidates(
    texture2d<uint, access::write> outCand0 [[texture(0)]],
    texture2d<uint, access::write> outCand1 [[texture(1)]],
    constant uint &siteCount [[buffer(0)]],
    constant uint &seed [[buffer(1)]],
    constant bool &perPixelMode [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = outCand0.get_width();
    uint height = outCand0.get_height();
    if (gid.x >= width || gid.y >= height) return;

    uint idx[8];

    if (perPixelMode && siteCount == width * height) {
        // Perfect initialization: pixel (x,y) -> site index = y * width + x
        uint selfIdx = gid.y * width + gid.x;

        // Put self as first candidate, then neighbors
        idx[0] = selfIdx;

        // Add 8-connected neighbors
        int2 neighbors[7] = {
            int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1),
            int2(-1, -1), int2(1, -1), int2(-1, 1)
        };

        for (uint i = 0; i < 7; ++i) {
            int2 npos = int2(gid) + neighbors[i];
            npos = clamp(npos, int2(0), int2(width-1, height-1));
            idx[i + 1] = uint(npos.y) * width + uint(npos.x);
        }
    } else {
        // Random initialization
        uint state = seed ^ (gid.x * 1973u) ^ (gid.y * 9277u);
        for (uint i = 0; i < 8; ++i) {
            state = xorshift32(state + i);
            idx[i] = state % siteCount;
        }
    }

    outCand0.write(uint4(idx[0], idx[1], idx[2], idx[3]), gid);
    outCand1.write(uint4(idx[4], idx[5], idx[6], idx[7]), gid);
}
