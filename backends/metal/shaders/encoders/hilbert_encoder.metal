#include "../sad_common.metal"

inline uint hilbertIndex(uint x, uint y, uint bits) {
    uint xi = x;
    uint yi = y;
    uint index = 0u;
    uint mask = (bits >= 32u) ? 0xffffffffu : ((1u << bits) - 1u);
    for (int i = int(bits) - 1; i >= 0; --i) {
        uint shift = uint(i);
        uint rx = (xi >> shift) & 1u;
        uint ry = (yi >> shift) & 1u;
        uint d = (3u * rx) ^ ry;
        index |= d << (2u * shift);
        if (ry == 0u) {
            if (rx == 1u) {
                xi = mask - xi;
                yi = mask - yi;
            }
            uint tmp = xi;
            xi = yi;
            yi = tmp;
        }
    }
    return index;
}

kernel void buildHilbertPairs(
    constant VoronoiSite *sites [[buffer(0)]],
    device uint2 *pairs [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    constant uint &paddedCount [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    constant uint &height [[buffer(5)]],
    constant uint &bits [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= paddedCount) return;
    if (gid < siteCount) {
        float2 pos = sites[gid].position;
        int px = clamp(int(pos.x), 0, int(width - 1u));
        int py = clamp(int(pos.y), 0, int(height - 1u));
        uint key = hilbertIndex(uint(px), uint(py), bits);
        pairs[gid] = uint2(key, gid);
    } else {
        pairs[gid] = uint2(0xffffffffu, 0u);
    }
}

kernel void writeHilbertOrder(
    const device uint2 *pairs [[buffer(0)]],
    device uint *order [[buffer(1)]],
    device uint *pos [[buffer(2)]],
    constant uint &siteCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= siteCount) return;
    uint siteIdx = pairs[gid].y;
    order[gid] = siteIdx;
    pos[siteIdx] = gid;
}
