#include "../sad_common.metal"

kernel void computePruneScorePairs(
    const device VoronoiSite *sites [[buffer(0)]],
    const device float *removal_delta [[buffer(1)]],
    device uint2 *pairs [[buffer(2)]],              // (key, siteID)
    constant uint &siteCount [[buffer(3)]],
    constant float &deltaNorm [[buffer(4)]],
    constant uint &pairCount [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= pairCount) return;

    uint2 outPair = uint2(0xffffffffu, 0xffffffffu);
    if (gid < siteCount) {
        bool active = (sites[gid].position.x >= 0.0f);
        float delta = removal_delta[gid] * deltaNorm;
        if (active && isfinite(delta)) {
            // For positive floats, uint bit order is monotonic.
            outPair = uint2(as_type<uint>(max(delta, 0.0f)), gid);
        }
    }
    pairs[gid] = outPair;
}

kernel void writeSplitIndicesFromSorted(
    const device uint2 *sortedPairs [[buffer(0)]],  // (key, siteID), ascending (best first)
    device uint *splitIndices [[buffer(1)]],
    constant uint &numToSplit [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numToSplit) return;
    splitIndices[gid] = sortedPairs[gid].y;
}

kernel void pruneSitesByIndex(
    device VoronoiSite *sites [[buffer(0)]],
    const device uint *indices [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint idx = indices[gid];
    if (idx == 0xffffffffu) return;
    sites[idx].position = float2(-1.0f, -1.0f);
}
