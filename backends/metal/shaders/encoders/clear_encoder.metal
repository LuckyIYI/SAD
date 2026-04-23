#include "../sad_common.metal"

kernel void clearAtomicBuffer(
    device float *buffer [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    buffer[gid] = 0.0f;
}
