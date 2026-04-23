#include <metal_stdlib>
using namespace metal;

// Radix sort specialized for `uint2` items, sorting by `.x` (key).
//
// This is an LSD radix sort over 8-bit digits. Each pass is stable.
// Implementation strategy:
// 1) Per-block histogram into `histFlat[bin * gridSize + blockId]`.
// 2) Exclusive prefix scan over the *linearized* histogram array (bin-major layout),
//    which yields correct base offsets per (bin, block) cell including per-bin bases.
// 3) Stable scatter per block using counting sort within each chunk.

namespace Radix {
constant uint kBlockSize = 256u;
constant uint kGrain = 4u;                 // 256 threads * 4 = 1024 elements per block
constant uint kBuckets = 256u;             // 8-bit digits
constant uint kElementsPerBlock = 1024u;

inline uint bucketForKey(uint key, uint shift) {
    return (key >> shift) & 0xFFu;
}

// Blelloch exclusive scan for 256 elements in threadgroup memory.
inline void exclusiveScan256UInt(threadgroup uint* shared, uint tid, thread uint& totalOut) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce).
    for (uint offset = 1; offset < kBlockSize; offset <<= 1) {
        uint ai = offset * (2 * tid + 1) - 1;
        uint bi = offset * (2 * tid + 2) - 1;
        if (bi < kBlockSize) {
            shared[bi] += shared[ai];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    totalOut = shared[kBlockSize - 1];

    // Down-sweep.
    if (tid == 0) {
        shared[kBlockSize - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = kBlockSize >> 1; offset > 0; offset >>= 1) {
        uint ai = offset * (2 * tid + 1) - 1;
        uint bi = offset * (2 * tid + 2) - 1;
        if (bi < kBlockSize) {
            uint t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Inclusive prefix sum for 256 elements
inline uint inclusiveScan256UInt(threadgroup uint* shared, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < kBlockSize; stride <<= 1) {
        uint val = shared[tid];
        uint other = (tid >= stride) ? shared[tid - stride] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = val + other;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return shared[tid];
}

} // namespace Radix

kernel void radixHistogramUInt2(
    device const uint2* input [[buffer(0)]],
    device uint* histFlat [[buffer(1)]],           // length = 256 * gridSize
    constant uint& paddedCount [[buffer(2)]],
    constant uint& shift [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    using namespace Radix;

    const uint gridSize = (paddedCount + kElementsPerBlock - 1u) / kElementsPerBlock;
    if (groupId >= gridSize) return;

    threadgroup atomic_uint localHist[kBuckets];

    // Clear local histogram (256 bins).
    if (localId < kBuckets) {
        atomic_store_explicit(&localHist[localId], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = groupId * kElementsPerBlock;
    for (uint i = 0; i < kGrain; ++i) {
        uint idx = base + uint(localId) + i * kBlockSize;
        if (idx >= paddedCount) break;
        uint key = input[idx].x;
        uint bin = bucketForKey(key, shift);
        atomic_fetch_add_explicit(&localHist[bin], 1u, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store histogram in bin-major layout: histFlat[bin * gridSize + groupId]
    if (localId < kBuckets) {
        histFlat[uint(localId) * gridSize + groupId] = atomic_load_explicit(&localHist[localId], memory_order_relaxed);
    }
}

// Exclusive scan each 256-value segment of histFlat (in-place), emitting one block sum per segment.
kernel void radixScanHistogramBlocks(
    device uint* histFlat [[buffer(0)]],      // length = 256 * gridSize
    device uint* blockSums [[buffer(1)]],     // length = ceil(histLength / 256)
    constant uint& paddedCount [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    using namespace Radix;

    const uint gridSize = (paddedCount + kElementsPerBlock - 1u) / kElementsPerBlock;
    const uint histLength = gridSize * kBuckets;
    const uint numBlocks = (histLength + kBlockSize - 1u) / kBlockSize;
    if (groupId >= numBlocks) return;

    const uint baseIndex = groupId * kBlockSize;
    threadgroup uint shared[kBlockSize];

    uint idx = baseIndex + uint(localId);
    uint value = (idx < histLength) ? histFlat[idx] : 0u;
    shared[localId] = value;

    uint total;
    exclusiveScan256UInt(shared, uint(localId), total);
    uint scanned = shared[localId];

    if (idx < histLength) {
        histFlat[idx] = scanned;
    }
    if (localId == kBlockSize - 1) {
        blockSums[groupId] = scanned + value;
    }
}

// Exclusive scan blockSums in-place. One threadgroup (256 threads).
// Handles up to 256*256 = 65536 block sums using a two-level approach.
kernel void radixExclusiveScanBlockSums(
    device uint* blockSums [[buffer(0)]],
    constant uint& paddedCount [[buffer(1)]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    using namespace Radix;

    const uint gridSize = (paddedCount + kElementsPerBlock - 1u) / kElementsPerBlock;
    const uint histLength = gridSize * kBuckets;
    const uint num = (histLength + kBlockSize - 1u) / kBlockSize;

    // Two-level scan: each thread handles ceil(num/256) elements
    const uint elemsPerThread = (num + kBlockSize - 1u) / kBlockSize;

    threadgroup uint shared[kBlockSize];

    // Phase 1: Each thread sums its chunk
    uint localSum = 0u;
    uint baseIdx = uint(localId) * elemsPerThread;
    for (uint i = 0; i < elemsPerThread; ++i) {
        uint idx = baseIdx + i;
        if (idx < num) {
            localSum += blockSums[idx];
        }
    }
    shared[localId] = localSum;

    // Phase 2: Scan the per-thread sums
    uint total;
    exclusiveScan256UInt(shared, uint(localId), total);
    uint threadBase = shared[localId];

    // Phase 3: Write back exclusive prefixes for each element
    uint running = threadBase;
    for (uint i = 0; i < elemsPerThread; ++i) {
        uint idx = baseIdx + i;
        if (idx < num) {
            uint val = blockSums[idx];
            blockSums[idx] = running;
            running += val;
        }
    }
}

// Add per-segment offsets to histFlat (in-place): histFlat += blockSums[groupId].
kernel void radixApplyOffsets(
    device uint* histFlat [[buffer(0)]],
    device const uint* blockSums [[buffer(1)]],
    constant uint& paddedCount [[buffer(2)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    using namespace Radix;

    const uint gridSize = (paddedCount + kElementsPerBlock - 1u) / kElementsPerBlock;
    const uint histLength = gridSize * kBuckets;
    const uint numBlocks = (histLength + kBlockSize - 1u) / kBlockSize;
    if (groupId >= numBlocks) return;

    const uint baseIndex = groupId * kBlockSize;
    uint idx = baseIndex + uint(localId);
    if (idx < histLength) {
        uint off = blockSums[groupId];
        histFlat[idx] += off;
    }
}

// Stable scatter: preserves original order within each bin
kernel void radixScatterUInt2(
    device const uint2* input [[buffer(0)]],
    device uint2* output [[buffer(1)]],
    device const uint* offsetsFlat [[buffer(2)]],    // histFlat after applyOffsets
    constant uint& paddedCount [[buffer(3)]],
    constant uint& shift [[buffer(4)]],
    uint groupId [[threadgroup_position_in_grid]],
    ushort localId [[thread_position_in_threadgroup]]
) {
    using namespace Radix;

    const uint gridSize = (paddedCount + kElementsPerBlock - 1u) / kElementsPerBlock;
    if (groupId >= gridSize) return;

    const uint blockBase = groupId * kElementsPerBlock;

    // Global offset for each bin (initialized from offsetsFlat, updated per chunk)
    threadgroup uint globalBinOffset[kBuckets];

    // Temporary for counting
    threadgroup uint binCounts[kBuckets];
    threadgroup ushort tgBins[kBlockSize];
    threadgroup uint tgRanks[kBlockSize];

    // Initialize global offsets from the scanned histogram
    if (localId < kBuckets) {
        globalBinOffset[localId] = offsetsFlat[uint(localId) * gridSize + groupId];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each chunk of 256 items
    for (uint chunk = 0; chunk < kGrain; ++chunk) {
        uint idx = blockBase + chunk * kBlockSize + uint(localId);

        // Load item (use sentinel for out-of-range)
        uint2 val = (idx < paddedCount) ? input[idx] : uint2(0xffffffffu, 0xfffffffeu);
        bool isValid = (val.y != 0xfffffffeu);

        uint bin = bucketForKey(val.x, shift);
        tgBins[localId] = isValid ? ushort(bin) : ushort(0xffffu);

        // Clear bin counts
        if (localId < kBuckets) {
            binCounts[localId] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Count items per bin (atomics for counting)
        if (isValid) {
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&binCounts[bin], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute rank within bin: count how many items with SMALLER thread ID have the same bin
        // This is O(n) per thread but guarantees stability
        uint rankInBin = 0;
        if (isValid) {
            for (uint i = 0; i < uint(localId); ++i) {
                if (tgBins[i] == ushort(bin)) {
                    rankInBin++;
                }
            }
        }
        tgRanks[localId] = rankInBin;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write to output using global offset + rank
        if (isValid) {
            uint dst = globalBinOffset[bin] + rankInBin;
            if (dst < paddedCount) {
                output[dst] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update global offsets for next chunk
        if (localId < kBuckets) {
            globalBinOffset[localId] += binCounts[localId];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
