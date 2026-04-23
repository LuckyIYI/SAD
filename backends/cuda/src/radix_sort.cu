#include "sad_common.cuh"
#include <cuda_runtime.h>

// CUDA Radix Sort for uint2 (key-value pairs)
// Sorts by .x (key) in ascending order
// Based on the Metal implementation strategy:
// 1) Per-block histogram
// 2) Exclusive prefix scan over histogram
// 3) Stable scatter using counting sort

namespace RadixSort {

constexpr uint32_t BLOCK_SIZE = 256;
constexpr uint32_t GRAIN = 4;  // 256 * 4 = 1024 elements per block
constexpr uint32_t BUCKETS = 256;  // 8-bit digits
constexpr uint32_t ELEMENTS_PER_BLOCK = 1024;

__device__ __forceinline__ uint32_t bucketForKey(uint32_t key, uint32_t shift) {
    return (key >> shift) & 0xFFu;
}

// Blelloch exclusive scan for 256 elements in shared memory
__device__ void exclusiveScan256(uint32_t* shared, uint32_t tid, uint32_t& totalOut) {
    __syncthreads();
    
    // Up-sweep (reduce)
    for (uint32_t offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        uint32_t ai = offset * (2 * tid + 1) - 1;
        uint32_t bi = offset * (2 * tid + 2) - 1;
        if (bi < BLOCK_SIZE) {
            shared[bi] += shared[ai];
        }
        __syncthreads();
    }
    
    totalOut = shared[BLOCK_SIZE - 1];
    
    // Down-sweep
    if (tid == 0) {
        shared[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();
    
    for (uint32_t offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        uint32_t ai = offset * (2 * tid + 1) - 1;
        uint32_t bi = offset * (2 * tid + 2) - 1;
        if (bi < BLOCK_SIZE) {
            uint32_t t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
        __syncthreads();
    }
}

// 1) Histogram kernel: compute per-block histograms
__global__ void histogramKernel(
    const uint2* __restrict__ input,
    uint32_t* __restrict__ histFlat,  // size: 256 * gridSize
    uint32_t paddedCount,
    uint32_t shift
) {
    const uint32_t gridSize = (paddedCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    const uint32_t groupId = blockIdx.x;
    const uint32_t localId = threadIdx.x;
    
    if (groupId >= gridSize) return;
    
    __shared__ uint32_t localHist[BUCKETS];
    
    // Clear local histogram
    if (localId < BUCKETS) {
        localHist[localId] = 0;
    }
    __syncthreads();
    
    // Count elements
    const uint32_t base = groupId * ELEMENTS_PER_BLOCK;
    for (uint32_t i = 0; i < GRAIN; ++i) {
        uint32_t idx = base + localId + i * BLOCK_SIZE;
        if (idx >= paddedCount) break;
        
        uint32_t key = input[idx].x;
        uint32_t bin = bucketForKey(key, shift);
        atomicAdd(&localHist[bin], 1);
    }
    
    __syncthreads();
    
    // Store histogram in bin-major layout: histFlat[bin * gridSize + groupId]
    if (localId < BUCKETS) {
        histFlat[localId * gridSize + groupId] = localHist[localId];
    }
}

// 2) Scan histogram blocks (in-place), emitting block sums
__global__ void scanHistogramBlocksKernel(
    uint32_t* __restrict__ histFlat,     // size: 256 * gridSize
    uint32_t* __restrict__ blockSums,    // size: ceil(histLength / 256)
    uint32_t paddedCount
) {
    const uint32_t gridSize = (paddedCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    const uint32_t histLength = gridSize * BUCKETS;
    const uint32_t numBlocks = (histLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint32_t groupId = blockIdx.x;
    const uint32_t localId = threadIdx.x;
    
    if (groupId >= numBlocks) return;
    
    const uint32_t baseIndex = groupId * BLOCK_SIZE;
    __shared__ uint32_t shared[BLOCK_SIZE];
    
    uint32_t idx = baseIndex + localId;
    uint32_t value = (idx < histLength) ? histFlat[idx] : 0;
    shared[localId] = value;
    
    uint32_t total;
    exclusiveScan256(shared, localId, total);
    uint32_t scanned = shared[localId];
    
    if (idx < histLength) {
        histFlat[idx] = scanned;
    }
    if (localId == BLOCK_SIZE - 1) {
        blockSums[groupId] = scanned + value;
    }
}

// 3) Exclusive scan block sums (single block)
__global__ void scanBlockSumsKernel(
    uint32_t* __restrict__ blockSums,
    uint32_t paddedCount
) {
    const uint32_t gridSize = (paddedCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    const uint32_t histLength = gridSize * BUCKETS;
    const uint32_t num = (histLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint32_t localId = threadIdx.x;
    
    // Two-level scan: each thread handles ceil(num/256) elements
    const uint32_t elemsPerThread = (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    __shared__ uint32_t shared[BLOCK_SIZE];
    
    // Phase 1: Each thread sums its chunk
    uint32_t localSum = 0;
    uint32_t baseIdx = localId * elemsPerThread;
    for (uint32_t i = 0; i < elemsPerThread; ++i) {
        uint32_t idx = baseIdx + i;
        if (idx < num) {
            localSum += blockSums[idx];
        }
    }
    shared[localId] = localSum;
    
    // Phase 2: Scan the per-thread sums
    uint32_t total;
    exclusiveScan256(shared, localId, total);
    uint32_t threadBase = shared[localId];
    
    // Phase 3: Write back exclusive prefixes for each element
    uint32_t running = threadBase;
    for (uint32_t i = 0; i < elemsPerThread; ++i) {
        uint32_t idx = baseIdx + i;
        if (idx < num) {
            uint32_t val = blockSums[idx];
            blockSums[idx] = running;
            running += val;
        }
    }
}

// 4) Apply offsets to histFlat (in-place)
__global__ void applyOffsetsKernel(
    uint32_t* __restrict__ histFlat,
    const uint32_t* __restrict__ blockSums,
    uint32_t paddedCount
) {
    const uint32_t gridSize = (paddedCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    const uint32_t histLength = gridSize * BUCKETS;
    const uint32_t numBlocks = (histLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint32_t groupId = blockIdx.x;
    const uint32_t localId = threadIdx.x;
    
    if (groupId >= numBlocks) return;
    
    const uint32_t baseIndex = groupId * BLOCK_SIZE;
    uint32_t idx = baseIndex + localId;
    if (idx < histLength) {
        uint32_t off = blockSums[groupId];
        histFlat[idx] += off;
    }
}

// 5) Stable scatter using counting sort
__global__ void scatterKernel(
    const uint2* __restrict__ input,
    uint2* __restrict__ output,
    const uint32_t* __restrict__ offsetsFlat,  // histFlat after applyOffsets
    uint32_t paddedCount,
    uint32_t shift
) {
    const uint32_t gridSize = (paddedCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    const uint32_t groupId = blockIdx.x;
    const uint32_t localId = threadIdx.x;
    
    if (groupId >= gridSize) return;
    
    const uint32_t blockBase = groupId * ELEMENTS_PER_BLOCK;
    
    __shared__ uint32_t globalBinOffset[BUCKETS];
    __shared__ uint32_t binCounts[BUCKETS];
    __shared__ uint16_t tgBins[BLOCK_SIZE];
    __shared__ uint32_t tgRanks[BLOCK_SIZE];
    
    // Initialize global offsets from the scanned histogram
    if (localId < BUCKETS) {
        globalBinOffset[localId] = offsetsFlat[localId * gridSize + groupId];
    }
    __syncthreads();
    
    // Process each chunk of 256 items
    for (uint32_t chunk = 0; chunk < GRAIN; ++chunk) {
        uint32_t idx = blockBase + chunk * BLOCK_SIZE + localId;
        
        // Load item (use sentinel for out-of-range)
        uint2 val = (idx < paddedCount) ? input[idx] : make_uint2(0xFFFFFFFFu, 0xFFFFFFFEu);
        bool isValid = (val.y != 0xFFFFFFFEu);
        
        uint32_t bin = bucketForKey(val.x, shift);
        tgBins[localId] = isValid ? (uint16_t)bin : 0xFFFFu;
        
        // Clear bin counts
        if (localId < BUCKETS) {
            binCounts[localId] = 0;
        }
        __syncthreads();
        
        // Count items per bin
        if (isValid) {
            atomicAdd(&binCounts[bin], 1);
        }
        __syncthreads();
        
        // Compute rank within bin: count how many items with SMALLER thread ID have the same bin
        // This is O(n) per thread but guarantees stability
        uint32_t rankInBin = 0;
        if (isValid) {
            for (uint32_t i = 0; i < localId; ++i) {
                if (tgBins[i] == (uint16_t)bin) {
                    rankInBin++;
                }
            }
        }
        tgRanks[localId] = rankInBin;
        __syncthreads();
        
        // Write to output using global offset + rank
        if (isValid) {
            uint32_t dst = globalBinOffset[bin] + rankInBin;
            if (dst < paddedCount) {
                output[dst] = val;
            }
        }
        __syncthreads();
        
        // Update global offsets for next chunk
        if (localId < BUCKETS) {
            globalBinOffset[localId] += binCounts[localId];
        }
        __syncthreads();
    }
}

} // namespace RadixSort

// C++ wrapper for radix sort
extern "C" {

void launchRadixSortUInt2(
    uint2* data,
    uint2* scratch,
    uint32_t* histFlat,
    uint32_t* blockSums,
    uint32_t paddedCount,
    uint32_t maxKeyExclusive,
    cudaStream_t stream
) {
    using namespace RadixSort;
    
    const uint32_t gridSize = (paddedCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    const uint32_t histLength = gridSize * BUCKETS;
    const uint32_t histBlocks = (histLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Determine number of passes (2 for ≤16-bit keys, else 4)
    int passes = (maxKeyExclusive <= (1u << 16)) ? 2 : 4;
    
    uint2* input = data;
    uint2* output = scratch;
    
    for (int pass = 0; pass < passes; ++pass) {
        uint32_t shift = pass * 8;
        
        // 1) Histogram
        histogramKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(
            input, histFlat, paddedCount, shift
        );
        
        // 2) Scan histogram blocks
        scanHistogramBlocksKernel<<<histBlocks, BLOCK_SIZE, 0, stream>>>(
            histFlat, blockSums, paddedCount
        );
        
        // 3) Scan block sums (single block)
        scanBlockSumsKernel<<<1, BLOCK_SIZE, 0, stream>>>(
            blockSums, paddedCount
        );
        
        // 4) Apply offsets
        applyOffsetsKernel<<<histBlocks, BLOCK_SIZE, 0, stream>>>(
            histFlat, blockSums, paddedCount
        );
        
        // 5) Scatter
        scatterKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(
            input, output, histFlat, paddedCount, shift
        );
        
        // Swap buffers
        uint2* tmp = input;
        input = output;
        output = tmp;
    }
    
    // After even number of passes, result is back in original buffer
    // If odd number of passes, we'd need to copy back (but we only use 2 or 4)
}

} // extern "C"

