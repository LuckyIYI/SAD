#include "sad_common.cuh"
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Insert into sorted list of 4 closest sites (for JFA)
__device__ __forceinline__ void insertClosest4(
    uint32_t bestIdx[4],
    float bestD2[4],
    uint32_t candIdx,
    float2 uv,
    const Site* sites,
    uint32_t siteCount,
    float inv_scale_sq
) {
    if (candIdx >= siteCount) return;
    
    // Check if already in list
    for (int i = 0; i < 4; i++) {
        if (bestIdx[i] == candIdx) return;
    }
    
    // Skip inactive sites
    if (sites[candIdx].position.x < 0.0f) return;
    
    float dMix2 = voronoi_dmix2(sites[candIdx], uv, inv_scale_sq);
    float tau = fmaxf(expf(sites[candIdx].log_tau), 1e-4f);
    float d2 = tau * dMix2;
    
    // Insertion sort into 4-element list
    if (d2 < bestD2[3]) {
        if (d2 < bestD2[1]) {
            if (d2 < bestD2[0]) {
                // Insert at 0
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = bestD2[1]; bestIdx[2] = bestIdx[1];
                bestD2[1] = bestD2[0]; bestIdx[1] = bestIdx[0];
                bestD2[0] = d2; bestIdx[0] = candIdx;
            } else {
                // Insert at 1
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = bestD2[1]; bestIdx[2] = bestIdx[1];
                bestD2[1] = d2; bestIdx[1] = candIdx;
            }
        } else {
            if (d2 < bestD2[2]) {
                // Insert at 2
                bestD2[3] = bestD2[2]; bestIdx[3] = bestIdx[2];
                bestD2[2] = d2; bestIdx[2] = candIdx;
            } else {
                // Insert at 3
                bestD2[3] = d2; bestIdx[3] = candIdx;
            }
        }
    }
}

// JFA Seed: Each active site writes itself to its home pixel
__global__ void jfaSeedKernel(
    uint32_t* __restrict__ cand0,
    const Site* __restrict__ sites,
    uint32_t siteCount,
    int width,
    int height,
    int candDownscale
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= siteCount) return;
    
    Site site = sites[gid];
    
    // Skip inactive sites
    if (site.position.x < 0.0f) return;
    
    // Clamp position to image bounds
    float invDownscale = 1.0f / fmaxf(static_cast<float>(candDownscale), 1.0f);
    int x = clampi(int(site.position.x * invDownscale), 0, width - 1);
    int y = clampi(int(site.position.y * invDownscale), 0, height - 1);
    int pixelIdx = y * width + x;
    
    // Overwrite only the first slot to preserve any existing candidates.
    uint4 existing = reinterpret_cast<uint4*>(cand0)[pixelIdx];
    existing.x = gid;
    reinterpret_cast<uint4*>(cand0)[pixelIdx] = existing;
}

// JFA Flood: One pass with given step size
__global__ void jfaFloodKernel(
    const uint32_t* __restrict__ inCand0,
    uint32_t* __restrict__ outCand0,
    const Site* __restrict__ sites,
    uint32_t siteCount,
    uint32_t stepSize,
    float inv_scale_sq,
    int width,
    int height,
    int candDownscale,
    int targetWidth,
    int targetHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    float2 uv = make_float2(float(x) * float(candDownscale),
                            float(y) * float(candDownscale));
    float2 maxUv = make_float2(float(targetWidth - 1), float(targetHeight - 1));
    uv.x = fminf(uv.x, maxUv.x);
    uv.y = fminf(uv.y, maxUv.y);
    
    // Start with empty list
    uint32_t bestIdx[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    float bestD2[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
    
    // Sample 3x3 grid at step offset (9 samples including self)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sx = clampi(int(x) + dx * int(stepSize), 0, width - 1);
            int sy = clampi(int(y) + dy * int(stepSize), 0, height - 1);
            int sampleIdx = sy * width + sx;
            
            // Read candidates at this position
            uint4 cand = reinterpret_cast<const uint4*>(inCand0)[sampleIdx];
            
            // Try to insert each candidate
            insertClosest4(bestIdx, bestD2, cand.x, uv, sites, siteCount, inv_scale_sq);
            insertClosest4(bestIdx, bestD2, cand.y, uv, sites, siteCount, inv_scale_sq);
            insertClosest4(bestIdx, bestD2, cand.z, uv, sites, siteCount, inv_scale_sq);
            insertClosest4(bestIdx, bestD2, cand.w, uv, sites, siteCount, inv_scale_sq);
        }
    }
    
    // Write result
    reinterpret_cast<uint4*>(outCand0)[pixelIdx] = make_uint4(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]);
}

__device__ __forceinline__ float2 unpackHalf2(__half2 v) {
    return __half22float2(v);
}

__device__ __forceinline__ float candidatePackedDistance(
    const PackedCandidateSite& packed,
    float2 uv,
    float inv_scale_sq
) {
    float2 pos = unpackHalf2(packed.a);
    if (pos.x < 0.0f) {
        return INFINITY;
    }

    float2 tauRad = unpackHalf2(packed.b);
    float2 dir = unpackHalf2(packed.c);
    float2 anisoVal = unpackHalf2(packed.d);

    Site site = {};
    site.position = pos;
    site.log_tau = tauRad.x;
    site.radius = tauRad.y;
    site_set_aniso_dir(site, dir);
    site.log_aniso = anisoVal.x;

    float dMix2 = voronoi_dmix2(site, uv, inv_scale_sq);
    float tau = fmaxf(expf(site.log_tau), 1e-4f);
    return tau * dMix2;
}

__device__ __forceinline__ void insertClosest8Packed(
    uint32_t bestIdx[8],
    float bestD2[8],
    uint32_t candIdx,
    float2 uv,
    const PackedCandidateSite* sites,
    uint32_t siteCount,
    float inv_scale_sq
) {
    if (candIdx >= siteCount) return;

    for (int i = 0; i < 8; i++) {
        if (bestIdx[i] == candIdx) return;
    }

    float d2 = candidatePackedDistance(sites[candIdx], uv, inv_scale_sq);

    bool inFirstHalf = d2 < bestD2[3];
    if (inFirstHalf) {
        if (d2 < bestD2[1]) {
            if (d2 < bestD2[0]) {
                for (int i = 6; i >= 0; i--) {
                    bestD2[i+1] = bestD2[i];
                    bestIdx[i+1] = bestIdx[i];
                }
                bestD2[0] = d2; bestIdx[0] = candIdx;
            } else {
                for (int i = 6; i >= 1; i--) {
                    bestD2[i+1] = bestD2[i];
                    bestIdx[i+1] = bestIdx[i];
                }
                bestD2[1] = d2; bestIdx[1] = candIdx;
            }
        } else {
            if (d2 < bestD2[2]) {
                for (int i = 6; i >= 2; i--) {
                    bestD2[i+1] = bestD2[i];
                    bestIdx[i+1] = bestIdx[i];
                }
                bestD2[2] = d2; bestIdx[2] = candIdx;
            } else if (d2 < bestD2[3]) {
                for (int i = 6; i >= 3; i--) {
                    bestD2[i+1] = bestD2[i];
                    bestIdx[i+1] = bestIdx[i];
                }
                bestD2[3] = d2; bestIdx[3] = candIdx;
            }
        }
    } else {
        if (d2 < bestD2[5]) {
            if (d2 < bestD2[4]) {
                for (int i = 6; i >= 4; i--) {
                    bestD2[i+1] = bestD2[i];
                    bestIdx[i+1] = bestIdx[i];
                }
                bestD2[4] = d2; bestIdx[4] = candIdx;
            } else {
                for (int i = 6; i >= 5; i--) {
                    bestD2[i+1] = bestD2[i];
                    bestIdx[i+1] = bestIdx[i];
                }
                bestD2[5] = d2; bestIdx[5] = candIdx;
            }
        } else {
            if (d2 < bestD2[6]) {
                bestD2[7] = bestD2[6]; bestIdx[7] = bestIdx[6];
                bestD2[6] = d2; bestIdx[6] = candIdx;
            } else if (d2 < bestD2[7]) {
                bestD2[7] = d2; bestIdx[7] = candIdx;
            }
        }
    }
}

__device__ __forceinline__ void mergeCandidates8Packed(
    uint32_t bestIdx[8],
    float bestD2[8],
    uint4 c0,
    uint4 c1,
    float2 uv,
    const PackedCandidateSite* sites,
    uint32_t siteCount,
    float inv_scale_sq
) {
    insertClosest8Packed(bestIdx, bestD2, c0.x, uv, sites, siteCount, inv_scale_sq);
    insertClosest8Packed(bestIdx, bestD2, c0.y, uv, sites, siteCount, inv_scale_sq);
    insertClosest8Packed(bestIdx, bestD2, c0.z, uv, sites, siteCount, inv_scale_sq);
    insertClosest8Packed(bestIdx, bestD2, c0.w, uv, sites, siteCount, inv_scale_sq);
    insertClosest8Packed(bestIdx, bestD2, c1.x, uv, sites, siteCount, inv_scale_sq);
    insertClosest8Packed(bestIdx, bestD2, c1.y, uv, sites, siteCount, inv_scale_sq);
    insertClosest8Packed(bestIdx, bestD2, c1.z, uv, sites, siteCount, inv_scale_sq);
    insertClosest8Packed(bestIdx, bestD2, c1.w, uv, sites, siteCount, inv_scale_sq);
}

// VPT (Voronoi Particle Tracking) - Update candidates
__global__ void updateCandidatesKernel(
    const uint32_t* __restrict__ inCand0,
    const uint32_t* __restrict__ inCand1,
    uint32_t* __restrict__ outCand0,
    uint32_t* __restrict__ outCand1,
    const PackedCandidateSite* __restrict__ sites,
    uint32_t siteCount,
    uint32_t step,
    float inv_scale_sq,
    uint32_t stepHigh,
    float radiusScale,
    uint32_t radiusProbes,
    uint32_t injectCount,
    const uint32_t* __restrict__ hilbertOrder,
    const uint32_t* __restrict__ hilbertPos,
    uint32_t hilbertProbeCount,
    uint32_t hilbertWindow,
    int candDownscale,
    int targetWidth,
    int targetHeight,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    
    // Packed step: high 16 bits = jump step size, low 16 bits = step index
    uint32_t stepIndex = step & 0xFFFFu;
    uint32_t jumpStep = step >> 16;
    if (jumpStep == 0u) jumpStep = 1u;
    uint32_t fullStepIndex = (stepHigh << 16) | stepIndex;
    
    float2 uv = make_float2(float(x) * float(candDownscale),
                            float(y) * float(candDownscale));
    float2 maxUv = make_float2(float(targetWidth - 1), float(targetHeight - 1));
    uv.x = fminf(uv.x, maxUv.x);
    uv.y = fminf(uv.y, maxUv.y);
    
    uint4 self0 = reinterpret_cast<const uint4*>(inCand0)[pixelIdx];
    uint4 self1 = reinterpret_cast<const uint4*>(inCand1)[pixelIdx];
    
    uint32_t bestIdx[8];
    float bestD2[8];
    for (int i = 0; i < 8; i++) {
        bestIdx[i] = 0xFFFFFFFFu;
        bestD2[i] = INFINITY;
    }
    
    // Reuse current candidates
    mergeCandidates8Packed(bestIdx, bestD2, self0, self1, uv, sites, siteCount, inv_scale_sq);
    
    // Sample 4-connected neighbors
    int offsets[4][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };
    
    for (int i = 0; i < 4; i++) {
        int nx = clampi(x + offsets[i][0] * int(jumpStep), 0, width - 1);
        int ny = clampi(y + offsets[i][1] * int(jumpStep), 0, height - 1);
        int nIdx = ny * width + nx;
        
        uint4 neigh0 = reinterpret_cast<const uint4*>(inCand0)[nIdx];
        uint4 neigh1 = reinterpret_cast<const uint4*>(inCand1)[nIdx];
        mergeCandidates8Packed(bestIdx, bestD2, neigh0, neigh1, uv, sites, siteCount, inv_scale_sq);
    }
    
    // Random probes (radius scales with image width)
    float rad = radiusScale * (float(width) / 1024.0f);
    uint32_t state = (x * 73856093u) ^ (y * 19349663u) ^ ((stepIndex + jumpStep) * 83492791u);
    
    for (uint32_t r = 0; r < radiusProbes; r++) {
        float a = rand01(state) * 6.2831853f;
        int dx = int(cosf(a) * rad);
        int dy = int(sinf(a) * rad);
        int px = clampi(x + dx, 0, width - 1);
        int py = clampi(y + dy, 0, height - 1);
        
        if (px == x && py == y) continue;
        
        int pIdx = py * width + px;
        uint4 probe0 = reinterpret_cast<const uint4*>(inCand0)[pIdx];
        uint4 probe1 = reinterpret_cast<const uint4*>(inCand1)[pIdx];
        mergeCandidates8Packed(bestIdx, bestD2, probe0, probe1, uv, sites, siteCount, inv_scale_sq);
    }

    if (hilbertProbeCount > 0 && hilbertWindow > 0) {
        uint32_t bestCand = bestIdx[0];
        if (bestCand < siteCount) {
            uint32_t pos = hilbertPos[bestCand];
            uint32_t span = hilbertWindow * 2u + 1u;
            uint32_t hState = (x * 2654435761u) ^ (y * 1597334677u) ^ (fullStepIndex * 374761393u);
            for (uint32_t i = 0; i < hilbertProbeCount; ++i) {
                hState = xorshift32(hState + i);
                int offset = int(hState % span) - int(hilbertWindow);
                int idx = clampi(int(pos) + offset, 0, int(siteCount - 1));
                uint32_t cand = hilbertOrder[idx];
                insertClosest8Packed(bestIdx, bestD2, cand, uv, sites, siteCount, inv_scale_sq);
            }
        }
    }
    
    // Deterministic per-pixel injection for global coverage
    uint32_t injectState = (x * 1664525u) ^ (y * 1013904223u) ^ (fullStepIndex * 374761393u);
    for (uint32_t i = 0; i < injectCount; i++) {
        injectState = xorshift32(injectState + i);
        uint32_t cand = injectState % siteCount;
        insertClosest8Packed(bestIdx, bestD2, cand, uv, sites, siteCount, inv_scale_sq);
    }
    
    reinterpret_cast<uint4*>(outCand0)[pixelIdx] = make_uint4(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]);
    reinterpret_cast<uint4*>(outCand1)[pixelIdx] = make_uint4(bestIdx[4], bestIdx[5], bestIdx[6], bestIdx[7]);
}

// C++ wrappers
extern "C" {

void launchJFASeed(
    uint32_t* cand0,
    const Site* sites,
    uint32_t siteCount,
    int width,
    int height,
    int candDownscale,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (siteCount + threads - 1) / threads;
    
    jfaSeedKernel<<<blocks, threads, 0, stream>>>(
        cand0, sites, siteCount, width, height, candDownscale
    );
}

void launchJFAFlood(
    const uint32_t* inCand0,
    uint32_t* outCand0,
    const Site* sites,
    uint32_t siteCount,
    uint32_t stepSize,
    float inv_scale_sq,
    int width,
    int height,
    int candDownscale,
    int targetWidth,
    int targetHeight,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    jfaFloodKernel<<<grid, block, 0, stream>>>(
        inCand0, outCand0, sites, siteCount, stepSize, inv_scale_sq,
        width, height, candDownscale, targetWidth, targetHeight
    );
}

void launchUpdateCandidates(
    const uint32_t* inCand0,
    const uint32_t* inCand1,
    uint32_t* outCand0,
    uint32_t* outCand1,
    const PackedCandidateSite* sites,
    uint32_t siteCount,
    uint32_t step,
    float inv_scale_sq,
    uint32_t stepHigh,
    float radiusScale,
    uint32_t radiusProbes,
    uint32_t injectCount,
    const uint32_t* hilbertOrder,
    const uint32_t* hilbertPos,
    uint32_t hilbertProbeCount,
    uint32_t hilbertWindow,
    int candDownscale,
    int targetWidth,
    int targetHeight,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    updateCandidatesKernel<<<grid, block, 0, stream>>>(
        inCand0, inCand1, outCand0, outCand1, sites, siteCount, step,
        inv_scale_sq, stepHigh, radiusScale, radiusProbes, injectCount,
        hilbertOrder, hilbertPos, hilbertProbeCount, hilbertWindow,
        candDownscale, targetWidth, targetHeight, width, height
    );
}

} // extern "C"
