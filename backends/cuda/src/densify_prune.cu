#include "sad_common.cuh"
#include <cuda_runtime.h>

// GPU gradient-weighted initialization
__global__ void initGradientWeightedKernel(
    Site* __restrict__ sites,
    uint32_t numSites,
    uint32_t* __restrict__ seedCounter,
    const float3* __restrict__ target,
    const float* __restrict__ mask,
    float gradThreshold,
    uint32_t maxAttempts,
    float init_log_tau,
    float init_radius,
    int width,
    int height
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numSites) return;

    float2 acceptedPos = make_float2(-1.0f, -1.0f);
    float3 acceptedColor = make_float3(0.5f, 0.5f, 0.5f);
    float acceptedGradient = 0.0f;

    // Try quasi-random positions using R2 sequence
    for (uint32_t attempt = 0; attempt < maxAttempts; attempt++) {
        uint32_t seed = atomicAdd(seedCounter, 1);

        // R2 low-discrepancy sequence
        const float g = 1.32471795724474602596f;
        const float a1 = 1.0f / g;
        const float a2 = 1.0f / (g * g);

        float2 uv = make_float2(
            fmodf(0.5f + float(seed) * a1, 1.0f),
            fmodf(0.5f + float(seed) * a2, 1.0f)
        );

        float2 pos = make_float2(
            uv.x * float(width - 1),
            uv.y * float(height - 1)
        );

        int ix = int(pos.x);
        int iy = int(pos.y);
        float mask_val = mask[iy * width + ix];
        if (mask_val <= 0.0f) {
            continue;
        }

        // Compute gradient at this position (Sobel)
        auto sample = [&](int dx, int dy) -> float {
            int sx = clampi(ix + dx, 0, width - 1);
            int sy = clampi(iy + dy, 0, height - 1);
            float3 rgb = target[sy * width + sx];
            return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
        };

        float TL = sample(-1, -1), T = sample(0, -1), TR = sample(1, -1);
        float L  = sample(-1,  0),                    R  = sample(1,  0);
        float BL = sample(-1,  1), B = sample(0,  1), BR = sample(1,  1);

        float Gx = (-TL + TR - 2.0f*L + 2.0f*R - BL + BR) / 4.0f;
        float Gy = (-TL - 2.0f*T - TR + BL + 2.0f*B + BR) / 4.0f;
        float grad = sqrtf(Gx*Gx + Gy*Gy) * mask_val;

        // Accept if gradient is high enough
        if (grad > gradThreshold) {
            acceptedPos = pos;
            acceptedColor = target[iy * width + ix];
            acceptedGradient = grad;
            break;
        }
    }

    // Initialize site with gradient-adjusted tau (matching Metal)
    sites[gid].position = acceptedPos;
    sites[gid].log_tau = init_log_tau + powf(acceptedGradient, 0.25f) * 2.0f;
    sites[gid].radius = init_radius;
    site_set_color(sites[gid], acceptedColor);
    site_set_aniso_dir(sites[gid], make_float2(1.0f, 0.0f));
    sites[gid].log_aniso = 0.0f;
}

// Split sites (densification)
__global__ void splitSitesKernel(
    Site* __restrict__ sites,
    AdamState* __restrict__ adam,
    const uint32_t* __restrict__ splitIndices,
    uint32_t numToSplit,
    const float* __restrict__ mass,
    const float* __restrict__ err_w,
    const float* __restrict__ err_wx,
    const float* __restrict__ err_wy,
    const float* __restrict__ err_wxx,
    const float* __restrict__ err_wxy,
    const float* __restrict__ err_wyy,
    uint32_t currentSiteCount,
    const float3* __restrict__ target,
    int width,
    int height
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numToSplit) return;
    
    uint32_t parentIdx = splitIndices[gid];
    Site parent = sites[parentIdx];
    
    if (parent.position.x < 0.0f) return;  // Skip inactive
    
    // Compute split direction from error-weighted covariance
    float ew = err_w[parentIdx];
    float2 axis = site_aniso_dir(parent);
    float2 center = parent.position;
    float logAniso = parent.log_aniso * 0.8f;
    
    if (ew > 1e-3f) {
        float mx = err_wx[parentIdx] / ew;
        float my = err_wy[parentIdx] / ew;
        
        // Mix with parent position
        center.x = 0.4f * parent.position.x + 0.6f * mx;
        center.y = 0.4f * parent.position.y + 0.6f * my;
        
        float exx = err_wxx[parentIdx] / ew - mx * mx;
        float exy = err_wxy[parentIdx] / ew - mx * my;
        float eyy = err_wyy[parentIdx] / ew - my * my;
        
        // Add regularization
        exx = fmaxf(exx, 1e-4f);
        eyy = fmaxf(eyy, 1e-4f);
        
        // Principal axis of covariance
        float theta = 0.5f * atan2f(2.0f * exy, exx - eyy);
        axis = make_float2(cosf(theta), sinf(theta));
        
        float trace = exx + eyy;
        float disc = sqrtf(fmaxf(0.0f, 0.25f * (exx - eyy) * (exx - eyy) + exy * exy));
        float lambda1 = fmaxf(1e-4f, 0.5f * trace + disc);
        float lambda2 = fmaxf(1e-4f, 0.5f * trace - disc);
        
        logAniso = clampf(-0.5f * logf(lambda1 / lambda2), -2.0f, 2.0f);
    } else {
        // Fallback: use image gradient
        int ix = clampi(int(parent.position.x), 0, width - 1);
        int iy = clampi(int(parent.position.y), 0, height - 1);
        
        auto sampleLum = [&](int dx, int dy) -> float {
            int sx = clampi(ix + dx, 0, width - 1);
            int sy = clampi(iy + dy, 0, height - 1);
            float3 rgb = target[sy * width + sx];
            return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
        };
        
        float TL = sampleLum(-1, -1), T = sampleLum(0, -1), TR = sampleLum(1, -1);
        float L  = sampleLum(-1,  0),                       R  = sampleLum(1,  0);
        float BL = sampleLum(-1,  1), B = sampleLum(0,  1), BR = sampleLum(1,  1);
        
        float Gx = (-TL + TR - 2.0f*L + 2.0f*R - BL + BR) / 4.0f;
        float Gy = (-TL - 2.0f*T - TR + BL + 2.0f*B + BR) / 4.0f;
        
        float2 grad = make_float2(Gx, Gy);
        float g2 = dot(grad, grad);
        if (g2 > 1e-8f) {
            axis = normalize(grad);
        }
    }
    
    // Split offset along axis
    float m = fmaxf(mass[parentIdx], 1.0f);
    float offsetDist = clampf(sqrtf(m) * 0.5f, 1.5f, 48.0f);
    float2 offset = make_float2(axis.x * offsetDist, axis.y * offsetDist);
    
    float2 posA = clamp2(
        make_float2(center.x + offset.x, center.y + offset.y),
        make_float2(0.0f, 0.0f),
        make_float2(float(width - 1), float(height - 1))
    );
    float2 posB = clamp2(
        make_float2(center.x - offset.x, center.y - offset.y),
        make_float2(0.0f, 0.0f),
        make_float2(float(width - 1), float(height - 1))
    );
    
    int ixA = int(posA.x), iyA = int(posA.y);
    int ixB = int(posB.x), iyB = int(posB.y);
    
    float3 colA = target[iyA * width + ixA];
    float3 colB = target[iyB * width + ixB];
    
    // Create child sites
    uint32_t childIdx = currentSiteCount + gid;
    if (childIdx == parentIdx) return;  // Safety check
    
    Site child0 = parent;
    Site child1 = parent;
    
    child0.position = posA;
    child1.position = posB;
    site_set_color(child0, colA);
    site_set_color(child1, colB);
    
    child0.log_tau = parent.log_tau - 0.25f;
    child1.log_tau = parent.log_tau - 0.25f;
    child0.radius = parent.radius * 0.85f;
    child1.radius = parent.radius * 0.85f;
    
    site_set_aniso_dir(child0, axis);
    site_set_aniso_dir(child1, axis);
    child0.log_aniso = logAniso;
    child1.log_aniso = logAniso;
    
    sites[parentIdx] = child0;
    sites[childIdx] = child1;
    
    // Reset optimizer state
    AdamState zero = {};
    adam[parentIdx] = zero;
    adam[childIdx] = zero;
}

// Compute densify score pairs for sorting
__global__ void computeDensifyScorePairsKernel(
    const Site* __restrict__ sites,
    const float* __restrict__ mass,
    const float* __restrict__ energy,
    uint2* __restrict__ pairs,
    uint32_t siteCount,
    float minMass,
    float scoreAlpha,
    uint32_t pairCount
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pairCount) return;
    
    uint2 outPair = make_uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
    
    if (gid < siteCount) {
        float m = mass[gid];
        float e = energy[gid];
        bool active = (sites[gid].position.x >= 0.0f);
        
        if (active && m > minMass && isfinite(e)) {
            float denom = powf(fmaxf(m, 1e-8f), scoreAlpha);
            float score = fmaxf(e, 0.0f) / denom;
            
            // Invert key for descending sort
            uint32_t key = 0xFFFFFFFFu - __float_as_uint(score);
            outPair = make_uint2(key, gid);
        }
    }
    
    pairs[gid] = outPair;
}

// Compute prune score pairs for sorting
__global__ void computePruneScorePairsKernel(
    const Site* __restrict__ sites,
    const float* __restrict__ removal_delta,
    uint2* __restrict__ pairs,
    uint32_t siteCount,
    float deltaNorm,
    uint32_t pairCount
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pairCount) return;
    
    uint2 outPair = make_uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
    
    if (gid < siteCount) {
        bool active = (sites[gid].position.x >= 0.0f);
        float delta = removal_delta[gid] * deltaNorm;
        
        if (active && isfinite(delta)) {
            uint32_t key = __float_as_uint(fmaxf(delta, 0.0f));
            outPair = make_uint2(key, gid);
        }
    }
    
    pairs[gid] = outPair;
}

// Write split/prune indices from sorted pairs
__global__ void writeSplitIndicesFromSortedKernel(
    const uint2* __restrict__ sortedPairs,
    uint32_t* __restrict__ splitIndices,
    uint32_t numToWrite
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numToWrite) return;
    splitIndices[gid] = sortedPairs[gid].y;
}

// Prune sites by marking them inactive
__global__ void pruneSitesByIndexKernel(
    Site* __restrict__ sites,
    const uint32_t* __restrict__ indices,
    uint32_t count
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    
    uint32_t idx = indices[gid];
    if (idx == 0xFFFFFFFFu) return;
    
    sites[idx].position = make_float2(-1.0f, -1.0f);
}

// C++ wrappers
extern "C" {

void launchInitGradientWeighted(
    Site* sites,
    uint32_t numSites,
    uint32_t* seedCounter,
    const float3* target,
    const float* mask,
    float gradThreshold,
    uint32_t maxAttempts,
    float init_log_tau,
    float init_radius,
    int width,
    int height,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (numSites + threads - 1) / threads;

    initGradientWeightedKernel<<<blocks, threads, 0, stream>>>(
        sites, numSites, seedCounter, target, mask,
        gradThreshold, maxAttempts, init_log_tau, init_radius, width, height
    );
}

void launchSplitSites(
    Site* sites,
    AdamState* adam,
    const uint32_t* splitIndices,
    uint32_t numToSplit,
    const float* mass,
    const float* err_w, const float* err_wx, const float* err_wy,
    const float* err_wxx, const float* err_wxy, const float* err_wyy,
    uint32_t currentSiteCount,
    const float3* target,
    int width,
    int height,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (numToSplit + threads - 1) / threads;
    
    splitSitesKernel<<<blocks, threads, 0, stream>>>(
        sites, adam, splitIndices, numToSplit,
        mass, err_w, err_wx, err_wy, err_wxx, err_wxy, err_wyy,
        currentSiteCount, target, width, height
    );
}

void launchComputeDensifyScorePairs(
    const Site* sites,
    const float* mass,
    const float* energy,
    uint2* pairs,
    uint32_t siteCount,
    float minMass,
    float scoreAlpha,
    uint32_t pairCount,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (pairCount + threads - 1) / threads;
    
    computeDensifyScorePairsKernel<<<blocks, threads, 0, stream>>>(
        sites, mass, energy, pairs, siteCount, minMass, scoreAlpha, pairCount
    );
}

void launchComputePruneScorePairs(
    const Site* sites,
    const float* removal_delta,
    uint2* pairs,
    uint32_t siteCount,
    float deltaNorm,
    uint32_t pairCount,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (pairCount + threads - 1) / threads;
    
    computePruneScorePairsKernel<<<blocks, threads, 0, stream>>>(
        sites, removal_delta, pairs, siteCount, deltaNorm, pairCount
    );
}

void launchWriteSplitIndicesFromSorted(
    const uint2* sortedPairs,
    uint32_t* splitIndices,
    uint32_t numToWrite,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (numToWrite + threads - 1) / threads;
    
    writeSplitIndicesFromSortedKernel<<<blocks, threads, 0, stream>>>(
        sortedPairs, splitIndices, numToWrite
    );
}

void launchPruneSitesByIndex(
    Site* sites,
    const uint32_t* indices,
    uint32_t count,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    
    pruneSitesByIndexKernel<<<blocks, threads, 0, stream>>>(
        sites, indices, count
    );
}

} // extern "C"
