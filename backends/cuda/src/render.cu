#include "sad_common.cuh"
#include <cuda_runtime.h>

// Render Voronoi diagram using candidate field
__global__ void renderVoronoiKernel(
    const uint32_t* __restrict__ cand0,
    const uint32_t* __restrict__ cand1,
    float3* __restrict__ output,
    const Site* __restrict__ sites,
    float inv_scale_sq,
    uint32_t siteCount,
    int width,
    int height,
    int candWidth,
    int candHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    float2 uv = make_float2(float(x), float(y));
    
    int candIdx = candidate_index_for_pixel(x, y, width, height, candWidth, candHeight);

    // Load 8 candidate site IDs
    uint32_t candIds[NUM_CANDIDATES];
    uint4 c0 = reinterpret_cast<const uint4*>(cand0)[candIdx];
    uint4 c1 = reinterpret_cast<const uint4*>(cand1)[candIdx];
    candIds[0] = c0.x; candIds[1] = c0.y; candIds[2] = c0.z; candIds[3] = c0.w;
    candIds[4] = c1.x; candIds[5] = c1.y; candIds[6] = c1.z; candIds[7] = c1.w;
    
    // Compute logits and max logit
    float logits[NUM_CANDIDATES];
    float max_logit = -INFINITY;
    
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        uint32_t idx = candIds[i];
        if (idx >= siteCount) {
            logits[i] = -INFINITY;
            continue;
        }
        
        Site site = sites[idx];
        if (site.position.x < 0.0f) {
            logits[i] = -INFINITY;
            continue;
        }
        
        float tau = expf(site.log_tau);
        float dmix2 = voronoi_dmix2(site, uv, inv_scale_sq);
        logits[i] = -tau * dmix2;
        max_logit = fmaxf(max_logit, logits[i]);
    }
    
    // Early exit if all candidates are invalid
    if (isinf(max_logit) && max_logit < 0.0f) {
        output[pixelIdx] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    
    // Softmax normalization
    float weights[NUM_CANDIDATES];
    float sum_w = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        weights[i] = expf(logits[i] - max_logit);
        sum_w += weights[i];
    }
    
    float inv_sum = 1.0f / fmaxf(sum_w, 1e-8f);
    
    // Blend colors
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        float w = weights[i] * inv_sum;
        uint32_t idx = candIds[i];
        if (!isnan(w) && !isinf(w) && idx < siteCount && sites[idx].position.x >= 0.0f) {
            float3 site_col = site_color(sites[idx]);
            color.x += w * site_col.x;
            color.y += w * site_col.y;
            color.z += w * site_col.z;
        }
    }
    
    output[pixelIdx] = color;
}

// Initialize candidate field
__global__ void initCandidatesKernel(
    uint32_t* __restrict__ cand0,
    uint32_t* __restrict__ cand1,
    uint32_t siteCount,
    uint32_t seed,
    bool perPixelMode,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    uint32_t idx[NUM_CANDIDATES];
    
    if (perPixelMode && siteCount == uint32_t(width * height)) {
        uint32_t selfIdx = pixelIdx;
        idx[0] = selfIdx;
        
        int neighbors[7][2] = {
            {-1, 0}, {1, 0}, {0, -1}, {0, 1},
            {-1, -1}, {1, -1}, {-1, 1}
        };
        
        for (int i = 0; i < 7; i++) {
            int nx = clampi(x + neighbors[i][0], 0, width - 1);
            int ny = clampi(y + neighbors[i][1], 0, height - 1);
            idx[i + 1] = ny * width + nx;
        }
    } else {
        uint32_t state = seed ^ (x * 1973u) ^ (y * 9277u);
        for (int i = 0; i < NUM_CANDIDATES; i++) {
            state = xorshift32(state + i);
            idx[i] = state % siteCount;
        }
    }
    
    reinterpret_cast<uint4*>(cand0)[pixelIdx] = make_uint4(idx[0], idx[1], idx[2], idx[3]);
    reinterpret_cast<uint4*>(cand1)[pixelIdx] = make_uint4(idx[4], idx[5], idx[6], idx[7]);
}

// Clear candidate field
__global__ void clearCandidatesKernel(
    uint32_t* __restrict__ cand0,
    uint32_t* __restrict__ cand1,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    uint4 empty = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    reinterpret_cast<uint4*>(cand0)[pixelIdx] = empty;
    reinterpret_cast<uint4*>(cand1)[pixelIdx] = empty;
}

// C++ wrappers
extern "C" {

void launchRenderVoronoi(
    const uint32_t* cand0,
    const uint32_t* cand1,
    float3* output,
    const Site* sites,
    float inv_scale_sq,
    uint32_t siteCount,
    int width,
    int height,
    int candWidth,
    int candHeight,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    renderVoronoiKernel<<<grid, block, 0, stream>>>(
        cand0, cand1, output, sites, inv_scale_sq, siteCount,
        width, height, candWidth, candHeight
    );
}

void launchInitCandidates(
    uint32_t* cand0,
    uint32_t* cand1,
    uint32_t siteCount,
    uint32_t seed,
    bool perPixelMode,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    initCandidatesKernel<<<grid, block, 0, stream>>>(
        cand0, cand1, siteCount, seed, perPixelMode, width, height
    );
}

void launchClearCandidates(
    uint32_t* cand0,
    uint32_t* cand1,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    clearCandidatesKernel<<<grid, block, 0, stream>>>(
        cand0, cand1, width, height
    );
}

} // extern "C"
