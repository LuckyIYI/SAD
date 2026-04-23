#include "sad_common.cuh"
#include <cuda_runtime.h>

// Compute PSNR accumulator
__global__ void computePSNRKernel(
    const float3* __restrict__ rendered,
    const float3* __restrict__ target,
    const float* __restrict__ mask,
    float* __restrict__ mse_accum,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    if (mask[pixelIdx] <= 0.0f) return;
    float3 r = rendered[pixelIdx];
    float3 t = target[pixelIdx];
    float3 diff = make_float3(r.x - t.x, r.y - t.y, r.z - t.z);
    float pixel_mse = dot(diff, diff);
    
    atomicAdd(mse_accum, pixel_mse);
}

// Compute SSIM accumulator
__global__ void computeSSIMKernel(
    const float3* __restrict__ rendered,
    const float3* __restrict__ target,
    const float* __restrict__ mask,
    float* __restrict__ ssim_accum,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    if (mask[y * width + x] <= 0.0f) return;
    
    const float3 kLumaWeights = make_float3(0.299f, 0.587f, 0.114f);
    
    float mu_x = 0.0f, mu_y = 0.0f;
    float ex2 = 0.0f, ey2 = 0.0f, exy = 0.0f;
    
    int max_x = width - 1;
    int max_y = height - 1;
    
    for (int dy = -1; dy <= 1; dy++) {
        int yy = clampi(y + dy, 0, max_y);
        for (int dx = -1; dx <= 1; dx++) {
            int xx = clampi(x + dx, 0, max_x);
            int idx = yy * width + xx;
            
            float xval = dot(rendered[idx], kLumaWeights);
            float yval = dot(target[idx], kLumaWeights);
            
            mu_x += xval;
            mu_y += yval;
            ex2 += xval * xval;
            ey2 += yval * yval;
            exy += xval * yval;
        }
    }
    
    mu_x *= SSIM_INV_N;
    mu_y *= SSIM_INV_N;
    ex2 *= SSIM_INV_N;
    ey2 *= SSIM_INV_N;
    exy *= SSIM_INV_N;
    
    float sigma_x2 = fmaxf(ex2 - mu_x * mu_x, 0.0f);
    float sigma_y2 = fmaxf(ey2 - mu_y * mu_y, 0.0f);
    float sigma_xy = exy - mu_x * mu_y;
    
    float A = 2.0f * mu_x * mu_y + SSIM_C1;
    float B = 2.0f * sigma_xy + SSIM_C2;
    float C = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
    float D = sigma_x2 + sigma_y2 + SSIM_C2;
    
    float denom = fmaxf(C * D, 1e-8f);
    float ssim = (A * B) / denom;
    
    atomicAdd(ssim_accum, ssim);
}

// Count active sites
__global__ void countActiveSitesKernel(
    const Site* __restrict__ sites,
    uint32_t* __restrict__ count,
    uint32_t siteCount
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= siteCount) return;
    
    if (sites[gid].position.x >= 0.0f) {
        atomicAdd(count, 1);
    }
}

// Compute statistics for each site (mass and energy)
// This is a simplified version - for full efficiency, should use the
// efficient reduced path similar to gradients
__global__ void computeSiteStatsSimpleKernel(
    const uint32_t* __restrict__ cand0,
    const uint32_t* __restrict__ cand1,
    const float3* __restrict__ target,
    const float* __restrict__ mask,
    const Site* __restrict__ sites,
    float inv_scale_sq,
    uint32_t siteCount,
    float* __restrict__ mass,
    float* __restrict__ energy,
    float* __restrict__ err_w,
    float* __restrict__ err_wx,
    float* __restrict__ err_wy,
    float* __restrict__ err_wxx,
    float* __restrict__ err_wxy,
    float* __restrict__ err_wyy,
    int width,
    int height,
    int candWidth,
    int candHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    if (mask[pixelIdx] <= 0.0f) return;
    float2 uv = make_float2(float(x), float(y));
    
    // Load candidates
    int candIdx = candidate_index_for_pixel(x, y, width, height, candWidth, candHeight);
    uint4 c0 = reinterpret_cast<const uint4*>(cand0)[candIdx];
    uint4 c1 = reinterpret_cast<const uint4*>(cand1)[candIdx];
    uint32_t candIds[NUM_CANDIDATES] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    
    // Forward pass
    float logits[NUM_CANDIDATES];
    float max_logit = -INFINITY;
    
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        uint32_t idx = candIds[i];
        if (idx >= siteCount || sites[idx].position.x < 0.0f) {
            logits[i] = -INFINITY;
            continue;
        }

        float tau = expf(sites[idx].log_tau);
        float dmix2 = voronoi_dmix2(sites[idx], uv, inv_scale_sq);
        logits[i] = -tau * dmix2;
        max_logit = fmaxf(max_logit, logits[i]);
    }
    
    if (isinf(max_logit) && max_logit < 0.0f) return;
    
    float weights[NUM_CANDIDATES];
    float sum_w = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        weights[i] = expf(logits[i] - max_logit);
        sum_w += weights[i];
    }
    
    float inv_sum = 1.0f / fmaxf(sum_w, 1e-8f);
    
    float3 pred = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        weights[i] *= inv_sum;
        uint32_t idx = candIds[i];
        if (idx < siteCount && sites[idx].position.x >= 0.0f) {
            float3 site_col = site_color(sites[idx]);
            pred.x += weights[i] * site_col.x;
            pred.y += weights[i] * site_col.y;
            pred.z += weights[i] * site_col.z;
        }
    }
    
    float3 tgt = target[pixelIdx];
    float3 diff = make_float3(pred.x - tgt.x, pred.y - tgt.y, pred.z - tgt.z);
    float err = dot(diff, diff);
    
    // Accumulate per unique site
    bool seen[NUM_CANDIDATES] = {false};
    
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        uint32_t idx = candIds[i];
        if (idx >= siteCount || sites[idx].position.x < 0.0f || seen[i]) continue;
        
        // Check if we've seen this site before
        bool duplicate = false;
        for (int j = 0; j < i; j++) {
            if (candIds[j] == idx) {
                duplicate = true;
                seen[j] = true;
                break;
            }
        }
        
        if (!duplicate) {
            float w_total = 0.0f;
            for (int j = i; j < NUM_CANDIDATES; j++) {
                if (candIds[j] == idx) {
                    w_total += weights[j];
                    seen[j] = true;
                }
            }
            
            if (w_total > 0.0f) {
                atomicAdd(&mass[idx], w_total);
                atomicAdd(&energy[idx], w_total * err);
                
                float werr = w_total * fminf(err, 1.0f);
                atomicAdd(&err_w[idx], werr);
                atomicAdd(&err_wx[idx], werr * uv.x);
                atomicAdd(&err_wy[idx], werr * uv.y);
                atomicAdd(&err_wxx[idx], werr * uv.x * uv.x);
                atomicAdd(&err_wxy[idx], werr * uv.x * uv.y);
                atomicAdd(&err_wyy[idx], werr * uv.y * uv.y);
            }
        }
    }
}

// Precompute site data (tau, aniso_scale)
__global__ void precomputeSiteDataKernel(
    Site* __restrict__ sites,
    uint32_t siteCount
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= siteCount) return;
    
    // No-op: we compute tau and aniso_scale inline with expf(log_tau) and expf(log_aniso)
    // This function kept for API compatibility but does nothing
}

// C++ wrappers
extern "C" {

void launchPrecomputeSiteData(
    Site* sites,
    uint32_t siteCount,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (siteCount + threads - 1) / threads;
    precomputeSiteDataKernel<<<blocks, threads, 0, stream>>>(sites, siteCount);
}

void launchComputePSNR(
    const float3* rendered,
    const float3* target,
    const float* mask,
    float* mse_accum,
    int width,
    int height,
    cudaStream_t stream
) {
    // Clear accumulator
    cudaMemsetAsync(mse_accum, 0, sizeof(float), stream);
    
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    computePSNRKernel<<<grid, block, 0, stream>>>(
        rendered, target, mask, mse_accum, width, height
    );
}

void launchComputeSSIM(
    const float3* rendered,
    const float3* target,
    const float* mask,
    float* ssim_accum,
    int width,
    int height,
    cudaStream_t stream
) {
    // Clear accumulator
    cudaMemsetAsync(ssim_accum, 0, sizeof(float), stream);
    
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    computeSSIMKernel<<<grid, block, 0, stream>>>(
        rendered, target, mask, ssim_accum, width, height
    );
}

void launchCountActiveSites(
    const Site* sites,
    uint32_t* count,
    uint32_t siteCount,
    cudaStream_t stream
) {
    cudaMemsetAsync(count, 0, sizeof(uint32_t), stream);
    
    int threads = 256;
    int blocks = (siteCount + threads - 1) / threads;
    
    countActiveSitesKernel<<<blocks, threads, 0, stream>>>(
        sites, count, siteCount
    );
}

void launchComputeSiteStatsSimple(
    const uint32_t* cand0,
    const uint32_t* cand1,
    const float3* target,
    const float* mask,
    const Site* sites,
    float inv_scale_sq,
    uint32_t siteCount,
    float* mass,
    float* energy,
    float* err_w, float* err_wx, float* err_wy,
    float* err_wxx, float* err_wxy, float* err_wyy,
    int width,
    int height,
    int candWidth,
    int candHeight,
    cudaStream_t stream
) {
    // Clear stat buffers
    cudaMemsetAsync(mass, 0, siteCount * sizeof(float), stream);
    cudaMemsetAsync(energy, 0, siteCount * sizeof(float), stream);
    cudaMemsetAsync(err_w, 0, siteCount * sizeof(float), stream);
    cudaMemsetAsync(err_wx, 0, siteCount * sizeof(float), stream);
    cudaMemsetAsync(err_wy, 0, siteCount * sizeof(float), stream);
    cudaMemsetAsync(err_wxx, 0, siteCount * sizeof(float), stream);
    cudaMemsetAsync(err_wxy, 0, siteCount * sizeof(float), stream);
    cudaMemsetAsync(err_wyy, 0, siteCount * sizeof(float), stream);
    
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    computeSiteStatsSimpleKernel<<<grid, block, 0, stream>>>(
        cand0, cand1, target, mask, sites, inv_scale_sq, siteCount,
        mass, energy, err_w, err_wx, err_wy, err_wxx, err_wxy, err_wyy,
        width, height, candWidth, candHeight
    );
}

} // extern "C"
