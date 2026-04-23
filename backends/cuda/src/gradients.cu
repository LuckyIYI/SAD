#include "sad_common.cuh"
#include <cuda_runtime.h>

// Compute SSIM window stats (3x3 window)
__device__ __forceinline__ void ssim_window_stats(
    const float3* rendered,
    const float3* target,
    int x, int y, int width, int height,
    float& mu_x, float& mu_y,
    float& ex2, float& ey2, float& exy
) {
    const float3 kLumaWeights = make_float3(0.299f, 0.587f, 0.114f);
    
    mu_x = 0.0f; mu_y = 0.0f;
    ex2 = 0.0f; ey2 = 0.0f; exy = 0.0f;
    
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
}

// Compute SSIM window stats using shared memory (3x3 window in 18x18 block)
__device__ __forceinline__ void ssim_window_stats_shared(
    const float3* smem_rendered,
    const float3* smem_target,
    int lx, int ly,
    float& mu_x, float& mu_y,
    float& ex2, float& ey2, float& exy
) {
    const float3 kLumaWeights = make_float3(0.299f, 0.587f, 0.114f);
    
    mu_x = 0.0f; mu_y = 0.0f;
    ex2 = 0.0f; ey2 = 0.0f; exy = 0.0f;
    
    #pragma unroll
    for (int dy = 0; dy <= 2; dy++) {
        #pragma unroll
        for (int dx = 0; dx <= 2; dx++) {
            int idx = (ly + dy) * 18 + (lx + dx);
            
            float xval = dot(smem_rendered[idx], kLumaWeights);
            float yval = dot(smem_target[idx], kLumaWeights);
            
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
}

// Compute removal delta for a site
__device__ __forceinline__ float removalDeltaForSite(
    float3 pred, float3 tgt, float3 siteColor, float w_total
) {
    float keep = 1.0f - w_total;
    if (keep < 1e-4f) {
        return 1.0e6f;  // Very costly to remove
    }
    
    float3 pred_keep = make_float3(
        (pred.x - w_total * siteColor.x) / keep,
        (pred.y - w_total * siteColor.y) / keep,
        (pred.z - w_total * siteColor.z) / keep
    );
    
    float3 diff = make_float3(pred.x - tgt.x, pred.y - tgt.y, pred.z - tgt.z);
    float3 diff_keep = make_float3(pred_keep.x - tgt.x, pred_keep.y - tgt.y, pred_keep.z - tgt.z);
    
    return dot(diff_keep, diff_keep) - dot(diff, diff);
}

// Tiled gradient computation using threadgroup hash reduction
__global__ void computeGradientsTiledKernel(
    const uint32_t* __restrict__ cand0,
    const uint32_t* __restrict__ cand1,
    const float3* __restrict__ target,
    const float3* __restrict__ rendered,
    const float* __restrict__ mask,
    float* __restrict__ grad_pos_x,
    float* __restrict__ grad_pos_y,
    float* __restrict__ grad_log_tau,
    float* __restrict__ grad_radius,
    float* __restrict__ grad_color_r,
    float* __restrict__ grad_color_g,
    float* __restrict__ grad_color_b,
    float* __restrict__ grad_dir_x,
    float* __restrict__ grad_dir_y,
    float* __restrict__ grad_log_aniso,
    const Site* __restrict__ sites,
    float inv_scale_sq,
    uint32_t siteCount,
    float* __restrict__ removal_delta,
    uint32_t computeRemoval,
    float ssim_weight,
    int width,
    int height,
    int candWidth,
    int candHeight
) {
    extern __shared__ uint32_t shared_mem[];
    
    uint32_t* tg_keys = shared_mem;
    float* tg_grad_pos_x = (float*)(tg_keys + TILE_HASH_SIZE);
    float* tg_grad_pos_y = tg_grad_pos_x + TILE_HASH_SIZE;
    float* tg_grad_log_tau = tg_grad_pos_y + TILE_HASH_SIZE;
    float* tg_grad_radius = tg_grad_log_tau + TILE_HASH_SIZE;
    float* tg_grad_color_r = tg_grad_radius + TILE_HASH_SIZE;
    float* tg_grad_color_g = tg_grad_color_r + TILE_HASH_SIZE;
    float* tg_grad_color_b = tg_grad_color_g + TILE_HASH_SIZE;
    float* tg_grad_dir_x = tg_grad_color_b + TILE_HASH_SIZE;
    float* tg_grad_dir_y = tg_grad_dir_x + TILE_HASH_SIZE;
    float* tg_grad_log_aniso = tg_grad_dir_y + TILE_HASH_SIZE;
    float* tg_delta = tg_grad_log_aniso + TILE_HASH_SIZE;
    
    // SSIM shared memory
    float3* smem_rendered = (float3*)(tg_delta + TILE_HASH_SIZE);
    float3* smem_target = smem_rendered + 324; // (16+2)*(16+2) = 324
    
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int local_threads = blockDim.x * blockDim.y;
    
    // Initialize shared memory
    for (int i = local_idx; i < TILE_HASH_SIZE; i += local_threads) {
        tg_keys[i] = TILE_EMPTY_KEY;
        tg_grad_pos_x[i] = 0.0f;
        tg_grad_pos_y[i] = 0.0f;
        tg_grad_log_tau[i] = 0.0f;
        tg_grad_radius[i] = 0.0f;
        tg_grad_color_r[i] = 0.0f;
        tg_grad_color_g[i] = 0.0f;
        tg_grad_color_b[i] = 0.0f;
        tg_grad_dir_x[i] = 0.0f;
        tg_grad_dir_y[i] = 0.0f;
        tg_grad_log_aniso[i] = 0.0f;
        tg_delta[i] = 0.0f;
    }
    
    // Load rendered and target into shared memory for SSIM
    if (ssim_weight > 0.0f) {
        for (int i = local_idx; i < 324; i += local_threads) {
            int l_x = i % 18;
            int l_y = i / 18;
            int g_x = clampi(blockIdx.x * 16 + l_x - 1, 0, width - 1);
            int g_y = clampi(blockIdx.y * 16 + l_y - 1, 0, height - 1);
            int g_idx = g_y * width + g_x;
            smem_rendered[i] = rendered[g_idx];
            smem_target[i] = target[g_idx];
        }
    }
    __syncthreads();
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool in_bounds = (x < width && y < height);
    
    if (in_bounds) {
        int pixelIdx = y * width + x;
        float mask_val = mask[pixelIdx];
        if (mask_val <= 0.0f) {
            return;
        }
        float2 uv = make_float2(float(x), float(y));
        
        // Load candidates
        int candIdx = candidate_index_for_pixel(x, y, width, height, candWidth, candHeight);
        uint4 c0 = reinterpret_cast<const uint4*>(cand0)[candIdx];
        uint4 c1 = reinterpret_cast<const uint4*>(cand1)[candIdx];
        uint32_t candIds[NUM_CANDIDATES] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        
        // Forward pass: compute logits and weights
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
        
        if (!isinf(max_logit) || max_logit > 0.0f) {
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
            
            // Loss gradient
            float3 tgt = target[pixelIdx];
            float3 dL_dpred = make_float3(
                2.0f * (pred.x - tgt.x),
                2.0f * (pred.y - tgt.y),
                2.0f * (pred.z - tgt.z)
            );
            
            // SSIM component (if enabled)
            if (ssim_weight > 0.0f) {
                float mu_x, mu_y, ex2, ey2, exy;
                ssim_window_stats_shared(smem_rendered, smem_target, threadIdx.x, threadIdx.y, mu_x, mu_y, ex2, ey2, exy);
                
                float sigma_x2 = fmaxf(ex2 - mu_x * mu_x, 0.0f);
                float sigma_y2 = fmaxf(ey2 - mu_y * mu_y, 0.0f);
                float sigma_xy = exy - mu_x * mu_y;
                
                float A = 2.0f * mu_x * mu_y + SSIM_C1;
                float B = 2.0f * sigma_xy + SSIM_C2;
                float C = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
                float D = sigma_x2 + sigma_y2 + SSIM_C2;
                
                float denom = fmaxf(C * D, 1e-8f);
                float num = A * B;
                
                const float3 kLumaWeights = make_float3(0.299f, 0.587f, 0.114f);
                float xk = dot(rendered[pixelIdx], kLumaWeights);
                float yk = dot(target[pixelIdx], kLumaWeights);
                
                float dmu_x = SSIM_INV_N;
                float dsig_x2 = (2.0f * SSIM_INV_N) * (xk - mu_x);
                float dcov = SSIM_INV_N * (yk - mu_y);
                
                float dA = 2.0f * mu_y * dmu_x;
                float dB = 2.0f * dcov;
                float dC = 2.0f * mu_x * dmu_x;
                float dD = dsig_x2;
                
                float dSSIM = (dA * B + A * dB) / denom - num * (dC * D + C * dD) / (denom * denom);
                
                dL_dpred.x += (-ssim_weight * dSSIM) * kLumaWeights.x;
                dL_dpred.y += (-ssim_weight * dSSIM) * kLumaWeights.y;
                dL_dpred.z += (-ssim_weight * dSSIM) * kLumaWeights.z;
            }
            
            // Removal delta computation (if enabled)
            if (computeRemoval != 0) {
                uint32_t uniqueSites[NUM_CANDIDATES];
                float uniqueWeights[NUM_CANDIDATES];
                int numUnique = 0;
                
                #pragma unroll
                for (int i = 0; i < NUM_CANDIDATES; i++) {
                    uint32_t idx = candIds[i];
                    if (idx >= siteCount || sites[idx].position.x < 0.0f) continue;
                    
                    float w = weights[i];
                    bool found = false;
                    
                    for (int j = 0; j < numUnique; j++) {
                        if (uniqueSites[j] == idx) {
                            uniqueWeights[j] += w;
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found && numUnique < NUM_CANDIDATES) {
                        uniqueSites[numUnique] = idx;
                        uniqueWeights[numUnique] = w;
                        numUnique++;
                    }
                }
                
                // Hash insert removal deltas
                for (int i = 0; i < numUnique; i++) {
                    uint32_t idx = uniqueSites[i];
                    float w_total = uniqueWeights[i];
                    
                    if (w_total > 0.0f) {
                        float delta = removalDeltaForSite(pred, tgt, site_color(sites[idx]), w_total);
                        bool inserted = false;
                        uint32_t base = tile_hash(idx) & TILE_HASH_MASK;
                        
                        #pragma unroll
                        for (uint32_t probe = 0; probe < TILE_MAX_PROBES; probe++) {
                            uint32_t slot = (base + probe) & TILE_HASH_MASK;
                            uint32_t old = atomicCAS(&tg_keys[slot], TILE_EMPTY_KEY, idx);
                            
                            if (old == TILE_EMPTY_KEY || old == idx) {
                                atomicAdd(&tg_delta[slot], delta);
                                inserted = true;
                                break;
                            }
                        }
                        
                        if (!inserted) {
                            atomicAdd(&removal_delta[idx], delta);
                        }
                    }
                }
            }
            
            // Compute gradients for each candidate site
            #pragma unroll
            for (int i = 0; i < NUM_CANDIDATES; i++) {
                uint32_t idx = candIds[i];
                if (idx >= siteCount) continue;
                
                Site site = sites[idx];
                if (site.position.x < 0.0f) continue;
                
                float w = weights[i];
                
                // Color gradient
                float3 dL_dcolor = make_float3(w * dL_dpred.x, w * dL_dpred.y, w * dL_dpred.z);
                
                // Logit gradient
                float3 site_col = site_color(site);
                float dL_dlogit = dL_dpred.x * w * (site_col.x - pred.x) +
                                  dL_dpred.y * w * (site_col.y - pred.y) +
                                  dL_dpred.z * w * (site_col.z - pred.z);
                
                // Parameter gradients
                float tau = expf(site.log_tau);
                float2 diff = make_float2(uv.x - site.position.x, uv.y - site.position.y);
                
                float diff2 = dot(diff, diff);
                float2 dir = site_aniso_dir(site);
                float proj = dot(dir, diff);
                float proj2 = proj * proj;
                float perp2 = fmaxf(diff2 - proj2, 0.0f);
                
                float l1 = expf(site.log_aniso);
                float l2 = 1.0f / fmaxf(l1, 1e-8f);
                
                float d2_norm = (l1 * proj2 + l2 * perp2) * inv_scale_sq;
                float d2_safe = fmaxf(d2_norm, 1e-8f);
                float inv_sqrt_d2 = rsqrtf(d2_safe);
                float inv_scale = sqrtf(inv_scale_sq);
                float r_norm = site.radius * inv_scale;
                float dmix2 = (sqrtf(d2_safe) - r_norm);
                
                // Position gradient
                float2 g_diff = make_float2(
                    l2 * diff.x + (l1 - l2) * dir.x * proj,
                    l2 * diff.y + (l1 - l2) * dir.y * proj
                );
                float2 dL_dpos = make_float2(
                    dL_dlogit * (tau * inv_scale_sq * inv_sqrt_d2) * g_diff.x,
                    dL_dlogit * (tau * inv_scale_sq * inv_sqrt_d2) * g_diff.y
                );
                
                // log_tau gradient
                float dL_dlog_tau = dL_dlogit * (-dmix2) * tau;
                
                // radius gradient
                float dL_dradius = dL_dlogit * tau * inv_scale;
                
                // log_aniso gradient
                float d2_dlog_aniso = (l1 * proj2 - l2 * perp2) * inv_scale_sq;
                float dL_dlog_aniso = dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_dlog_aniso;
                
                // aniso_dir gradient
                float2 d2_ddir = make_float2(
                    (2.0f * (l1 - l2) * proj * inv_scale_sq) * diff.x,
                    (2.0f * (l1 - l2) * proj * inv_scale_sq) * diff.y
                );
                float2 dL_ddir = make_float2(
                    dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_ddir.x,
                    dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_ddir.y
                );
                
                // Hash insert into threadgroup memory
                bool inserted = false;
                uint32_t base = tile_hash(idx) & TILE_HASH_MASK;
                
                #pragma unroll
                for (uint32_t probe = 0; probe < TILE_MAX_PROBES; probe++) {
                    uint32_t slot = (base + probe) & TILE_HASH_MASK;
                    uint32_t old = atomicCAS(&tg_keys[slot], TILE_EMPTY_KEY, idx);
                    
                    if (old == TILE_EMPTY_KEY || old == idx) {
                        atomicAdd(&tg_grad_pos_x[slot], dL_dpos.x);
                        atomicAdd(&tg_grad_pos_y[slot], dL_dpos.y);
                        atomicAdd(&tg_grad_log_tau[slot], dL_dlog_tau);
                        atomicAdd(&tg_grad_radius[slot], dL_dradius);
                        atomicAdd(&tg_grad_color_r[slot], dL_dcolor.x);
                        atomicAdd(&tg_grad_color_g[slot], dL_dcolor.y);
                        atomicAdd(&tg_grad_color_b[slot], dL_dcolor.z);
                        atomicAdd(&tg_grad_dir_x[slot], dL_ddir.x);
                        atomicAdd(&tg_grad_dir_y[slot], dL_ddir.y);
                        atomicAdd(&tg_grad_log_aniso[slot], dL_dlog_aniso);
                        inserted = true;
                        break;
                    }
                }
                
                // Fallback to global memory if hash table is full
                if (!inserted) {
                    atomicAdd(&grad_pos_x[idx], dL_dpos.x);
                    atomicAdd(&grad_pos_y[idx], dL_dpos.y);
                    atomicAdd(&grad_log_tau[idx], dL_dlog_tau);
                    atomicAdd(&grad_radius[idx], dL_dradius);
                    atomicAdd(&grad_color_r[idx], dL_dcolor.x);
                    atomicAdd(&grad_color_g[idx], dL_dcolor.y);
                    atomicAdd(&grad_color_b[idx], dL_dcolor.z);
                    atomicAdd(&grad_dir_x[idx], dL_ddir.x);
                    atomicAdd(&grad_dir_y[idx], dL_ddir.y);
                    atomicAdd(&grad_log_aniso[idx], dL_dlog_aniso);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Flush threadgroup hash table to global memory
    for (int i = local_idx; i < TILE_HASH_SIZE; i += local_threads) {
        uint32_t siteID = tg_keys[i];
        if (siteID == TILE_EMPTY_KEY) continue;
        
        atomicAdd(&grad_pos_x[siteID], tg_grad_pos_x[i]);
        atomicAdd(&grad_pos_y[siteID], tg_grad_pos_y[i]);
        atomicAdd(&grad_log_tau[siteID], tg_grad_log_tau[i]);
        atomicAdd(&grad_radius[siteID], tg_grad_radius[i]);
        atomicAdd(&grad_color_r[siteID], tg_grad_color_r[i]);
        atomicAdd(&grad_color_g[siteID], tg_grad_color_g[i]);
        atomicAdd(&grad_color_b[siteID], tg_grad_color_b[i]);
        atomicAdd(&grad_dir_x[siteID], tg_grad_dir_x[i]);
        atomicAdd(&grad_dir_y[siteID], tg_grad_dir_y[i]);
        atomicAdd(&grad_log_aniso[siteID], tg_grad_log_aniso[i]);
        
        if (computeRemoval != 0 && tg_delta[i] != 0.0f) {
            atomicAdd(&removal_delta[siteID], tg_delta[i]);
        }
    }
}

// C++ wrapper
extern "C" {

void launchComputeGradientsTiled(
    const uint32_t* cand0,
    const uint32_t* cand1,
    const float3* target,
    const float3* rendered,
    const float* mask,
    float* grad_pos_x, float* grad_pos_y,
    float* grad_log_tau, float* grad_radius,
    float* grad_color_r, float* grad_color_g, float* grad_color_b,
    float* grad_dir_x, float* grad_dir_y, float* grad_log_aniso,
    const Site* sites,
    float inv_scale_sq,
    uint32_t siteCount,
    float* removal_delta,
    uint32_t computeRemoval,
    float ssim_weight,
    int width,
    int height,
    int candWidth,
    int candHeight,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    // Shared memory: 
    // - Hash table (keys + 10 gradient arrays + removal delta): 12 * TILE_HASH_SIZE * 4 bytes
    // - SSIM window cache: 2 * 324 * sizeof(float3) = 648 * 12 bytes = 7776 bytes
    size_t smem = 12 * TILE_HASH_SIZE * sizeof(float) + 7776;
    
    computeGradientsTiledKernel<<<grid, block, smem, stream>>>(
        cand0, cand1, target, rendered,
        mask,
        grad_pos_x, grad_pos_y, grad_log_tau, grad_radius,
        grad_color_r, grad_color_g, grad_color_b,
        grad_dir_x, grad_dir_y, grad_log_aniso,
        sites, inv_scale_sq, siteCount,
        removal_delta, computeRemoval, ssim_weight,
        width, height, candWidth, candHeight
    );
}

} // extern "C"
