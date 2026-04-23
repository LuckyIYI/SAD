#include "sad_common.cuh"
#include <cuda_runtime.h>

struct TauDiffuseParams {
    uint32_t siteCount;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
    float lambda;
    float _pad3;
    float _pad4;
    float _pad5;
};

__global__ void tau_extract_kernel(const float *grad_log_tau, float *out, uint32_t siteCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(siteCount)) return;
    out[idx] = grad_log_tau[idx];
}

__global__ void tau_writeback_kernel(const float *in, float *grad_log_tau, uint32_t siteCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(siteCount)) return;
    grad_log_tau[idx] = in[idx];
}

__global__ void tau_diffuse_kernel(
    const uint32_t *cand0,
    const uint32_t *cand1,
    const Site *sites,
    const float *grad_raw,
    const float *grad_in,
    float *grad_out,
    uint32_t siteCount,
    float lambda,
    int width,
    int height,
    int candDownscale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(siteCount)) return;

    if (lambda <= 0.0f) {
        grad_out[idx] = grad_in[idx];
        return;
    }

    Site site = sites[idx];
    if (site.position.x < 0.0f) {
        grad_out[idx] = grad_in[idx];
        return;
    }

    float invDownscale = 1.0f / fmaxf(static_cast<float>(candDownscale), 1.0f);
    int px = static_cast<int>(site.position.x * invDownscale + 0.5f);
    int py = static_cast<int>(site.position.y * invDownscale + 0.5f);
    px = clampi(px, 0, width - 1);
    py = clampi(py, 0, height - 1);
    int p = py * width + px;

    uint4 c0 = reinterpret_cast<const uint4*>(cand0)[p];
    uint4 c1 = reinterpret_cast<const uint4*>(cand1)[p];
    uint32_t cand_ids[NUM_CANDIDATES] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};

    float sum = 0.0f;
    uint32_t count = 0;

    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        uint32_t n = cand_ids[i];
        if (n >= siteCount || n == static_cast<uint32_t>(idx)) continue;

        bool seen = false;
        for (int j = 0; j < i; ++j) {
            if (cand_ids[j] == n) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        if (sites[n].position.x < 0.0f) continue;

        sum += grad_in[n];
        count += 1;
    }

    float diag = 1.0f + lambda * static_cast<float>(count);
    grad_out[idx] = (grad_raw[idx] + lambda * sum) / diag;
}

extern "C" {

void launchTauExtract(const float* grad_log_tau, float* out, uint32_t siteCount, cudaStream_t stream) {
    int threads = 256;
    int blocks = (siteCount + threads - 1) / threads;
    tau_extract_kernel<<<blocks, threads, 0, stream>>>(grad_log_tau, out, siteCount);
}

void launchTauWriteback(const float* in, float* grad_log_tau, uint32_t siteCount, cudaStream_t stream) {
    int threads = 256;
    int blocks = (siteCount + threads - 1) / threads;
    tau_writeback_kernel<<<blocks, threads, 0, stream>>>(in, grad_log_tau, siteCount);
}

void launchTauDiffuse(
    const uint32_t* cand0, const uint32_t* cand1,
    const Site* sites, const float* grad_raw,
    const float* grad_in, float* grad_out,
    uint32_t siteCount, float lambda,
    int width, int height, int candDownscale, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (siteCount + threads - 1) / threads;
    tau_diffuse_kernel<<<blocks, threads, 0, stream>>>(
        cand0, cand1, sites, grad_raw, grad_in, grad_out,
        siteCount, lambda, width, height, candDownscale
    );
}

} // extern "C"
