#include "sad_common.cuh"

__global__ void packCandidateSitesKernel(
    const Site* __restrict__ sites,
    PackedCandidateSite* __restrict__ packed,
    uint32_t siteCount) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= siteCount) return;

    const Site site = sites[idx];
    PackedCandidateSite out;

    if (site.position.x < 0.0f) {
        __half neg = __float2half(-1.0f);
        out.a = __halves2half2(neg, neg);
        out.b = __halves2half2(__float2half(0.0f), __float2half(0.0f));
        out.c = __halves2half2(__float2half(0.0f), __float2half(0.0f));
        out.d = __halves2half2(__float2half(0.0f), __float2half(0.0f));
    } else {
        out.a = __halves2half2(__float2half(site.position.x), __float2half(site.position.y));
        out.b = __halves2half2(__float2half(site.log_tau), __float2half(site.radius));
        out.c = __halves2half2(__float2half(site.aniso_dir_x), __float2half(site.aniso_dir_y));
        out.d = __halves2half2(__float2half(site.log_aniso), __float2half(0.0f));
    }

    packed[idx] = out;
}

extern "C" {

void launchPackCandidateSites(
    const Site* sites,
    PackedCandidateSite* packed,
    uint32_t siteCount,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (siteCount + threads - 1) / threads;
    packCandidateSitesKernel<<<blocks, threads, 0, stream>>>(sites, packed, siteCount);
}

} // extern "C"
