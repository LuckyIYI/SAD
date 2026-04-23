#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "sad_common.cuh"

namespace {

inline cudaStream_t get_stream_for(const torch::Tensor& tensor) {
    const int device_index = tensor.get_device();
    c10::cuda::CUDAGuard device_guard(device_index);
    return at::cuda::getDefaultCUDAStream(device_index).stream();
}

inline cudaStream_t get_current_stream() {
    const int device_index = at::cuda::current_device();
    c10::cuda::CUDAGuard device_guard(device_index);
    return at::cuda::getDefaultCUDAStream(device_index).stream();
}

inline void check_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

inline void check_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

inline void check_int32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
}

inline void check_float16(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, " must be float16");
}

__device__ __forceinline__ bool compute_candidate_weights_raw(
    const uint32_t cand_ids[NUM_CANDIDATES],
    const Site* __restrict__ sites,
    uint32_t site_count,
    float2 uv,
    float inv_scale_sq,
    float weights[NUM_CANDIDATES],
    float3& pred)
{
    float logits[NUM_CANDIDATES];
    float max_logit = -INFINITY;
    bool any = false;

    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        uint32_t idx = cand_ids[i];
        if (idx >= site_count) {
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
        any = true;
    }

    if (!any || isinf(max_logit)) {
        return false;
    }

    float sum_w = 0.0f;
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        weights[i] = expf(logits[i] - max_logit);
        sum_w += weights[i];
    }
    float inv_sum = 1.0f / fmaxf(sum_w, 1e-8f);

    pred = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        weights[i] *= inv_sum;
        uint32_t idx = cand_ids[i];
        if (idx < site_count && sites[idx].position.x >= 0.0f) {
            float3 c = site_color(sites[idx]);
            pred.x += weights[i] * c.x;
            pred.y += weights[i] * c.y;
            pred.z += weights[i] * c.z;
        }
    }
    return true;
}

__global__ void render_sad_backward_kernel(
    const uint32_t* __restrict__ cand0,
    const uint32_t* __restrict__ cand1,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_sites,
    const Site* __restrict__ sites,
    float inv_scale_sq,
    uint32_t site_count,
    int width,
    int height,
    int cand_width,
    int cand_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;
    float2 uv = make_float2(float(x), float(y));

    int cand_idx = candidate_index_for_pixel(x, y, width, height, cand_width, cand_height);
    uint4 c0 = reinterpret_cast<const uint4*>(cand0)[cand_idx];
    uint4 c1 = reinterpret_cast<const uint4*>(cand1)[cand_idx];
    uint32_t cand_ids[NUM_CANDIDATES] = {
        c0.x, c0.y, c0.z, c0.w,
        c1.x, c1.y, c1.z, c1.w
    };

    float weights[NUM_CANDIDATES];
    float3 pred;
    if (!compute_candidate_weights_raw(cand_ids, sites, site_count, uv, inv_scale_sq, weights, pred)) {
        return;
    }

    float3 dL_dpred = make_float3(
        grad_output[pixel_idx * 3 + 0],
        grad_output[pixel_idx * 3 + 1],
        grad_output[pixel_idx * 3 + 2]);

    #pragma unroll
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        uint32_t idx = cand_ids[i];
        if (idx >= site_count) continue;
        Site site = sites[idx];
        if (site.position.x < 0.0f) continue;

        float w = weights[i];
        float3 dL_dcolor = make_float3(
            w * dL_dpred.x,
            w * dL_dpred.y,
            w * dL_dpred.z);
        float3 site_col = site_color(site);
        float3 diff_col = make_float3(site_col.x - pred.x,
                                      site_col.y - pred.y,
                                      site_col.z - pred.z);
        float dL_dlogit = dL_dpred.x * (w * diff_col.x)
                        + dL_dpred.y * (w * diff_col.y)
                        + dL_dpred.z * (w * diff_col.z);

        float tau = expf(site.log_tau);
        float2 diff = make_float2(uv.x - site.position.x, uv.y - site.position.y);
        float diff2 = dot(diff, diff);
        float proj = dot(site_aniso_dir(site), diff);
        float proj2 = proj * proj;
        float perp2 = fmaxf(diff2 - proj2, 0.0f);
        float l1 = expf(site.log_aniso);
        float l2 = 1.0f / l1;

        float d2_norm = (l1 * proj2 + l2 * perp2) * inv_scale_sq;
        float d2_safe = fmaxf(d2_norm, 1e-8f);
        float inv_sqrt_d2 = rsqrtf(d2_safe);
        float inv_scale = sqrtf(inv_scale_sq);
        float r_norm = site.radius * inv_scale;
        float dmix2 = (sqrtf(d2_safe) - r_norm);

        float2 g_diff = make_float2(
            l2 * diff.x + (l1 - l2) * site.aniso_dir_x * proj,
            l2 * diff.y + (l1 - l2) * site.aniso_dir_y * proj);
        float2 dL_dpos = make_float2(
            dL_dlogit * (tau * inv_scale_sq * inv_sqrt_d2) * g_diff.x,
            dL_dlogit * (tau * inv_scale_sq * inv_sqrt_d2) * g_diff.y);

        float dL_dlog_tau = dL_dlogit * (-dmix2) * tau;
        float dL_dradius = dL_dlogit * tau * inv_scale;

        float d2_dlog_aniso = (l1 * proj2 - l2 * perp2) * inv_scale_sq;
        float dL_dlog_aniso = dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_dlog_aniso;

        float2 d2_ddir = make_float2(
            (2.0f * (l1 - l2) * proj * inv_scale_sq) * diff.x,
            (2.0f * (l1 - l2) * proj * inv_scale_sq) * diff.y);
        float2 dL_ddir = make_float2(
            dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_ddir.x,
            dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_ddir.y);

        float* g = grad_sites + idx * 10u;
        atomicAdd(&g[0], dL_dpos.x);
        atomicAdd(&g[1], dL_dpos.y);
        atomicAdd(&g[2], dL_dlog_tau);
        atomicAdd(&g[3], dL_dradius);
        atomicAdd(&g[4], dL_dcolor.x);
        atomicAdd(&g[5], dL_dcolor.y);
        atomicAdd(&g[6], dL_dcolor.z);
        atomicAdd(&g[7], dL_ddir.x);
        atomicAdd(&g[8], dL_ddir.y);
        atomicAdd(&g[9], dL_dlog_aniso);
    }
}

__device__ __forceinline__ uint32_t hilbert_index_device(uint32_t x, uint32_t y, int bits) {
    uint32_t xi = x;
    uint32_t yi = y;
    uint32_t index = 0;
    uint32_t mask = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    for (int i = bits - 1; i >= 0; --i) {
        uint32_t shift = static_cast<uint32_t>(i);
        uint32_t rx = (xi >> shift) & 1u;
        uint32_t ry = (yi >> shift) & 1u;
        uint32_t d = (3u * rx) ^ ry;
        index |= d << (2u * shift);
        if (ry == 0u) {
            if (rx == 1u) {
                xi = mask - xi;
                yi = mask - yi;
            }
            uint32_t tmp = xi;
            xi = yi;
            yi = tmp;
        }
    }
    return index;
}

__global__ void compute_hilbert_pairs_kernel(
    const Site* __restrict__ sites,
    uint2* __restrict__ pairs,
    uint32_t site_count,
    uint32_t padded_count,
    int width,
    int height,
    int bits)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_count) return;
    if (idx < site_count) {
        float2 pos = sites[idx].position;
        int px = static_cast<int>(pos.x);
        int py = static_cast<int>(pos.y);
        px = px < 0 ? 0 : (px >= width ? (width - 1) : px);
        py = py < 0 ? 0 : (py >= height ? (height - 1) : py);
        uint32_t key = hilbert_index_device(static_cast<uint32_t>(px),
                                            static_cast<uint32_t>(py),
                                            bits);
        pairs[idx] = make_uint2(key, idx);
    } else {
        pairs[idx] = make_uint2(0xFFFFFFFFu, 0u);
    }
}

__global__ void write_hilbert_order_kernel(
    const uint2* __restrict__ sorted_pairs,
    uint32_t* __restrict__ order,
    uint32_t* __restrict__ pos,
    uint32_t site_count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= site_count) return;
    uint32_t site_idx = sorted_pairs[idx].y;
    order[idx] = site_idx;
    pos[site_idx] = idx;
}

} // namespace

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
    cudaStream_t stream);

void launchInitCandidates(
    uint32_t* cand0,
    uint32_t* cand1,
    uint32_t siteCount,
    uint32_t seed,
    bool perPixelMode,
    int width,
    int height,
    cudaStream_t stream);

void launchClearCandidates(
    uint32_t* cand0,
    uint32_t* cand1,
    int width,
    int height,
    cudaStream_t stream);

void launchJFASeed(
    uint32_t* cand0,
    const Site* sites,
    uint32_t siteCount,
    int width,
    int height,
    int candDownscale,
    cudaStream_t stream);

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
    cudaStream_t stream);

void launchPackCandidateSites(
    const Site* sites,
    PackedCandidateSite* packed,
    uint32_t siteCount,
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

void launchAdamUpdate(
    Site* sites,
    AdamState* adam,
    const float* grad_pos_x, const float* grad_pos_y,
    const float* grad_log_tau, const float* grad_radius,
    const float* grad_color_r, const float* grad_color_g, const float* grad_color_b,
    const float* grad_dir_x, const float* grad_dir_y, const float* grad_log_aniso,
    float lr_pos, float lr_tau, float lr_radius,
    float lr_color, float lr_dir, float lr_aniso,
    float beta1, float beta2, float eps,
    uint32_t t,
    uint32_t siteCount,
    int width,
    int height,
    cudaStream_t stream);

void launchClearBuffer(
    float* buffer,
    uint32_t count,
    cudaStream_t stream);

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
    cudaStream_t stream);

void launchComputeDensifyScorePairs(
    const Site* sites,
    const float* mass,
    const float* energy,
    uint2* pairs,
    uint32_t siteCount,
    float minMass,
    float scoreAlpha,
    uint32_t pairCount,
    cudaStream_t stream);

void launchComputePruneScorePairs(
    const Site* sites,
    const float* removal_delta,
    uint2* pairs,
    uint32_t siteCount,
    float deltaNorm,
    uint32_t pairCount,
    cudaStream_t stream);

void launchRadixSortUInt2(
    uint2* data,
    uint2* scratch,
    uint32_t* histFlat,
    uint32_t* blockSums,
    uint32_t paddedCount,
    uint32_t maxKeyExclusive,
    cudaStream_t stream);

void launchWriteSplitIndicesFromSorted(
    const uint2* sortedPairs,
    uint32_t* splitIndices,
    uint32_t numToWrite,
    cudaStream_t stream);

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
    cudaStream_t stream);

void launchPruneSitesByIndex(
    Site* sites,
    const uint32_t* indices,
    uint32_t count,
    cudaStream_t stream);

void launchTauDiffuse(
    const uint32_t* cand0, const uint32_t* cand1,
    const Site* sites, const float* grad_raw,
    const float* grad_in, float* grad_out,
    uint32_t siteCount, float lambda,
    int width, int height, int candDownscale, cudaStream_t stream);

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
    cudaStream_t stream);
}

torch::Tensor renderVoronoi(
    torch::Tensor cand0,
    torch::Tensor cand1,
    torch::Tensor sites,
    double inv_scale_sq,
    int64_t width,
    int64_t height)
{
    check_cuda(cand0, "cand0");
    check_cuda(cand1, "cand1");
    check_cuda(sites, "sites");
    check_int32(cand0, "cand0");
    check_int32(cand1, "cand1");
    check_float32(sites, "sites");
    TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
    TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(width > 0 && height > 0, "width/height must be positive");
    TORCH_CHECK(cand0.size(0) == width * height, "cand0 size must match width*height");
    TORCH_CHECK(cand1.size(0) == width * height, "cand1 size must match width*height");

    cand0 = cand0.contiguous();
    cand1 = cand1.contiguous();
    sites = sites.contiguous();

    auto output = torch::zeros({height, width, 3},
                               torch::TensorOptions().dtype(torch::kFloat32).device(cand0.device()));

    cudaStream_t stream = get_stream_for(cand0);
    launchRenderVoronoi(
        reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(output.data_ptr<float>()),
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        static_cast<float>(inv_scale_sq),
        static_cast<uint32_t>(sites.size(0)),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(height),
        stream);

    return output;
}

torch::Tensor renderVoronoiPadded(
    torch::Tensor cand0,
    torch::Tensor cand1,
    torch::Tensor sites,
    double inv_scale_sq,
    int64_t width,
    int64_t height,
    int64_t cand_width,
    int64_t cand_height)
{
    check_cuda(cand0, "cand0");
    check_cuda(cand1, "cand1");
    check_cuda(sites, "sites");
    check_int32(cand0, "cand0");
    check_int32(cand1, "cand1");
    check_float32(sites, "sites");
    TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
    TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(width > 0 && height > 0, "width/height must be positive");
    TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
    TORCH_CHECK(cand0.size(0) == cand_width * cand_height,
                "cand0 size must match cand_width*cand_height");
    TORCH_CHECK(cand1.size(0) == cand_width * cand_height,
                "cand1 size must match cand_width*cand_height");

    cand0 = cand0.contiguous();
    cand1 = cand1.contiguous();
    sites = sites.contiguous();

    auto output = torch::zeros({height, width, 3},
                               torch::TensorOptions().dtype(torch::kFloat32).device(cand0.device()));

    cudaStream_t stream = get_stream_for(cand0);
    launchRenderVoronoi(
        reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(output.data_ptr<float>()),
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        static_cast<float>(inv_scale_sq),
        static_cast<uint32_t>(sites.size(0)),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<int>(cand_width),
        static_cast<int>(cand_height),
        stream);

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> initCandidates(
    int64_t width,
    int64_t height,
    int64_t site_count,
    int64_t seed)
{
    TORCH_CHECK(width > 0 && height > 0, "invalid cand size");
    TORCH_CHECK(site_count >= 0, "site_count must be non-negative");

    int device_index = at::cuda::current_device();
    at::cuda::CUDAGuard device_guard(device_index);
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_index);

    int64_t count = width * height;
    auto cand0 = torch::zeros({count, 4}, opts);
    auto cand1 = torch::zeros({count, 4}, opts);

    cudaStream_t stream = get_current_stream();
    bool per_pixel = (site_count == count);
    launchInitCandidates(
        reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
        static_cast<uint32_t>(site_count),
        static_cast<uint32_t>(seed),
        per_pixel,
        static_cast<int>(width),
        static_cast<int>(height),
        stream);

    return std::make_tuple(cand0, cand1);
}

std::tuple<torch::Tensor, torch::Tensor> vptPass(
    torch::Tensor cand0,
    torch::Tensor cand1,
    torch::Tensor sites,
    double inv_scale_sq,
    int64_t width,
    int64_t height,
    int64_t jump,
    int64_t radius_probes,
    double radius_scale,
    int64_t inject_count,
    int64_t seed)
{
    (void)seed;
    check_cuda(cand0, "cand0");
    check_cuda(cand1, "cand1");
    check_cuda(sites, "sites");
    check_int32(cand0, "cand0");
    check_int32(cand1, "cand1");
    check_float32(sites, "sites");
    TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
    TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(cand0.size(0) == width * height, "cand0 size must match width*height");
    TORCH_CHECK(cand1.size(0) == width * height, "cand1 size must match width*height");

    cand0 = cand0.contiguous();
    cand1 = cand1.contiguous();
    sites = sites.contiguous();

    auto out0 = torch::zeros_like(cand0);
    auto out1 = torch::zeros_like(cand1);
    auto packed = torch::zeros({sites.size(0), 8},
                               torch::TensorOptions().dtype(torch::kFloat16).device(sites.device()));

    cudaStream_t stream = get_stream_for(cand0);
    launchPackCandidateSites(
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        reinterpret_cast<PackedCandidateSite*>(packed.data_ptr<at::Half>()),
        static_cast<uint32_t>(sites.size(0)),
        stream);

    uint32_t step = static_cast<uint32_t>(jump) << 16;
    uint32_t step_high = 0;
    uint32_t hilbert_probes = 0;
    uint32_t hilbert_window = 0;
    const uint32_t* hilbert_order = nullptr;
    const uint32_t* hilbert_pos = nullptr;
    launchUpdateCandidates(
        reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(out0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(out1.data_ptr<int32_t>()),
        reinterpret_cast<PackedCandidateSite*>(packed.data_ptr<at::Half>()),
        static_cast<uint32_t>(sites.size(0)),
        step,
        static_cast<float>(inv_scale_sq),
        step_high,
        static_cast<float>(radius_scale),
        static_cast<uint32_t>(radius_probes),
        static_cast<uint32_t>(inject_count),
        hilbert_order,
        hilbert_pos,
        hilbert_probes,
        hilbert_window,
        1,
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(height),
        stream);

    return std::make_tuple(out0, out1);
}

torch::Tensor renderVoronoiBackward(
    torch::Tensor cand0,
    torch::Tensor cand1,
    torch::Tensor sites,
    torch::Tensor grad_output,
    double inv_scale_sq,
    int64_t width,
    int64_t height)
{
    check_cuda(cand0, "cand0");
    check_cuda(cand1, "cand1");
    check_cuda(sites, "sites");
    check_cuda(grad_output, "grad_output");
    check_int32(cand0, "cand0");
    check_int32(cand1, "cand1");
    check_float32(sites, "sites");
    check_float32(grad_output, "grad_output");
    TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
    TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(2) == 3, "grad_output must be [H,W,3]");
    TORCH_CHECK(grad_output.size(0) == height && grad_output.size(1) == width,
                "grad_output size must match height/width");
    TORCH_CHECK(cand0.size(0) == width * height, "cand0 size must match width*height");
    TORCH_CHECK(cand1.size(0) == width * height, "cand1 size must match width*height");

    cand0 = cand0.contiguous();
    cand1 = cand1.contiguous();
    sites = sites.contiguous();
    grad_output = grad_output.contiguous();

    auto grad_sites = torch::zeros({sites.size(0), 10},
                                   torch::TensorOptions().dtype(torch::kFloat32).device(sites.device()));

    cudaStream_t stream = get_stream_for(cand0);
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    render_sad_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
        reinterpret_cast<float*>(grad_output.data_ptr<float>()),
        reinterpret_cast<float*>(grad_sites.data_ptr<float>()),
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        static_cast<float>(inv_scale_sq),
        static_cast<uint32_t>(sites.size(0)),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(height));

    return grad_sites;
}

torch::Tensor clearBuffer(torch::Tensor buffer, int64_t count) {
    check_cuda(buffer, "buffer");
    check_float32(buffer, "buffer");
    buffer = buffer.contiguous();
    int64_t numel = buffer.numel();
    if (count <= 0 || count > numel) {
        count = numel;
    }
    cudaStream_t stream = get_stream_for(buffer);
    launchClearBuffer(reinterpret_cast<float*>(buffer.data_ptr<float>()),
                      static_cast<uint32_t>(count),
                      stream);
    return buffer;
}

torch::Tensor packCandidateSites(torch::Tensor sites, torch::Tensor packed, int64_t site_count) {
    check_cuda(sites, "sites");
    check_cuda(packed, "packed");
    check_float32(sites, "sites");
    check_float16(packed, "packed");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(packed.dim() == 2 && packed.size(1) == 8, "packed must be [N,8]");
    sites = sites.contiguous();
    packed = packed.contiguous();
    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    cudaStream_t stream = get_stream_for(sites);
    launchPackCandidateSites(reinterpret_cast<Site*>(sites.data_ptr<float>()),
                             reinterpret_cast<PackedCandidateSite*>(packed.data_ptr<at::Half>()),
                             static_cast<uint32_t>(site_count),
                             stream);
    return packed;
}

torch::Tensor updateCandidatesCompact(torch::Tensor cand0_in, torch::Tensor cand1_in,
                                      torch::Tensor cand0_out, torch::Tensor cand1_out,
                                      torch::Tensor packed_sites, torch::Tensor hilbert_order,
                                      torch::Tensor hilbert_pos, double inv_scale_sq,
                                      int64_t site_count, int64_t step, int64_t step_high,
                                      double radius_scale, int64_t radius_probes, int64_t inject_count,
                                      int64_t hilbert_probes, int64_t hilbert_window,
                                      int64_t cand_downscale, int64_t target_width, int64_t target_height,
                                      int64_t cand_width, int64_t cand_height) {
    check_cuda(cand0_in, "cand0_in");
    check_cuda(cand1_in, "cand1_in");
    check_cuda(cand0_out, "cand0_out");
    check_cuda(cand1_out, "cand1_out");
    check_cuda(packed_sites, "packed_sites");
    check_int32(cand0_in, "cand0_in");
    check_int32(cand1_in, "cand1_in");
    check_int32(cand0_out, "cand0_out");
    check_int32(cand1_out, "cand1_out");
    check_float16(packed_sites, "packed_sites");
    TORCH_CHECK(cand0_in.dim() == 2 && cand0_in.size(1) == 4, "cand0_in must be [M,4]");
    TORCH_CHECK(cand1_in.dim() == 2 && cand1_in.size(1) == 4, "cand1_in must be [M,4]");
    TORCH_CHECK(cand0_out.dim() == 2 && cand0_out.size(1) == 4, "cand0_out must be [M,4]");
    TORCH_CHECK(cand1_out.dim() == 2 && cand1_out.size(1) == 4, "cand1_out must be [M,4]");
    TORCH_CHECK(packed_sites.dim() == 2 && packed_sites.size(1) == 8, "packed_sites must be [N,8]");
    TORCH_CHECK(cand0_in.size(0) == cand_width * cand_height, "cand0_in size mismatch");
    TORCH_CHECK(cand1_in.size(0) == cand_width * cand_height, "cand1_in size mismatch");
    TORCH_CHECK(cand0_out.size(0) == cand_width * cand_height, "cand0_out size mismatch");
    TORCH_CHECK(cand1_out.size(0) == cand_width * cand_height, "cand1_out size mismatch");

    cand0_in = cand0_in.contiguous();
    cand1_in = cand1_in.contiguous();
    cand0_out = cand0_out.contiguous();
    cand1_out = cand1_out.contiguous();
    packed_sites = packed_sites.contiguous();
    hilbert_order = hilbert_order.contiguous();
    hilbert_pos = hilbert_pos.contiguous();

    if (site_count <= 0) {
        site_count = packed_sites.size(0);
    }
    cudaStream_t stream = get_stream_for(cand0_in);
    const uint32_t* hilbert_order_ptr = nullptr;
    const uint32_t* hilbert_pos_ptr = nullptr;
    if (hilbert_probes > 0 && hilbert_window > 0) {
        check_cuda(hilbert_order, "hilbert_order");
        check_cuda(hilbert_pos, "hilbert_pos");
        check_int32(hilbert_order, "hilbert_order");
        check_int32(hilbert_pos, "hilbert_pos");
        hilbert_order_ptr = reinterpret_cast<uint32_t*>(hilbert_order.data_ptr<int32_t>());
        hilbert_pos_ptr = reinterpret_cast<uint32_t*>(hilbert_pos.data_ptr<int32_t>());
    }

    launchUpdateCandidates(
        reinterpret_cast<uint32_t*>(cand0_in.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1_in.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand0_out.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1_out.data_ptr<int32_t>()),
        reinterpret_cast<PackedCandidateSite*>(packed_sites.data_ptr<at::Half>()),
        static_cast<uint32_t>(site_count),
        static_cast<uint32_t>(step),
        static_cast<float>(inv_scale_sq),
        static_cast<uint32_t>(step_high),
        static_cast<float>(radius_scale),
        static_cast<uint32_t>(radius_probes),
        static_cast<uint32_t>(inject_count),
        hilbert_order_ptr,
        hilbert_pos_ptr,
        static_cast<uint32_t>(hilbert_probes),
        static_cast<uint32_t>(hilbert_window),
        static_cast<int>(cand_downscale),
        static_cast<int>(target_width),
        static_cast<int>(target_height),
        static_cast<int>(cand_width),
        static_cast<int>(cand_height),
        stream);

    return cand0_out;
}

torch::Tensor jfaClear(torch::Tensor cand0, torch::Tensor cand1, int64_t cand_width, int64_t cand_height) {
    check_cuda(cand0, "cand0");
    check_cuda(cand1, "cand1");
    check_int32(cand0, "cand0");
    check_int32(cand1, "cand1");
    TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
    TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
    cand0 = cand0.contiguous();
    cand1 = cand1.contiguous();
    cudaStream_t stream = get_stream_for(cand0);
    launchClearCandidates(reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
                          reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
                          static_cast<int>(cand_width),
                          static_cast<int>(cand_height),
                          stream);
    return cand0;
}

torch::Tensor jfaSeed(torch::Tensor cand0, torch::Tensor sites, int64_t site_count,
                      int64_t cand_downscale, int64_t cand_width, int64_t cand_height) {
    check_cuda(cand0, "cand0");
    check_cuda(sites, "sites");
    check_int32(cand0, "cand0");
    check_float32(sites, "sites");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
    TORCH_CHECK(cand0.size(0) == cand_width * cand_height, "cand0 size must match cand_width*cand_height");
    cand0 = cand0.contiguous();
    sites = sites.contiguous();
    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    cudaStream_t stream = get_stream_for(cand0);
    launchJFASeed(reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
                  reinterpret_cast<Site*>(sites.data_ptr<float>()),
                  static_cast<uint32_t>(site_count),
                  static_cast<int>(cand_width),
                  static_cast<int>(cand_height),
                  static_cast<int>(cand_downscale),
                  stream);
    return cand0;
}

torch::Tensor jfaFlood(torch::Tensor cand0_in, torch::Tensor cand0_out, torch::Tensor sites,
                       double inv_scale_sq, int64_t site_count, int64_t step_size,
                       int64_t cand_downscale, int64_t cand_width, int64_t cand_height) {
    check_cuda(cand0_in, "cand0_in");
    check_cuda(cand0_out, "cand0_out");
    check_cuda(sites, "sites");
    check_int32(cand0_in, "cand0_in");
    check_int32(cand0_out, "cand0_out");
    check_float32(sites, "sites");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(cand0_in.dim() == 2 && cand0_in.size(1) == 4, "cand0_in must be [M,4]");
    TORCH_CHECK(cand0_out.dim() == 2 && cand0_out.size(1) == 4, "cand0_out must be [M,4]");
    TORCH_CHECK(cand0_in.size(0) == cand_width * cand_height, "cand0_in size must match cand_width*cand_height");
    TORCH_CHECK(cand0_out.size(0) == cand_width * cand_height, "cand0_out size must match cand_width*cand_height");
    cand0_in = cand0_in.contiguous();
    cand0_out = cand0_out.contiguous();
    sites = sites.contiguous();
    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    cudaStream_t stream = get_stream_for(cand0_in);
    launchJFAFlood(reinterpret_cast<uint32_t*>(cand0_in.data_ptr<int32_t>()),
                   reinterpret_cast<uint32_t*>(cand0_out.data_ptr<int32_t>()),
                   reinterpret_cast<Site*>(sites.data_ptr<float>()),
                   static_cast<uint32_t>(site_count),
                   static_cast<uint32_t>(step_size),
                   static_cast<float>(inv_scale_sq),
                   static_cast<int>(cand_width),
                   static_cast<int>(cand_height),
                   static_cast<int>(cand_downscale),
                   static_cast<int>(cand_width * cand_downscale),
                   static_cast<int>(cand_height * cand_downscale),
                   stream);
    return cand0_out;
}

torch::Tensor computeGradientsTiled(torch::Tensor cand0, torch::Tensor cand1,
                                    torch::Tensor target, torch::Tensor rendered, torch::Tensor mask,
                                    torch::Tensor sites,
                                    torch::Tensor grad_pos_x, torch::Tensor grad_pos_y,
                                    torch::Tensor grad_log_tau, torch::Tensor grad_radius,
                                    torch::Tensor grad_color_r, torch::Tensor grad_color_g, torch::Tensor grad_color_b,
                                    torch::Tensor grad_dir_x, torch::Tensor grad_dir_y, torch::Tensor grad_log_aniso,
                                    torch::Tensor removal_delta,
                                    double inv_scale_sq, int64_t site_count,
                                    int64_t compute_removal, double ssim_weight,
                                    int64_t cand_width, int64_t cand_height)
{
    check_cuda(cand0, "cand0");
    check_cuda(cand1, "cand1");
    check_cuda(target, "target");
    check_cuda(rendered, "rendered");
    check_cuda(mask, "mask");
    check_cuda(sites, "sites");
    check_float32(target, "target");
    check_float32(rendered, "rendered");
    check_float32(mask, "mask");
    check_float32(sites, "sites");
    check_int32(cand0, "cand0");
    check_int32(cand1, "cand1");
    TORCH_CHECK(target.dim() == 3 && target.size(2) == 3, "target must be [H,W,3]");
    TORCH_CHECK(rendered.dim() == 3 && rendered.size(2) == 3, "rendered must be [H,W,3]");
    TORCH_CHECK(mask.dim() == 2, "mask must be [H,W]");
    TORCH_CHECK(mask.size(0) == target.size(0) && mask.size(1) == target.size(1),
                "mask size must match target");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    cand0 = cand0.contiguous();
    cand1 = cand1.contiguous();
    target = target.contiguous();
    rendered = rendered.contiguous();
    mask = mask.contiguous();
    sites = sites.contiguous();

    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    cudaStream_t stream = get_stream_for(cand0);
    launchComputeGradientsTiled(
        reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(target.data_ptr<float>()),
        reinterpret_cast<float3*>(rendered.data_ptr<float>()),
        reinterpret_cast<float*>(mask.data_ptr<float>()),
        reinterpret_cast<float*>(grad_pos_x.data_ptr<float>()),
        reinterpret_cast<float*>(grad_pos_y.data_ptr<float>()),
        reinterpret_cast<float*>(grad_log_tau.data_ptr<float>()),
        reinterpret_cast<float*>(grad_radius.data_ptr<float>()),
        reinterpret_cast<float*>(grad_color_r.data_ptr<float>()),
        reinterpret_cast<float*>(grad_color_g.data_ptr<float>()),
        reinterpret_cast<float*>(grad_color_b.data_ptr<float>()),
        reinterpret_cast<float*>(grad_dir_x.data_ptr<float>()),
        reinterpret_cast<float*>(grad_dir_y.data_ptr<float>()),
        reinterpret_cast<float*>(grad_log_aniso.data_ptr<float>()),
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        static_cast<float>(inv_scale_sq),
        static_cast<uint32_t>(site_count),
        reinterpret_cast<float*>(removal_delta.data_ptr<float>()),
        static_cast<uint32_t>(compute_removal),
        static_cast<float>(ssim_weight),
        static_cast<int>(target.size(1)),
        static_cast<int>(target.size(0)),
        static_cast<int>(cand_width),
        static_cast<int>(cand_height),
        stream);
    return grad_pos_x;
}

torch::Tensor tauDiffuse(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                         torch::Tensor grad_raw, torch::Tensor grad_in, torch::Tensor grad_out,
                         int64_t site_count, double lambda, int64_t cand_downscale,
                         int64_t cand_width, int64_t cand_height)
{
    check_cuda(cand0, "cand0");
    check_cuda(cand1, "cand1");
    check_cuda(sites, "sites");
    check_cuda(grad_raw, "grad_raw");
    check_cuda(grad_in, "grad_in");
    check_cuda(grad_out, "grad_out");
    check_float32(sites, "sites");
    check_float32(grad_raw, "grad_raw");
    check_float32(grad_in, "grad_in");
    check_float32(grad_out, "grad_out");
    check_int32(cand0, "cand0");
    check_int32(cand1, "cand1");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    cand0 = cand0.contiguous();
    cand1 = cand1.contiguous();
    sites = sites.contiguous();
    grad_raw = grad_raw.contiguous();
    grad_in = grad_in.contiguous();
    grad_out = grad_out.contiguous();
    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    cudaStream_t stream = get_stream_for(cand0);
    launchTauDiffuse(
        reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        reinterpret_cast<float*>(grad_raw.data_ptr<float>()),
        reinterpret_cast<float*>(grad_in.data_ptr<float>()),
        reinterpret_cast<float*>(grad_out.data_ptr<float>()),
        static_cast<uint32_t>(site_count),
        static_cast<float>(lambda),
        static_cast<int>(cand_width),
        static_cast<int>(cand_height),
        static_cast<int>(cand_downscale),
        stream);
    return grad_out;
}

torch::Tensor adamUpdate(torch::Tensor sites, torch::Tensor adam,
                         torch::Tensor grad_pos_x, torch::Tensor grad_pos_y,
                         torch::Tensor grad_log_tau, torch::Tensor grad_radius,
                         torch::Tensor grad_color_r, torch::Tensor grad_color_g, torch::Tensor grad_color_b,
                         torch::Tensor grad_dir_x, torch::Tensor grad_dir_y, torch::Tensor grad_log_aniso,
                         double lr_pos, double lr_tau, double lr_radius, double lr_color,
                         double lr_dir, double lr_aniso, double beta1, double beta2, double eps,
                         int64_t t, int64_t width, int64_t height) {
    check_cuda(sites, "sites");
    check_cuda(adam, "adam");
    check_float32(sites, "sites");
    check_float32(adam, "adam");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(adam.dim() == 2 && adam.size(1) == 20, "adam must be [N,20]");
    sites = sites.contiguous();
    adam = adam.contiguous();
    cudaStream_t stream = get_stream_for(sites);
    launchAdamUpdate(
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        reinterpret_cast<AdamState*>(adam.data_ptr<float>()),
        reinterpret_cast<float*>(grad_pos_x.data_ptr<float>()),
        reinterpret_cast<float*>(grad_pos_y.data_ptr<float>()),
        reinterpret_cast<float*>(grad_log_tau.data_ptr<float>()),
        reinterpret_cast<float*>(grad_radius.data_ptr<float>()),
        reinterpret_cast<float*>(grad_color_r.data_ptr<float>()),
        reinterpret_cast<float*>(grad_color_g.data_ptr<float>()),
        reinterpret_cast<float*>(grad_color_b.data_ptr<float>()),
        reinterpret_cast<float*>(grad_dir_x.data_ptr<float>()),
        reinterpret_cast<float*>(grad_dir_y.data_ptr<float>()),
        reinterpret_cast<float*>(grad_log_aniso.data_ptr<float>()),
        static_cast<float>(lr_pos),
        static_cast<float>(lr_tau),
        static_cast<float>(lr_radius),
        static_cast<float>(lr_color),
        static_cast<float>(lr_dir),
        static_cast<float>(lr_aniso),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(eps),
        static_cast<uint32_t>(t),
        static_cast<uint32_t>(sites.size(0)),
        static_cast<int>(width),
        static_cast<int>(height),
        stream);
    return sites;
}

torch::Tensor computeSiteStatsTiled(torch::Tensor cand0, torch::Tensor cand1,
                                    torch::Tensor target, torch::Tensor mask, torch::Tensor sites,
                                    torch::Tensor mass, torch::Tensor energy,
                                    torch::Tensor err_w, torch::Tensor err_wx, torch::Tensor err_wy,
                                    torch::Tensor err_wxx, torch::Tensor err_wxy, torch::Tensor err_wyy,
                                    double inv_scale_sq, int64_t site_count,
                                    int64_t cand_width, int64_t cand_height)
{
    check_cuda(cand0, "cand0");
    check_cuda(cand1, "cand1");
    check_cuda(target, "target");
    check_cuda(mask, "mask");
    check_cuda(sites, "sites");
    check_float32(target, "target");
    check_float32(mask, "mask");
    check_float32(sites, "sites");
    check_int32(cand0, "cand0");
    check_int32(cand1, "cand1");
    TORCH_CHECK(target.dim() == 3 && target.size(2) == 3, "target must be [H,W,3]");
    TORCH_CHECK(mask.dim() == 2, "mask must be [H,W]");
    TORCH_CHECK(mask.size(0) == target.size(0) && mask.size(1) == target.size(1),
                "mask size must match target");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    cand0 = cand0.contiguous();
    cand1 = cand1.contiguous();
    target = target.contiguous();
    mask = mask.contiguous();
    sites = sites.contiguous();

    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    cudaStream_t stream = get_stream_for(cand0);
    launchComputeSiteStatsSimple(
        reinterpret_cast<uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(target.data_ptr<float>()),
        reinterpret_cast<float*>(mask.data_ptr<float>()),
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        static_cast<float>(inv_scale_sq),
        static_cast<uint32_t>(site_count),
        reinterpret_cast<float*>(mass.data_ptr<float>()),
        reinterpret_cast<float*>(energy.data_ptr<float>()),
        reinterpret_cast<float*>(err_w.data_ptr<float>()),
        reinterpret_cast<float*>(err_wx.data_ptr<float>()),
        reinterpret_cast<float*>(err_wy.data_ptr<float>()),
        reinterpret_cast<float*>(err_wxx.data_ptr<float>()),
        reinterpret_cast<float*>(err_wxy.data_ptr<float>()),
        reinterpret_cast<float*>(err_wyy.data_ptr<float>()),
        static_cast<int>(target.size(1)),
        static_cast<int>(target.size(0)),
        static_cast<int>(cand_width),
        static_cast<int>(cand_height),
        stream);
    return mass;
}

torch::Tensor computeDensifyScorePairs(torch::Tensor sites, torch::Tensor mass, torch::Tensor energy,
                                       torch::Tensor pairs, int64_t site_count,
                                       double min_mass, double score_alpha, int64_t pair_count) {
    check_cuda(sites, "sites");
    check_cuda(mass, "mass");
    check_cuda(energy, "energy");
    check_cuda(pairs, "pairs");
    check_float32(sites, "sites");
    check_float32(mass, "mass");
    check_float32(energy, "energy");
    check_int32(pairs, "pairs");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [M,2]");
    sites = sites.contiguous();
    mass = mass.contiguous();
    energy = energy.contiguous();
    pairs = pairs.contiguous();
    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    if (pair_count <= 0) {
        pair_count = pairs.size(0);
    }
    cudaStream_t stream = get_stream_for(sites);
    launchComputeDensifyScorePairs(
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        reinterpret_cast<float*>(mass.data_ptr<float>()),
        reinterpret_cast<float*>(energy.data_ptr<float>()),
        reinterpret_cast<uint2*>(pairs.data_ptr<int32_t>()),
        static_cast<uint32_t>(site_count),
        static_cast<float>(min_mass),
        static_cast<float>(score_alpha),
        static_cast<uint32_t>(pair_count),
        stream);
    return pairs;
}

torch::Tensor computePruneScorePairs(torch::Tensor sites, torch::Tensor removal_delta, torch::Tensor pairs,
                                     int64_t site_count, double delta_norm, int64_t pair_count) {
    check_cuda(sites, "sites");
    check_cuda(removal_delta, "removal_delta");
    check_cuda(pairs, "pairs");
    check_float32(sites, "sites");
    check_float32(removal_delta, "removal_delta");
    check_int32(pairs, "pairs");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [M,2]");
    sites = sites.contiguous();
    removal_delta = removal_delta.contiguous();
    pairs = pairs.contiguous();
    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    if (pair_count <= 0) {
        pair_count = pairs.size(0);
    }
    cudaStream_t stream = get_stream_for(sites);
    launchComputePruneScorePairs(
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        reinterpret_cast<float*>(removal_delta.data_ptr<float>()),
        reinterpret_cast<uint2*>(pairs.data_ptr<int32_t>()),
        static_cast<uint32_t>(site_count),
        static_cast<float>(delta_norm),
        static_cast<uint32_t>(pair_count),
        stream);
    return pairs;
}

torch::Tensor radixSortPairs(torch::Tensor pairs, int64_t max_key_exclusive) {
    check_cuda(pairs, "pairs");
    check_int32(pairs, "pairs");
    TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [N,2]");
    pairs = pairs.contiguous();
    int64_t count = pairs.size(0);
    TORCH_CHECK(count > 0, "pairs must be non-empty");

    auto scratch = torch::zeros_like(pairs);

    constexpr uint32_t kBlockSize = 256;
    constexpr uint32_t kBuckets = 256;
    constexpr uint32_t kElementsPerBlock = 1024;
    uint32_t padded_count = static_cast<uint32_t>(count);
    uint32_t grid_size = (padded_count + kElementsPerBlock - 1) / kElementsPerBlock;
    uint32_t hist_length = grid_size * kBuckets;
    uint32_t hist_blocks = (hist_length + kBlockSize - 1) / kBlockSize;

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(pairs.device());
    auto hist_flat = torch::zeros({static_cast<int64_t>(hist_length)}, opts);
    auto block_sums = torch::zeros({static_cast<int64_t>(hist_blocks)}, opts);

    cudaStream_t stream = get_stream_for(pairs);
    launchRadixSortUInt2(
        reinterpret_cast<uint2*>(pairs.data_ptr<int32_t>()),
        reinterpret_cast<uint2*>(scratch.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(hist_flat.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(block_sums.data_ptr<int32_t>()),
        padded_count,
        static_cast<uint32_t>(max_key_exclusive),
        stream);

    return pairs;
}

torch::Tensor writeSplitIndices(torch::Tensor pairs, torch::Tensor indices, int64_t num_to_split) {
    check_cuda(pairs, "pairs");
    check_cuda(indices, "indices");
    check_int32(pairs, "pairs");
    check_int32(indices, "indices");
    TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [N,2]");
    TORCH_CHECK(indices.dim() == 1, "indices must be [K]");
    pairs = pairs.contiguous();
    indices = indices.contiguous();
    cudaStream_t stream = get_stream_for(pairs);
    launchWriteSplitIndicesFromSorted(
        reinterpret_cast<uint2*>(pairs.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(indices.data_ptr<int32_t>()),
        static_cast<uint32_t>(num_to_split),
        stream);
    return indices;
}

torch::Tensor splitSites(torch::Tensor sites, torch::Tensor adam, torch::Tensor split_indices,
                         torch::Tensor mass, torch::Tensor err_w, torch::Tensor err_wx, torch::Tensor err_wy,
                         torch::Tensor err_wxx, torch::Tensor err_wxy, torch::Tensor err_wyy,
                         int64_t current_site_count, int64_t num_to_split, torch::Tensor target) {
    check_cuda(sites, "sites");
    check_cuda(adam, "adam");
    check_cuda(split_indices, "split_indices");
    check_cuda(target, "target");
    check_float32(sites, "sites");
    check_float32(adam, "adam");
    check_int32(split_indices, "split_indices");
    check_float32(target, "target");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(adam.dim() == 2 && adam.size(1) == 20, "adam must be [N,20]");
    TORCH_CHECK(target.dim() == 3 && target.size(2) == 3, "target must be [H,W,3]");
    sites = sites.contiguous();
    adam = adam.contiguous();
    split_indices = split_indices.contiguous();
    target = target.contiguous();
    cudaStream_t stream = get_stream_for(sites);
    launchSplitSites(
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        reinterpret_cast<AdamState*>(adam.data_ptr<float>()),
        reinterpret_cast<uint32_t*>(split_indices.data_ptr<int32_t>()),
        static_cast<uint32_t>(num_to_split),
        reinterpret_cast<float*>(mass.data_ptr<float>()),
        reinterpret_cast<float*>(err_w.data_ptr<float>()),
        reinterpret_cast<float*>(err_wx.data_ptr<float>()),
        reinterpret_cast<float*>(err_wy.data_ptr<float>()),
        reinterpret_cast<float*>(err_wxx.data_ptr<float>()),
        reinterpret_cast<float*>(err_wxy.data_ptr<float>()),
        reinterpret_cast<float*>(err_wyy.data_ptr<float>()),
        static_cast<uint32_t>(current_site_count),
        reinterpret_cast<float3*>(target.data_ptr<float>()),
        static_cast<int>(target.size(1)),
        static_cast<int>(target.size(0)),
        stream);
    return sites;
}

torch::Tensor pruneSites(torch::Tensor sites, torch::Tensor indices, int64_t count) {
    check_cuda(sites, "sites");
    check_cuda(indices, "indices");
    check_float32(sites, "sites");
    check_int32(indices, "indices");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    sites = sites.contiguous();
    indices = indices.contiguous();
    cudaStream_t stream = get_stream_for(sites);
    launchPruneSitesByIndex(
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        reinterpret_cast<uint32_t*>(indices.data_ptr<int32_t>()),
        static_cast<uint32_t>(count),
        stream);
    return sites;
}

torch::Tensor buildHilbertPairs(torch::Tensor sites, torch::Tensor pairs,
                                int64_t site_count, int64_t padded_count,
                                int64_t width, int64_t height, int64_t bits) {
    check_cuda(sites, "sites");
    check_cuda(pairs, "pairs");
    check_float32(sites, "sites");
    check_int32(pairs, "pairs");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [M,2]");
    sites = sites.contiguous();
    pairs = pairs.contiguous();
    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    if (padded_count <= 0) {
        padded_count = pairs.size(0);
    } else {
        TORCH_CHECK(pairs.size(0) == padded_count, "pairs size must match padded_count");
    }
    cudaStream_t stream = get_stream_for(sites);
    int threads = 256;
    int blocks = static_cast<int>((padded_count + threads - 1) / threads);
    compute_hilbert_pairs_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        reinterpret_cast<uint2*>(pairs.data_ptr<int32_t>()),
        static_cast<uint32_t>(site_count),
        static_cast<uint32_t>(padded_count),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<int>(bits));
    return pairs;
}

torch::Tensor writeHilbertOrder(torch::Tensor pairs, torch::Tensor order, torch::Tensor pos, int64_t site_count) {
    check_cuda(pairs, "pairs");
    check_cuda(order, "order");
    check_cuda(pos, "pos");
    check_int32(pairs, "pairs");
    check_int32(order, "order");
    check_int32(pos, "pos");
    TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [N,2]");
    pairs = pairs.contiguous();
    order = order.contiguous();
    pos = pos.contiguous();
    if (site_count <= 0) {
        site_count = order.size(0);
    }
    cudaStream_t stream = get_stream_for(pairs);
    int threads = 256;
    int blocks = static_cast<int>((site_count + threads - 1) / threads);
    write_hilbert_order_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint2*>(pairs.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(order.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(pos.data_ptr<int32_t>()),
        static_cast<uint32_t>(site_count));
    return order;
}

torch::Tensor initGradientWeighted(torch::Tensor sites, torch::Tensor target, torch::Tensor mask,
                                   int64_t site_count, double init_log_tau, double init_radius,
                                   double gradient_alpha) {
    check_cuda(sites, "sites");
    check_cuda(target, "target");
    check_cuda(mask, "mask");
    check_float32(sites, "sites");
    check_float32(target, "target");
    check_float32(mask, "mask");
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(target.dim() == 3 && target.size(2) == 3, "target must be [H,W,3]");
    TORCH_CHECK(mask.dim() == 2, "mask must be [H,W]");
    TORCH_CHECK(mask.size(0) == target.size(0) && mask.size(1) == target.size(1),
                "mask size must match target");
    sites = sites.contiguous();
    target = target.contiguous();
    mask = mask.contiguous();
    if (site_count <= 0) {
        site_count = sites.size(0);
    }
    auto seed_counter = torch::zeros({1},
                                     torch::TensorOptions().dtype(torch::kInt32).device(sites.device()));
    cudaStream_t stream = get_stream_for(sites);
    float grad_threshold = 0.01f * static_cast<float>(gradient_alpha);
    launchInitGradientWeighted(
        reinterpret_cast<Site*>(sites.data_ptr<float>()),
        static_cast<uint32_t>(site_count),
        reinterpret_cast<uint32_t*>(seed_counter.data_ptr<int32_t>()),
        reinterpret_cast<float3*>(target.data_ptr<float>()),
        reinterpret_cast<float*>(mask.data_ptr<float>()),
        grad_threshold,
        256,
        static_cast<float>(init_log_tau),
        static_cast<float>(init_radius),
        static_cast<int>(target.size(1)),
        static_cast<int>(target.size(0)),
        stream);
    return sites;
}
