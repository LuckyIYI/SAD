#include "sad_common.cuh"
#include <cuda_runtime.h>

// Adam optimizer update step
__global__ void adamUpdateKernel(
    Site* __restrict__ sites,
    AdamState* __restrict__ adam,
    const float* __restrict__ grad_pos_x,
    const float* __restrict__ grad_pos_y,
    const float* __restrict__ grad_log_tau,
    const float* __restrict__ grad_radius,
    const float* __restrict__ grad_color_r,
    const float* __restrict__ grad_color_g,
    const float* __restrict__ grad_color_b,
    const float* __restrict__ grad_dir_x,
    const float* __restrict__ grad_dir_y,
    const float* __restrict__ grad_log_aniso,
    float lr_pos,
    float lr_tau,
    float lr_radius,
    float lr_color,
    float lr_dir,
    float lr_aniso,
    float beta1,
    float beta2,
    float eps,
    uint32_t t,
    uint32_t siteCount,
    int width,
    int height
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against overrun when siteCount is not a multiple of blockDim.x
    if (gid >= siteCount) return;
    
    // Normalize gradients by image size
    float norm = 1.0f / float(width * height);
    
    float2 g_pos = make_float2(grad_pos_x[gid] * norm, grad_pos_y[gid] * norm);
    float g_log_tau = grad_log_tau[gid] * norm;
    float g_radius = grad_radius[gid] * norm;
    float3 g_color = make_float3(
        grad_color_r[gid] * norm,
        grad_color_g[gid] * norm,
        grad_color_b[gid] * norm
    );
    float2 g_dir = make_float2(grad_dir_x[gid] * norm, grad_dir_y[gid] * norm);
    float g_log_aniso = grad_log_aniso[gid] * norm;
    
    // Skip inactive sites
    if (sites[gid].position.x < 0.0f) return;
    
    // Adam bias correction
    float tt = float(t);
    float b1t_corr = 1.0f / (1.0f - powf(beta1, tt));
    float b2t_corr = 1.0f / (1.0f - powf(beta2, tt));
    
    // Position (IsotropicAdam)
    adam[gid].m_pos.x = beta1 * adam[gid].m_pos.x + (1.0f - beta1) * g_pos.x;
    adam[gid].m_pos.y = beta1 * adam[gid].m_pos.y + (1.0f - beta1) * g_pos.y;
    float g2_pos = dot(g_pos, g_pos);
    adam[gid].v_pos = beta2 * adam[gid].v_pos + (1.0f - beta2) * g2_pos;
    
    float2 m_hat_pos = make_float2(
        adam[gid].m_pos.x * b1t_corr,
        adam[gid].m_pos.y * b1t_corr
    );
    float v_hat_pos = adam[gid].v_pos * b2t_corr;
    float inv_denom_pos = 1.0f / (sqrtf(v_hat_pos) + eps);
    float2 step_pos = make_float2(
        lr_pos * m_hat_pos.x * inv_denom_pos,
        lr_pos * m_hat_pos.y * inv_denom_pos
    );
    
    sites[gid].position.x -= step_pos.x;
    sites[gid].position.y -= step_pos.y;
    sites[gid].position = clamp2(
        sites[gid].position,
        make_float2(0.0f, 0.0f),
        make_float2(float(width - 1), float(height - 1))
    );
    
    // log_tau
    adam[gid].m_log_tau = beta1 * adam[gid].m_log_tau + (1.0f - beta1) * g_log_tau;
    adam[gid].v_log_tau = beta2 * adam[gid].v_log_tau + (1.0f - beta2) * (g_log_tau * g_log_tau);
    float m_hat_tau = adam[gid].m_log_tau * b1t_corr;
    float v_hat_tau = adam[gid].v_log_tau * b2t_corr;
    float step_tau = lr_tau * m_hat_tau / (sqrtf(v_hat_tau) + eps);
    
    sites[gid].log_tau -= step_tau;
    
    // radius
    adam[gid].m_radius = beta1 * adam[gid].m_radius + (1.0f - beta1) * g_radius;
    adam[gid].v_radius = beta2 * adam[gid].v_radius + (1.0f - beta2) * (g_radius * g_radius);
    float m_hat_rad = adam[gid].m_radius * b1t_corr;
    float v_hat_rad = adam[gid].v_radius * b2t_corr;
    float step_rad = lr_radius * m_hat_rad / (sqrtf(v_hat_rad) + eps);
    
    sites[gid].radius -= step_rad;
    sites[gid].radius = clampf(sites[gid].radius, 1.0f, 512.0f);
    
    // Color (per-channel Adam)
    adam[gid].m_color.x = beta1 * adam[gid].m_color.x + (1.0f - beta1) * g_color.x;
    adam[gid].m_color.y = beta1 * adam[gid].m_color.y + (1.0f - beta1) * g_color.y;
    adam[gid].m_color.z = beta1 * adam[gid].m_color.z + (1.0f - beta1) * g_color.z;
    
    adam[gid].v_color.x = beta2 * adam[gid].v_color.x + (1.0f - beta2) * (g_color.x * g_color.x);
    adam[gid].v_color.y = beta2 * adam[gid].v_color.y + (1.0f - beta2) * (g_color.y * g_color.y);
    adam[gid].v_color.z = beta2 * adam[gid].v_color.z + (1.0f - beta2) * (g_color.z * g_color.z);
    
    float3 m_hat_col = make_float3(
        adam[gid].m_color.x * b1t_corr,
        adam[gid].m_color.y * b1t_corr,
        adam[gid].m_color.z * b1t_corr
    );
    float3 v_hat_col = make_float3(
        adam[gid].v_color.x * b2t_corr,
        adam[gid].v_color.y * b2t_corr,
        adam[gid].v_color.z * b2t_corr
    );
    float3 step_col = make_float3(
        lr_color * m_hat_col.x / (sqrtf(v_hat_col.x) + eps),
        lr_color * m_hat_col.y / (sqrtf(v_hat_col.y) + eps),
        lr_color * m_hat_col.z / (sqrtf(v_hat_col.z) + eps)
    );
    
    float3 curr_color = site_color(sites[gid]);
    curr_color.x -= step_col.x;
    curr_color.y -= step_col.y;
    curr_color.z -= step_col.z;
    curr_color = clamp3(curr_color, make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
    site_set_color(sites[gid], curr_color);
    
    // Anisotropy direction (tangent space + IsotropicAdam)
    float2 dir = site_aniso_dir(sites[gid]);
    float2 g_tan = make_float2(
        g_dir.x - dir.x * dot(g_dir, dir),
        g_dir.y - dir.y * dot(g_dir, dir)
    );
    
    adam[gid].m_dir.x = beta1 * adam[gid].m_dir.x + (1.0f - beta1) * g_tan.x;
    adam[gid].m_dir.y = beta1 * adam[gid].m_dir.y + (1.0f - beta1) * g_tan.y;
    float g2_dir = dot(g_tan, g_tan);
    adam[gid].v_dir = beta2 * adam[gid].v_dir + (1.0f - beta2) * g2_dir;
    
    float2 m_hat_dir = make_float2(
        adam[gid].m_dir.x * b1t_corr,
        adam[gid].m_dir.y * b1t_corr
    );
    float v_hat_dir = adam[gid].v_dir * b2t_corr;
    float inv_denom_dir = 1.0f / (sqrtf(v_hat_dir) + eps);
    float2 step_dir = make_float2(
        lr_dir * m_hat_dir.x * inv_denom_dir,
        lr_dir * m_hat_dir.y * inv_denom_dir
    );
    
    float2 new_dir = make_float2(dir.x - step_dir.x, dir.y - step_dir.y);
    float len2 = dot(new_dir, new_dir);
    if (len2 < 1e-12f) {
        new_dir = make_float2(1.0f, 0.0f);
    } else {
        float inv_len = rsqrtf(len2);
        new_dir.x *= inv_len;
        new_dir.y *= inv_len;
    }
    site_set_aniso_dir(sites[gid], new_dir);
    
    // log_aniso
    adam[gid].m_log_aniso = beta1 * adam[gid].m_log_aniso + (1.0f - beta1) * g_log_aniso;
    adam[gid].v_log_aniso = beta2 * adam[gid].v_log_aniso + (1.0f - beta2) * (g_log_aniso * g_log_aniso);
    float m_hat_aniso = adam[gid].m_log_aniso * b1t_corr;
    float v_hat_aniso = adam[gid].v_log_aniso * b2t_corr;
    float step_aniso = lr_aniso * m_hat_aniso / (sqrtf(v_hat_aniso) + eps);
    
    sites[gid].log_aniso -= step_aniso;

    // Clamp to prevent extreme anisotropy
    sites[gid].log_aniso = clampf(sites[gid].log_aniso, -4.0f, 4.0f);
}

// Clear gradients kernel
__global__ void clearGradientsKernel(
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
    uint32_t count
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    
    grad_pos_x[gid] = 0.0f;
    grad_pos_y[gid] = 0.0f;
    grad_log_tau[gid] = 0.0f;
    grad_radius[gid] = 0.0f;
    grad_color_r[gid] = 0.0f;
    grad_color_g[gid] = 0.0f;
    grad_color_b[gid] = 0.0f;
    grad_dir_x[gid] = 0.0f;
    grad_dir_y[gid] = 0.0f;
    grad_log_aniso[gid] = 0.0f;
}

// Clear buffer kernel (generic)
__global__ void clearBufferKernel(
    float* __restrict__ buffer,
    uint32_t count
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    buffer[gid] = 0.0f;
}

// C++ wrappers
extern "C" {

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
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (siteCount + threads - 1) / threads;
    
    adamUpdateKernel<<<blocks, threads, 0, stream>>>(
        sites, adam,
        grad_pos_x, grad_pos_y, grad_log_tau, grad_radius,
        grad_color_r, grad_color_g, grad_color_b,
        grad_dir_x, grad_dir_y, grad_log_aniso,
        lr_pos, lr_tau, lr_radius, lr_color, lr_dir, lr_aniso,
        beta1, beta2, eps, t, siteCount, width, height
    );
}

void launchClearGradients(
    float* grad_pos_x, float* grad_pos_y,
    float* grad_log_tau, float* grad_radius,
    float* grad_color_r, float* grad_color_g, float* grad_color_b,
    float* grad_dir_x, float* grad_dir_y, float* grad_log_aniso,
    uint32_t count,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    
    clearGradientsKernel<<<blocks, threads, 0, stream>>>(
        grad_pos_x, grad_pos_y, grad_log_tau, grad_radius,
        grad_color_r, grad_color_g, grad_color_b,
        grad_dir_x, grad_dir_y, grad_log_aniso,
        count
    );
}

void launchClearBuffer(
    float* buffer,
    uint32_t count,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    
    clearBufferKernel<<<blocks, threads, 0, stream>>>(buffer, count);
}

} // extern "C"
