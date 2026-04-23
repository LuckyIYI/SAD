#include "../sad_common.metal"

kernel void adamUpdate(
    device VoronoiSite *sites [[buffer(0)]],
    device AdamState *adam [[buffer(1)]],
    device float *grad_pos_x [[buffer(2)]],
    device float *grad_pos_y [[buffer(3)]],
    device float *grad_log_tau [[buffer(4)]],
    device float *grad_radius [[buffer(5)]],
    device float *grad_color_r [[buffer(6)]],
    device float *grad_color_g [[buffer(7)]],
    device float *grad_color_b [[buffer(8)]],
    device float *grad_dir_x [[buffer(9)]],
    device float *grad_dir_y [[buffer(10)]],
    device float *grad_log_aniso [[buffer(11)]],
    constant float &lr_pos [[buffer(12)]],
    constant float &lr_tau [[buffer(13)]],
    constant float &lr_radius [[buffer(14)]],
    constant float &lr_color [[buffer(15)]],
    constant float &lr_dir [[buffer(16)]],
    constant float &lr_aniso [[buffer(17)]],
    constant float &beta1 [[buffer(18)]],
    constant float &beta2 [[buffer(19)]],
    constant float &eps [[buffer(20)]],
    constant uint &t [[buffer(21)]],
    constant uint &width [[buffer(22)]],
    constant uint &height [[buffer(23)]],
    uint gid [[thread_position_in_grid]])
{
    // Normalize gradients by image size
    float norm = 1.0f / float(width * height);

    float2 g_pos = float2(grad_pos_x[gid], grad_pos_y[gid]) * norm;
    float g_log_tau = grad_log_tau[gid] * norm;
    float g_radius = grad_radius[gid] * norm;
    float3 g_color = float3(grad_color_r[gid], grad_color_g[gid], grad_color_b[gid]) * norm;
    float2 g_dir = float2(grad_dir_x[gid], grad_dir_y[gid]) * norm;
    float g_log_aniso = grad_log_aniso[gid] * norm;

    // Skip inactive sites (marked with negative position)
    if (sites[gid].position.x < 0.0f) return;

    // Adam update
    float tt = float(t);
    float b1t_corr = 1.0f / (1.0f - pow(beta1, tt));
    float b2t_corr = 1.0f / (1.0f - pow(beta2, tt));

    // Position (IsotropicAdam)
    adam[gid].m_pos = beta1 * adam[gid].m_pos + (1.0f - beta1) * g_pos;
    float g2_pos = dot(g_pos, g_pos);
    adam[gid].v_pos = beta2 * adam[gid].v_pos + (1.0f - beta2) * g2_pos;
    float2 m_hat_pos = adam[gid].m_pos * b1t_corr;
    float v_hat_pos = adam[gid].v_pos * b2t_corr;
    float inv_denom_pos = 1.0f / (sqrt(v_hat_pos) + eps);
    float2 step_pos = lr_pos * m_hat_pos * inv_denom_pos;

    sites[gid].position -= step_pos;
    sites[gid].position = clamp(sites[gid].position, float2(0.0f), float2(width-1, height-1));

    // log_tau
    adam[gid].m_log_tau = beta1 * adam[gid].m_log_tau + (1.0f - beta1) * g_log_tau;
    adam[gid].v_log_tau = beta2 * adam[gid].v_log_tau + (1.0f - beta2) * (g_log_tau * g_log_tau);
    float m_hat_tau = adam[gid].m_log_tau * b1t_corr;
    float v_hat_tau = adam[gid].v_log_tau * b2t_corr;
    float step_tau = lr_tau * m_hat_tau / (sqrt(v_hat_tau) + eps);

    sites[gid].log_tau -= step_tau;

    // radius
    adam[gid].m_radius = beta1 * adam[gid].m_radius + (1.0f - beta1) * g_radius;
    adam[gid].v_radius = beta2 * adam[gid].v_radius + (1.0f - beta2) * (g_radius * g_radius);
    float m_hat_rad = adam[gid].m_radius * b1t_corr;
    float v_hat_rad = adam[gid].v_radius * b2t_corr;
    float step_rad = lr_radius * m_hat_rad / (sqrt(v_hat_rad) + eps);

    sites[gid].radius -= step_rad;

    // Color (per-channel Adam)
    adam[gid].m_color = beta1 * adam[gid].m_color + (1.0f - beta1) * g_color;
    adam[gid].v_color = beta2 * adam[gid].v_color + (1.0f - beta2) * (g_color * g_color);
    float3 m_hat_col = adam[gid].m_color * b1t_corr;
    float3 v_hat_col = adam[gid].v_color * b2t_corr;
    float3 step_col = lr_color * m_hat_col / (sqrt(v_hat_col) + eps);

    sites[gid].color -= step_col;

    // Anisotropy direction (tangent + IsotropicAdam)
    float2 dir = sites[gid].aniso_dir;
    float2 g_tan = g_dir - dir * dot(g_dir, dir);
    adam[gid].m_dir = beta1 * adam[gid].m_dir + (1.0f - beta1) * g_tan;
    float g2_dir = dot(g_tan, g_tan);
    adam[gid].v_dir = beta2 * adam[gid].v_dir + (1.0f - beta2) * g2_dir;
    float2 m_hat_dir = adam[gid].m_dir * b1t_corr;
    float v_hat_dir = adam[gid].v_dir * b2t_corr;
    float inv_denom_dir = 1.0f / (sqrt(v_hat_dir) + eps);
    float2 step_dir = lr_dir * m_hat_dir * inv_denom_dir;

    float2 new_dir = dir - step_dir;
    float len2 = dot(new_dir, new_dir);
    if (len2 < 1e-12f) {
        new_dir = float2(1.0f, 0.0f);
    } else {
        new_dir *= rsqrt(len2);
    }
    sites[gid].aniso_dir = new_dir;

    // log_aniso (det=1 via eigenvalues e^a and e^-a)
    adam[gid].m_log_aniso = beta1 * adam[gid].m_log_aniso + (1.0f - beta1) * g_log_aniso;
    adam[gid].v_log_aniso = beta2 * adam[gid].v_log_aniso + (1.0f - beta2) * (g_log_aniso * g_log_aniso);
    float m_hat_aniso = adam[gid].m_log_aniso * b1t_corr;
    float v_hat_aniso = adam[gid].v_log_aniso * b2t_corr;
    float step_aniso = lr_aniso * m_hat_aniso / (sqrt(v_hat_aniso) + eps);

    sites[gid].log_aniso -= step_aniso;
    sites[gid].log_aniso = clamp(sites[gid].log_aniso, -4.0f, 4.0f);

    // Clear gradients (regular assignment, no atomics)
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
