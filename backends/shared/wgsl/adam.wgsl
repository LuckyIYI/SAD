@group(0) @binding(0) var<storage, read_write> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> adam : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_pos_x : array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> grad_pos_y : array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> grad_log_tau : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> grad_radius_sq : array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> grad_color_r : array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> grad_color_g : array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> grad_color_b : array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> grad_dir_x : array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> grad_dir_y : array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> grad_log_aniso : array<atomic<i32>>;
@group(0) @binding(12) var<uniform> params : AdamParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let norm = 1.0 / f32(params.width * params.height);
  let scale = K_GRAD_QUANT_SCALE_INV * norm;
  let g_pos = vec2<f32>(f32(atomicLoad(&grad_pos_x[idx])),
                        f32(atomicLoad(&grad_pos_y[idx]))) * scale;
  let g_log_tau = f32(atomicLoad(&grad_log_tau[idx])) * scale;
  let g_radius_sq = f32(atomicLoad(&grad_radius_sq[idx])) * scale;
  let g_color = vec3<f32>(f32(atomicLoad(&grad_color_r[idx])),
                          f32(atomicLoad(&grad_color_g[idx])),
                          f32(atomicLoad(&grad_color_b[idx]))) * scale;
  let g_dir = vec2<f32>(f32(atomicLoad(&grad_dir_x[idx])),
                        f32(atomicLoad(&grad_dir_y[idx]))) * scale;
  var g_log_aniso = f32(atomicLoad(&grad_log_aniso[idx])) * scale;
  var safe_g_dir = g_dir;
  if (is_bad2(safe_g_dir)) { safe_g_dir = vec2<f32>(0.0, 0.0); }
  if (is_bad(g_log_aniso)) { g_log_aniso = 0.0; }

  if (sites[idx].position.x < 0.0) {
    return;
  }
  if (is_bad(sites[idx].position.x) || is_bad(sites[idx].position.y) || is_bad(sites[idx].log_tau) ||
      is_bad(sites[idx].radius_sq) || is_bad3(site_color(sites[idx])) || is_bad(sites[idx].log_aniso) ||
      is_bad(g_pos.x) || is_bad(g_pos.y) || is_bad(g_log_tau) || is_bad(g_radius_sq) ||
      is_bad3(g_color) || is_bad(safe_g_dir.x) || is_bad(safe_g_dir.y) || is_bad(g_log_aniso)) {
    atomicStore(&grad_pos_x[idx], 0);
    atomicStore(&grad_pos_y[idx], 0);
    atomicStore(&grad_log_tau[idx], 0);
    atomicStore(&grad_radius_sq[idx], 0);
    atomicStore(&grad_color_r[idx], 0);
    atomicStore(&grad_color_g[idx], 0);
    atomicStore(&grad_color_b[idx], 0);
    atomicStore(&grad_dir_x[idx], 0);
    atomicStore(&grad_dir_y[idx], 0);
    atomicStore(&grad_log_aniso[idx], 0);
    return;
  }

  let base = idx * 24u;
  var m_pos = vec2<f32>(adam[base + 0u], adam[base + 1u]);
  var v_pos = vec2<f32>(adam[base + 2u], adam[base + 3u]);
  var m_log_tau = adam[base + 4u];
  var v_log_tau = adam[base + 5u];
  var m_radius_sq = adam[base + 6u];
  var v_radius_sq = adam[base + 7u];
  var m_color = vec3<f32>(adam[base + 8u], adam[base + 9u], adam[base + 10u]);
  var v_color = vec3<f32>(adam[base + 11u], adam[base + 12u], adam[base + 13u]);
  var m_dir = vec2<f32>(adam[base + 14u], adam[base + 15u]);
  var v_dir = vec2<f32>(adam[base + 16u], adam[base + 17u]);
  var m_log_aniso = adam[base + 18u];
  var v_log_aniso = adam[base + 19u];

  let tt = f32(params.t);
  let b1t_corr = 1.0 / (1.0 - pow(params.beta1, tt));
  let b2t_corr = 1.0 / (1.0 - pow(params.beta2, tt));

  m_pos = params.beta1 * m_pos + (1.0 - params.beta1) * g_pos;
  let g2_pos = dot(g_pos, g_pos);
  let v_pos_scalar = params.beta2 * v_pos.x + (1.0 - params.beta2) * g2_pos;
  let m_hat_pos = m_pos * b1t_corr;
  let v_hat_pos = v_pos_scalar * b2t_corr;
  let step_pos = params.lr_pos * m_hat_pos / (sqrt(v_hat_pos) + params.eps);
  sites[idx].position = clamp(sites[idx].position - step_pos, vec2<f32>(0.0), vec2<f32>(f32(params.width - 1u), f32(params.height - 1u)));
  v_pos = vec2<f32>(v_pos_scalar, v_pos_scalar);

  m_log_tau = params.beta1 * m_log_tau + (1.0 - params.beta1) * g_log_tau;
  v_log_tau = params.beta2 * v_log_tau + (1.0 - params.beta2) * (g_log_tau * g_log_tau);
  let m_hat_tau = m_log_tau * b1t_corr;
  let v_hat_tau = v_log_tau * b2t_corr;
  let step_tau = params.lr_tau * m_hat_tau / (sqrt(v_hat_tau) + params.eps);
  sites[idx].log_tau = sites[idx].log_tau - step_tau;

  m_radius_sq = params.beta1 * m_radius_sq + (1.0 - params.beta1) * g_radius_sq;
  v_radius_sq = params.beta2 * v_radius_sq + (1.0 - params.beta2) * (g_radius_sq * g_radius_sq);
  let m_hat_rad = m_radius_sq * b1t_corr;
  let v_hat_rad = v_radius_sq * b2t_corr;
  let step_rad = params.lr_radius * m_hat_rad / (sqrt(v_hat_rad) + params.eps);
  sites[idx].radius_sq = sites[idx].radius_sq - step_rad;

  m_color = params.beta1 * m_color + (1.0 - params.beta1) * g_color;
  v_color = params.beta2 * v_color + (1.0 - params.beta2) * (g_color * g_color);
  let m_hat_col = m_color * b1t_corr;
  let v_hat_col = v_color * b2t_corr;
  let step_col = params.lr_color * m_hat_col / (sqrt(v_hat_col) + params.eps);
  let new_col = site_color(sites[idx]) - step_col;
  sites[idx].color_r = new_col.x;
  sites[idx].color_g = new_col.y;
  sites[idx].color_b = new_col.z;

  let curr_dir = safe_dir(site_aniso_dir(sites[idx]));
  let g_tan = safe_g_dir - curr_dir * dot(safe_g_dir, curr_dir);
  m_dir = params.beta1 * m_dir + (1.0 - params.beta1) * g_tan;
  let g2_dir = dot(g_tan, g_tan);
  let v_dir_scalar = params.beta2 * v_dir.x + (1.0 - params.beta2) * g2_dir;
  let m_hat_dir = m_dir * b1t_corr;
  let v_hat_dir = v_dir_scalar * b2t_corr;
  let step_dir = params.lr_dir * m_hat_dir / (sqrt(v_hat_dir) + params.eps);

  var new_dir = curr_dir - step_dir;
  new_dir = safe_dir(new_dir);
  sites[idx].aniso_dir_x = new_dir.x;
  sites[idx].aniso_dir_y = new_dir.y;
  v_dir = vec2<f32>(v_dir_scalar, v_dir_scalar);

  m_log_aniso = params.beta1 * m_log_aniso + (1.0 - params.beta1) * g_log_aniso;
  v_log_aniso = params.beta2 * v_log_aniso + (1.0 - params.beta2) * (g_log_aniso * g_log_aniso);
  let m_hat_aniso = m_log_aniso * b1t_corr;
  let v_hat_aniso = v_log_aniso * b2t_corr;
  let step_aniso = params.lr_aniso * m_hat_aniso / (sqrt(v_hat_aniso) + params.eps);
  sites[idx].log_aniso = sites[idx].log_aniso - step_aniso;

  // Clamp to prevent extreme anisotropy
  sites[idx].log_aniso = clamp(sites[idx].log_aniso, -2.0, 2.0);

  adam[base + 0u] = m_pos.x;
  adam[base + 1u] = m_pos.y;
  adam[base + 2u] = v_pos.x;
  adam[base + 3u] = v_pos.y;
  adam[base + 4u] = m_log_tau;
  adam[base + 5u] = v_log_tau;
  adam[base + 6u] = m_radius_sq;
  adam[base + 7u] = v_radius_sq;
  adam[base + 8u] = m_color.x;
  adam[base + 9u] = m_color.y;
  adam[base + 10u] = m_color.z;
  adam[base + 11u] = v_color.x;
  adam[base + 12u] = v_color.y;
  adam[base + 13u] = v_color.z;
  adam[base + 14u] = m_dir.x;
  adam[base + 15u] = m_dir.y;
  adam[base + 16u] = v_dir.x;
  adam[base + 17u] = v_dir.y;
  adam[base + 18u] = m_log_aniso;
  adam[base + 19u] = v_log_aniso;

  atomicStore(&grad_pos_x[idx], 0);
  atomicStore(&grad_pos_y[idx], 0);
  atomicStore(&grad_log_tau[idx], 0);
  atomicStore(&grad_radius_sq[idx], 0);
  atomicStore(&grad_color_r[idx], 0);
  atomicStore(&grad_color_g[idx], 0);
  atomicStore(&grad_color_b[idx], 0);
  atomicStore(&grad_dir_x[idx], 0);
  atomicStore(&grad_dir_y[idx], 0);
  atomicStore(&grad_log_aniso[idx], 0);
}
