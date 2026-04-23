@group(0) @binding(0) var<storage, read_write> grad_in : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> grad_out : array<f32>;
@group(0) @binding(2) var<uniform> params : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  grad_out[idx] = f32(atomicLoad(&grad_in[idx])) * K_GRAD_QUANT_SCALE_INV;
}
