@group(0) @binding(0) var<storage, read> grad_in : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_out : array<atomic<i32>>;
@group(0) @binding(2) var<uniform> params : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  let val = grad_in[idx];
  if (is_bad(val)) {
    atomicStore(&grad_out[idx], 0);
  } else {
    atomicStore(&grad_out[idx], i32(val * K_GRAD_QUANT_SCALE));
  }
}
