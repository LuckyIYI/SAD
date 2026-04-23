@group(0) @binding(0) var<storage, read_write> buffer : array<atomic<i32>>;
@group(0) @binding(1) var<uniform> params : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.count) { return; }
  atomicStore(&buffer[gid.x], 0);
}
