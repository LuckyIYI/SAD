@group(0) @binding(0) var<storage, read> sortedPairs : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> indices : array<u32>;
@group(0) @binding(2) var<uniform> params : PruneParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.count) { return; }
  indices[gid.x] = sortedPairs[gid.x].y;
}
