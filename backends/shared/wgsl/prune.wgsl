@group(0) @binding(0) var<storage, read_write> sites : array<Site>;
@group(0) @binding(1) var<storage, read> indices : array<u32>;
@group(0) @binding(2) var<uniform> params : PruneParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.count) { return; }
  let idx = indices[gid.x];
  if (idx == 0xffffffffu) { return; }
  var site = sites[idx];
  site.position = vec2<f32>(-1.0, -1.0);
  sites[idx] = site;
}
