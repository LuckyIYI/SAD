@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var outCand0 : texture_storage_2d<rgba32uint, write>;
@group(0) @binding(2) var outCand1 : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.candWidth || gid.y >= params.candHeight) { return; }
  let empty = vec4<u32>(0xffffffffu);
  textureStore(outCand0, vec2<i32>(gid.xy), empty);
  textureStore(outCand1, vec2<i32>(gid.xy), empty);
}
