@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var outCand0 : texture_storage_2d<rgba32uint, write>;
@group(0) @binding(2) var outCand1 : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.candWidth || gid.y >= params.candHeight) { return; }

  var state = params.seed ^ (gid.x * 1973u) ^ (gid.y * 9277u);
  var idx : array<u32, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    state = xorshift32(state + i);
    idx[i] = state % params.siteCount;
  }

  textureStore(outCand0, vec2<i32>(gid.xy), vec4<u32>(idx[0], idx[1], idx[2], idx[3]));
  textureStore(outCand1, vec2<i32>(gid.xy), vec4<u32>(idx[4], idx[5], idx[6], idx[7]));
}
