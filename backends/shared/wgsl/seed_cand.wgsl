@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var outCand0 : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.siteCount) { return; }

  let site = sites[gid.x];
  if (site.position.x < 0.0) { return; }

  let outW = max(params.width, 1u);
  let outH = max(params.height, 1u);
  let candW = max(params.candWidth, 1u);
  let candH = max(params.candHeight, 1u);
  let fx = clamp(site.position.x, 0.0, f32(outW - 1u));
  let fy = clamp(site.position.y, 0.0, f32(outH - 1u));
  let candX = min(u32(fx * f32(candW) / f32(outW)), candW - 1u);
  let candY = min(u32(fy * f32(candH) / f32(outH)), candH - 1u);
  let homePixel = vec2<i32>(i32(candX), i32(candY));

  let existing = textureLoad(inCand0, homePixel, 0);
  textureStore(outCand0, homePixel, vec4<u32>(gid.x, existing.y, existing.z, existing.w));
}
