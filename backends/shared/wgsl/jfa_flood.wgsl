@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var outCand0 : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.candWidth || gid.y >= params.candHeight) { return; }

  let uv = candidateUvFromCoord(
    vec2<u32>(gid.xy),
    vec2<u32>(params.width, params.height),
    vec2<u32>(params.candWidth, params.candHeight)
  );
  let stepSize = i32(params.step);

  var bestIdx : array<u32, 4>;
  let inf = 1e20;
  var bestD2 : array<f32, 4>;
  for (var i = 0u; i < 4u; i = i + 1u) {
    bestIdx[i] = 0xffffffffu;
    bestD2[i] = inf;
  }

  let w = i32(params.candWidth);
  let h = i32(params.candHeight);
  let gi = vec2<i32>(i32(gid.x), i32(gid.y));

  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let samplePos = clamp(gi + vec2<i32>(dx, dy) * stepSize, vec2<i32>(0), vec2<i32>(w - 1, h - 1));
      let cand = textureLoad(inCand0, samplePos, 0);

      if (cand.x < params.siteCount) {
        let site = sites[cand.x];
        insertClosest4(&bestIdx, &bestD2, cand.x, uv, site, params.invScaleSq);
      }
      if (cand.y < params.siteCount) {
        let site = sites[cand.y];
        insertClosest4(&bestIdx, &bestD2, cand.y, uv, site, params.invScaleSq);
      }
      if (cand.z < params.siteCount) {
        let site = sites[cand.z];
        insertClosest4(&bestIdx, &bestD2, cand.z, uv, site, params.invScaleSq);
      }
      if (cand.w < params.siteCount) {
        let site = sites[cand.w];
        insertClosest4(&bestIdx, &bestD2, cand.w, uv, site, params.invScaleSq);
      }
    }
  }

  textureStore(outCand0, vec2<i32>(gid.xy), vec4<u32>(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]));
}
