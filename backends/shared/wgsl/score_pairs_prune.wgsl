struct PruneScoreParams {
  siteCount : u32,
  pairCount : u32,
  _pad0 : u32,
  _pad1 : u32,
  deltaNorm : f32,
  _pad2 : f32,
  _pad3 : f32,
  _pad4 : f32,
}

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> removal_delta : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> pairs : array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params : PruneScoreParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.pairCount) { return; }

  var outPair = vec2<u32>(0xffffffffu, 0xffffffffu);
  if (gid.x < params.siteCount) {
    let site = sites[gid.x];
    let delta = bitcast<f32>(atomicLoad(&removal_delta[gid.x])) * params.deltaNorm;
    if (site.position.x >= 0.0 && !is_bad(delta)) {
      let key = bitcast<u32>(max(delta, 0.0));
      outPair = vec2<u32>(key, gid.x);
    }
  }
  pairs[gid.x] = outPair;
}
