struct DensifyScoreParams {
  siteCount : u32,
  pairCount : u32,
  _pad0 : u32,
  _pad1 : u32,
  minMass : f32,
  scoreAlpha : f32,
  _pad2 : f32,
  _pad3 : f32,
}

@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> mass : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> energy : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> pairs : array<vec2<u32>>;
@group(0) @binding(4) var<uniform> params : DensifyScoreParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.pairCount) { return; }

  var outPair = vec2<u32>(0xffffffffu, 0xffffffffu);
  if (gid.x < params.siteCount) {
    let site = sites[gid.x];
    let m = bitcast<f32>(atomicLoad(&mass[gid.x]));
    let e = bitcast<f32>(atomicLoad(&energy[gid.x]));
    if (site.position.x >= 0.0 && !is_bad(m) && !is_bad(e) && m > params.minMass) {
      let denom = pow(max(m, 1e-8), params.scoreAlpha);
      let score = max(e, 0.0) / denom;
      if (!is_bad(score)) {
        let key = 0xffffffffu - bitcast<u32>(score);
        outPair = vec2<u32>(key, gid.x);
      }
    }
  }
  pairs[gid.x] = outPair;
}
