@group(0) @binding(0) var candidates0 : texture_2d<u32>;
@group(0) @binding(1) var candidates1 : texture_2d<u32>;
@group(0) @binding(2) var<storage, read> sites : array<Site>;
@group(0) @binding(3) var<storage, read> grad_raw : array<f32>;
@group(0) @binding(4) var<storage, read> grad_in : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_out : array<f32>;
@group(0) @binding(6) var<uniform> params : TauDiffuseParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.siteCount) { return; }
  if (params.lambda <= 0.0) {
    grad_out[idx] = grad_in[idx];
    return;
  }

  let site = sites[idx];
  if (site.position.x < 0.0) {
    grad_out[idx] = grad_in[idx];
    return;
  }

  let dims = textureDimensions(candidates0);
  let width = i32(dims.x);
  let height = i32(dims.y);
  let invDownscale = 1.0 / max(f32(params.candDownscale), 1.0);
  var p = vec2<i32>(site.position * invDownscale + vec2<f32>(0.5, 0.5));
  p = clamp(p, vec2<i32>(0, 0), vec2<i32>(width - 1, height - 1));
  let c0 = textureLoad(candidates0, p, 0);
  let c1 = textureLoad(candidates1, p, 0);
  let candIds = array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);

  var sum = 0.0;
  var count = 0u;
  for (var i = 0u; i < 8u; i = i + 1u) {
    let n = candIds[i];
    if (n == 0xffffffffu || n >= params.siteCount || n == idx) { continue; }

    var seen = false;
    for (var j = 0u; j < i; j = j + 1u) {
      if (candIds[j] == n) { seen = true; break; }
    }
    if (seen) { continue; }

    if (sites[n].position.x < 0.0) { continue; }
    sum = sum + grad_in[n];
    count = count + 1u;
  }

  let diag = 1.0 + params.lambda * f32(count);
  grad_out[idx] = (grad_raw[idx] + params.lambda * sum) / diag;
}
