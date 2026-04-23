@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> packed_sites : array<PackedCandidateSite>;
@group(0) @binding(2) var<uniform> packParams : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= packParams.count) { return; }

  let site = sites[idx];
  var out : PackedCandidateSite;
  if (site.position.x < 0.0) {
    out.data = vec4<u32>(
      pack2x16float(vec2<f32>(-1.0, -1.0)),
      pack2x16float(vec2<f32>(0.0, 0.0)),
      pack2x16float(vec2<f32>(0.0, 0.0)),
      pack2x16float(vec2<f32>(0.0, 0.0))
    );
  } else {
    out.data = vec4<u32>(
      pack2x16float(site.position),
      pack2x16float(vec2<f32>(site.log_tau, site.radius_sq)),
      pack2x16float(site_aniso_dir(site)),
      pack2x16float(vec2<f32>(site.log_aniso, 0.0))
    );
  }
  packed_sites[idx] = out;
}
