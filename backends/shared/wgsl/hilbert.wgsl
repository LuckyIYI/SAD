struct HilbertParams {
  siteCount : u32,
  paddedCount : u32,
  width : u32,
  height : u32,
  bits : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

fn hilbertIndex(x : u32, y : u32, bits : u32) -> u32 {
  var xi = x;
  var yi = y;
  var index = 0u;
  let mask = select(0xffffffffu, (1u << bits) - 1u, bits < 32u);
  var i = i32(bits) - 1;
  loop {
    if (i < 0) { break; }
    let shift = u32(i);
    let rx = (xi >> shift) & 1u;
    let ry = (yi >> shift) & 1u;
    let d = (3u * rx) ^ ry;
    index = index | (d << (2u * shift));
    if (ry == 0u) {
      if (rx == 1u) {
        xi = mask - xi;
        yi = mask - yi;
      }
      let tmp = xi;
      xi = yi;
      yi = tmp;
    }
    i = i - 1;
  }
  return index;
}

@group(0) @binding(0) var<storage, read> hilbertSites : array<Site>;
@group(0) @binding(1) var<storage, read_write> hilbertPairs : array<vec2<u32>>;
@group(0) @binding(2) var<uniform> hilbertParams : HilbertParams;

@compute @workgroup_size(256)
fn buildHilbertPairs(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= hilbertParams.paddedCount) { return; }
  if (idx < hilbertParams.siteCount) {
    let site = hilbertSites[idx];
    let px = clamp(i32(site.position.x), 0, i32(hilbertParams.width - 1u));
    let py = clamp(i32(site.position.y), 0, i32(hilbertParams.height - 1u));
    let key = hilbertIndex(u32(px), u32(py), hilbertParams.bits);
    hilbertPairs[idx] = vec2<u32>(key, idx);
  } else {
    hilbertPairs[idx] = vec2<u32>(0xffffffffu, 0u);
  }
}

@group(0) @binding(0) var<storage, read> hilbertPairsIn : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> hilbertOrder : array<u32>;
@group(0) @binding(2) var<storage, read_write> hilbertPos : array<u32>;
@group(0) @binding(3) var<uniform> hilbertParams2 : HilbertParams;

@compute @workgroup_size(256)
fn writeHilbertOrder(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= hilbertParams2.siteCount) { return; }
  let siteIdx = hilbertPairsIn[idx].y;
  hilbertOrder[idx] = siteIdx;
  hilbertPos[siteIdx] = idx;
}
