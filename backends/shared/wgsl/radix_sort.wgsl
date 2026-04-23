const BLOCK_SIZE : u32 = 256u;
const GRAIN : u32 = 4u;
const BUCKETS : u32 = 256u;
const ELEMENTS_PER_BLOCK : u32 = BLOCK_SIZE * GRAIN;

var<workgroup> radix_hist : array<atomic<u32>, 256>;
var<workgroup> radix_scan_scratch : array<u32, 256>;
var<workgroup> radix_scatter_bin_offset : array<u32, 256>;
var<workgroup> radix_scatter_bin_counts : array<atomic<u32>, 256>;
var<workgroup> radix_scatter_bins : array<u32, 256>;
var<workgroup> radix_scatter_ranks : array<u32, 256>;

struct RadixParams {
  paddedCount : u32,
  shift : u32,
  _pad0 : u32,
  _pad1 : u32,
}

fn bucketForKey(key : u32, shift : u32) -> u32 {
  return (key >> shift) & 0xffu;
}

fn exclusiveScan256(tid : u32) -> u32 {
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= BLOCK_SIZE) { break; }
    let ai = offset * (2u * tid + 1u) - 1u;
    let bi = offset * (2u * tid + 2u) - 1u;
    if (bi < BLOCK_SIZE) {
      radix_scan_scratch[bi] = radix_scan_scratch[bi] + radix_scan_scratch[ai];
    }
    workgroupBarrier();
    offset = offset << 1u;
  }

  let total = radix_scan_scratch[BLOCK_SIZE - 1u];
  if (tid == 0u) {
    radix_scan_scratch[BLOCK_SIZE - 1u] = 0u;
  }
  workgroupBarrier();

  offset = BLOCK_SIZE >> 1u;
  loop {
    if (offset == 0u) { break; }
    let ai = offset * (2u * tid + 1u) - 1u;
    let bi = offset * (2u * tid + 2u) - 1u;
    if (bi < BLOCK_SIZE) {
      let t = radix_scan_scratch[ai];
      radix_scan_scratch[ai] = radix_scan_scratch[bi];
      radix_scan_scratch[bi] = radix_scan_scratch[bi] + t;
    }
    workgroupBarrier();
    offset = offset >> 1u;
  }
  return total;
}

@group(0) @binding(0) var<storage, read> inputPairs : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> histFlat : array<u32>;
@group(0) @binding(2) var<uniform> params : RadixParams;

@compute @workgroup_size(256)
fn radixHistogramUInt2(@builtin(workgroup_id) groupId : vec3<u32>,
                       @builtin(local_invocation_id) localId : vec3<u32>) {
  let gridSize = (params.paddedCount + ELEMENTS_PER_BLOCK - 1u) / ELEMENTS_PER_BLOCK;
  if (groupId.x >= gridSize) { return; }

  if (localId.x < BUCKETS) {
    atomicStore(&radix_hist[localId.x], 0u);
  }
  workgroupBarrier();

  let base = groupId.x * ELEMENTS_PER_BLOCK;
  for (var i = 0u; i < GRAIN; i = i + 1u) {
    let idx = base + localId.x + i * BLOCK_SIZE;
    if (idx >= params.paddedCount) { break; }
    let key = inputPairs[idx].x;
    let bin = bucketForKey(key, params.shift);
    atomicAdd(&radix_hist[bin], 1u);
  }
  workgroupBarrier();

  if (localId.x < BUCKETS) {
    histFlat[localId.x * gridSize + groupId.x] = atomicLoad(&radix_hist[localId.x]);
  }
}

@group(0) @binding(0) var<storage, read_write> histScan : array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums : array<u32>;
@group(0) @binding(2) var<uniform> scanParams : RadixParams;

@compute @workgroup_size(256)
fn radixScanHistogramBlocks(@builtin(workgroup_id) groupId : vec3<u32>,
                            @builtin(local_invocation_id) localId : vec3<u32>) {
  let gridSize = (scanParams.paddedCount + ELEMENTS_PER_BLOCK - 1u) / ELEMENTS_PER_BLOCK;
  let histLength = gridSize * BUCKETS;
  let numBlocks = (histLength + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  if (groupId.x >= numBlocks) { return; }

  let baseIndex = groupId.x * BLOCK_SIZE;
  let idx = baseIndex + localId.x;
  var value = 0u;
  if (idx < histLength) {
    value = histScan[idx];
  }
  radix_scan_scratch[localId.x] = value;

  _ = exclusiveScan256(localId.x);
  let scanned = radix_scan_scratch[localId.x];

  if (idx < histLength) {
    histScan[idx] = scanned;
  }
  if (localId.x == BLOCK_SIZE - 1u) {
    blockSums[groupId.x] = scanned + value;
  }
}

@group(0) @binding(0) var<storage, read_write> blockScan : array<u32>;
@group(0) @binding(1) var<uniform> blockParams : RadixParams;

@compute @workgroup_size(256)
fn radixExclusiveScanBlockSums(@builtin(workgroup_id) groupId : vec3<u32>,
                               @builtin(local_invocation_id) localId : vec3<u32>) {
  if (groupId.x > 0u) { return; }

  let gridSize = (blockParams.paddedCount + ELEMENTS_PER_BLOCK - 1u) / ELEMENTS_PER_BLOCK;
  let histLength = gridSize * BUCKETS;
  let num = (histLength + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  let elemsPerThread = (num + BLOCK_SIZE - 1u) / BLOCK_SIZE;

  var localSum = 0u;
  let baseIdx = localId.x * elemsPerThread;
  for (var i = 0u; i < elemsPerThread; i = i + 1u) {
    let idx = baseIdx + i;
    if (idx < num) {
      localSum = localSum + blockScan[idx];
    }
  }
  radix_scan_scratch[localId.x] = localSum;

  _ = exclusiveScan256(localId.x);
  let threadBase = radix_scan_scratch[localId.x];

  var running = threadBase;
  for (var i = 0u; i < elemsPerThread; i = i + 1u) {
    let idx = baseIdx + i;
    if (idx < num) {
      let val = blockScan[idx];
      blockScan[idx] = running;
      running = running + val;
    }
  }
}

@group(0) @binding(0) var<storage, read_write> histOffsets : array<u32>;
@group(0) @binding(1) var<storage, read> blockOffsets : array<u32>;
@group(0) @binding(2) var<uniform> offsetParams : RadixParams;

@compute @workgroup_size(256)
fn radixApplyOffsets(@builtin(workgroup_id) groupId : vec3<u32>,
                     @builtin(local_invocation_id) localId : vec3<u32>) {
  let gridSize = (offsetParams.paddedCount + ELEMENTS_PER_BLOCK - 1u) / ELEMENTS_PER_BLOCK;
  let histLength = gridSize * BUCKETS;
  let numBlocks = (histLength + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  if (groupId.x >= numBlocks) { return; }

  let baseIndex = groupId.x * BLOCK_SIZE;
  let idx = baseIndex + localId.x;
  if (idx < histLength) {
    histOffsets[idx] = histOffsets[idx] + blockOffsets[groupId.x];
  }
}

@group(0) @binding(0) var<storage, read> scatterInput : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> scatterOutput : array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> offsetsFlat : array<u32>;
@group(0) @binding(3) var<uniform> scatterParams : RadixParams;

@compute @workgroup_size(256)
fn radixScatterUInt2(@builtin(workgroup_id) groupId : vec3<u32>,
                     @builtin(local_invocation_id) localId : vec3<u32>) {
  let gridSize = (scatterParams.paddedCount + ELEMENTS_PER_BLOCK - 1u) / ELEMENTS_PER_BLOCK;
  if (groupId.x >= gridSize) { return; }

  let blockBase = groupId.x * ELEMENTS_PER_BLOCK;

  if (localId.x < BUCKETS) {
    radix_scatter_bin_offset[localId.x] = offsetsFlat[localId.x * gridSize + groupId.x];
  }
  workgroupBarrier();

  for (var chunk = 0u; chunk < GRAIN; chunk = chunk + 1u) {
    let idx = blockBase + chunk * BLOCK_SIZE + localId.x;
    var val = vec2<u32>(0xffffffffu, 0xfffffffeu);
    if (idx < scatterParams.paddedCount) {
      val = scatterInput[idx];
    }
    let isValid = val.y != 0xfffffffeu;
    let bin = bucketForKey(val.x, scatterParams.shift);
    radix_scatter_bins[localId.x] = select(0xffffffffu, bin, isValid);

    if (localId.x < BUCKETS) {
      atomicStore(&radix_scatter_bin_counts[localId.x], 0u);
    }
    workgroupBarrier();

    if (isValid) {
      atomicAdd(&radix_scatter_bin_counts[bin], 1u);
    }
    workgroupBarrier();

    var rankInBin = 0u;
    if (isValid) {
      var i = 0u;
      loop {
        if (i >= localId.x) { break; }
        if (radix_scatter_bins[i] == bin) { rankInBin = rankInBin + 1u; }
        i = i + 1u;
      }
    }
    radix_scatter_ranks[localId.x] = rankInBin;
    workgroupBarrier();

    if (isValid) {
      let dst = radix_scatter_bin_offset[bin] + rankInBin;
      if (dst < scatterParams.paddedCount) {
        scatterOutput[dst] = val;
      }
    }
    workgroupBarrier();

    if (localId.x < BUCKETS) {
      radix_scatter_bin_offset[localId.x] = radix_scatter_bin_offset[localId.x] +
        atomicLoad(&radix_scatter_bin_counts[localId.x]);
    }
    workgroupBarrier();
  }
}
