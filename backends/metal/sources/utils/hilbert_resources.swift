import Metal

struct HilbertResources {
    var order: MTLBuffer
    var pos: MTLBuffer
    var pairs: MTLBuffer
    var sort: RadixSortUInt2
    var paddedCount: Int
    var siteCount: Int
    var ready: Bool
}

func makeHilbertResources(device: MTLDevice,
                          library: MTLLibrary,
                          siteCapacity: Int) -> HilbertResources {
    let count = max(1, siteCapacity)
    let order = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride,
                                  options: .storageModeShared)!
    let pos = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride,
                                options: .storageModeShared)!
    let paddedCount = ((count + 1023) / 1024) * 1024
    let pairs = device.makeBuffer(length: paddedCount * MemoryLayout<SIMD2<UInt32>>.stride,
                                  options: .storageModeShared)!
    guard let sort = try? RadixSortUInt2(device: device, library: library, paddedCount: paddedCount) else {
        fatalError("Hilbert sort kernels are required; radix sort pipeline is unavailable.")
    }

    return HilbertResources(order: order,
                            pos: pos,
                            pairs: pairs,
                            sort: sort,
                            paddedCount: paddedCount,
                            siteCount: 0,
                            ready: false)
}

func updateHilbertResources(resources: inout HilbertResources,
                            encoder: HilbertEncoder,
                            commandBuffer: MTLCommandBuffer,
                            sitesBuffer: MTLBuffer,
                            siteCount: Int,
                            width: Int,
                            height: Int) {
    guard siteCount > 0 else { return }

    let bits = UInt32(hilbertBitsForSize(width: width, height: height))
    let paddedCount = UInt32(resources.paddedCount)
    let siteCountU = UInt32(siteCount)
    let widthU = UInt32(width)
    let heightU = UInt32(height)

    encoder.encodeHilbertPairs(sitesBuffer: sitesBuffer,
                               pairsBuffer: resources.pairs,
                               siteCount: siteCountU,
                               paddedCount: paddedCount,
                               width: widthU,
                               height: heightU,
                               bits: bits,
                               in: commandBuffer)

    let maxKeyExclusive: UInt32 = bits >= 16 ? UInt32.max : (1 << (bits * 2))
    resources.sort.encode(data: resources.pairs,
                          maxKeyExclusive: maxKeyExclusive,
                          in: commandBuffer)

    encoder.encodeHilbertWrite(pairsBuffer: resources.pairs,
                               orderBuffer: resources.order,
                               posBuffer: resources.pos,
                               siteCount: siteCountU,
                               in: commandBuffer)

    resources.siteCount = siteCount
    resources.ready = true
}
