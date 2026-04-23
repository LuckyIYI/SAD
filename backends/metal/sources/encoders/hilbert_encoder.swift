import Metal

final class HilbertEncoder {
    private let hilbertPairsPipeline: MTLComputePipelineState
    private let hilbertWritePipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let hilbertPairsFunc = library.makeFunction(name: "buildHilbertPairs"),
              let hilbertWriteFunc = library.makeFunction(name: "writeHilbertOrder") else {
            throw NSError(domain: "HilbertEncoder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Missing Hilbert kernels"])
        }
        self.hilbertPairsPipeline = try device.makeComputePipelineState(function: hilbertPairsFunc)
        self.hilbertWritePipeline = try device.makeComputePipelineState(function: hilbertWriteFunc)
    }

    func encodeHilbertPairs(sitesBuffer: MTLBuffer,
                            pairsBuffer: MTLBuffer,
                            siteCount: UInt32,
                            paddedCount: UInt32,
                            width: UInt32,
                            height: UInt32,
                            bits: UInt32,
                            in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Build Hilbert Pairs"
        encoder.setComputePipelineState(hilbertPairsPipeline)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(pairsBuffer, offset: 0, index: 1)

        var siteCountVar = siteCount
        var paddedCountVar = paddedCount
        var widthVar = width
        var heightVar = height
        var bitsVar = bits

        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&paddedCountVar, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&widthVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&heightVar, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&bitsVar, length: MemoryLayout<UInt32>.stride, index: 6)

        let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (Int(paddedCount) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }

    func encodeHilbertWrite(pairsBuffer: MTLBuffer,
                            orderBuffer: MTLBuffer,
                            posBuffer: MTLBuffer,
                            siteCount: UInt32,
                            in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Write Hilbert Order"
        encoder.setComputePipelineState(hilbertWritePipeline)
        encoder.setBuffer(pairsBuffer, offset: 0, index: 0)
        encoder.setBuffer(orderBuffer, offset: 0, index: 1)
        encoder.setBuffer(posBuffer, offset: 0, index: 2)

        var siteCountVar = siteCount
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (Int(siteCount) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
