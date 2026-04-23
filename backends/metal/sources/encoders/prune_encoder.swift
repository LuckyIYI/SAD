import Metal

/// Encoder for pruning sites
final class PruneEncoder {
    private let scorePairsPipeline: MTLComputePipelineState
    private let writeSplitIndicesPipeline: MTLComputePipelineState
    private let pruneSitesPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let scorePairsFunc = library.makeFunction(name: "computePruneScorePairs"),
              let writeSplitFunc = library.makeFunction(name: "writeSplitIndicesFromSorted"),
              let pruneSitesFunc = library.makeFunction(name: "pruneSitesByIndex") else {
            throw NSError(domain: "PruneEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing prune kernels"])
        }

        self.scorePairsPipeline = try device.makeComputePipelineState(function: scorePairsFunc)
        self.writeSplitIndicesPipeline = try device.makeComputePipelineState(function: writeSplitFunc)
        self.pruneSitesPipeline = try device.makeComputePipelineState(function: pruneSitesFunc)
    }

    /// Compute prune score pairs (for sorting)
    func encodeScorePairs(sitesBuffer: MTLBuffer, removalDeltaBuffer: MTLBuffer,
                          pairsBuffer: MTLBuffer,
                          siteCount: UInt32, deltaNorm: Float, pairCount: UInt32,
                          in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Compute Prune Score Pairs"
        encoder.setComputePipelineState(scorePairsPipeline)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(removalDeltaBuffer, offset: 0, index: 1)
        encoder.setBuffer(pairsBuffer, offset: 0, index: 2)

        var siteCountVar = siteCount
        var deltaNormVar = deltaNorm
        var pairCountVar = pairCount

        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&deltaNormVar, length: MemoryLayout<Float>.stride, index: 4)
        encoder.setBytes(&pairCountVar, length: MemoryLayout<UInt32>.stride, index: 5)

        let tgs = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (Int(pairCount) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }

    /// Write split indices from sorted pairs
    func encodeWriteSplitIndices(sortedPairsBuffer: MTLBuffer, splitIndicesBuffer: MTLBuffer,
                                 numToSplit: UInt32, in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Write Split Indices"
        encoder.setComputePipelineState(writeSplitIndicesPipeline)
        encoder.setBuffer(sortedPairsBuffer, offset: 0, index: 0)
        encoder.setBuffer(splitIndicesBuffer, offset: 0, index: 1)

        var numVar = numToSplit
        encoder.setBytes(&numVar, length: MemoryLayout<UInt32>.stride, index: 2)

        let tgs = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (Int(numToSplit) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }

    /// Prune sites by index (mark position as invalid)
    func encodePruneSites(sitesBuffer: MTLBuffer, indicesBuffer: MTLBuffer, count: UInt32,
                          in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Prune Sites"
        encoder.setComputePipelineState(pruneSitesPipeline)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(indicesBuffer, offset: 0, index: 1)

        var countVar = count
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.stride, index: 2)

        let tgs = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (Int(count) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }
}
