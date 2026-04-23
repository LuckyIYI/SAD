import Metal

/// Encoder for densification (site splitting and statistics)
final class DensifyEncoder {
    private let splitPipeline: MTLComputePipelineState
    private let statsPipeline: MTLComputePipelineState
    private let scorePairsPipeline: MTLComputePipelineState
    private let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
    private let tileHashSize = 256  // Must match kTileHashSize in sad_common.metal

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let splitFunc = library.makeFunction(name: "splitSites"),
              let statsFunc = library.makeFunction(name: "computeSiteStatsTiled"),
              let scorePairsFunc = library.makeFunction(name: "computeDensifyScorePairs") else {
            throw NSError(domain: "DensifyEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing densify kernels"])
        }

        self.splitPipeline = try device.makeComputePipelineState(function: splitFunc)
        self.statsPipeline = try device.makeComputePipelineState(function: statsFunc)
        self.scorePairsPipeline = try device.makeComputePipelineState(function: scorePairsFunc)
    }

    /// Compute site statistics using tiled hash reduction.
    func encodeStats(cand0: MTLTexture, cand1: MTLTexture,
                     targetTexture: MTLTexture, maskTexture: MTLTexture,
                     sitesBuffer: MTLBuffer, invScaleSq: Float, siteCount: UInt32,
                     massBuffer: MTLBuffer, energyBuffer: MTLBuffer,
                     errWBuffer: MTLBuffer, errWxBuffer: MTLBuffer, errWyBuffer: MTLBuffer,
                     errWxxBuffer: MTLBuffer, errWxyBuffer: MTLBuffer, errWyyBuffer: MTLBuffer,
                     in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Compute Site Stats (Tiled)"
        encoder.setComputePipelineState(statsPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(targetTexture, index: 2)
        encoder.setTexture(maskTexture, index: 3)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var invScaleSqVar = invScaleSq
        var siteCountVar = siteCount
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)

        encoder.setBuffer(massBuffer, offset: 0, index: 3)
        encoder.setBuffer(energyBuffer, offset: 0, index: 4)
        encoder.setBuffer(errWBuffer, offset: 0, index: 5)
        encoder.setBuffer(errWxBuffer, offset: 0, index: 6)
        encoder.setBuffer(errWyBuffer, offset: 0, index: 7)
        encoder.setBuffer(errWxxBuffer, offset: 0, index: 8)
        encoder.setBuffer(errWxyBuffer, offset: 0, index: 9)
        encoder.setBuffer(errWyyBuffer, offset: 0, index: 10)

        let keyMemSize = tileHashSize * MemoryLayout<UInt32>.stride
        let statMemSize = tileHashSize * MemoryLayout<Int32>.stride
        encoder.setThreadgroupMemoryLength(keyMemSize, index: 0)
        for i in 0..<8 {
            encoder.setThreadgroupMemoryLength(statMemSize, index: 1 + i)
        }

        let width = targetTexture.width
        let height = targetTexture.height
        let threadGroups = MTLSize(
            width: (width + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (height + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// Compute densify score pairs (for sorting)
    func encodeScorePairs(sitesBuffer: MTLBuffer, massBuffer: MTLBuffer, energyBuffer: MTLBuffer,
                          pairsBuffer: MTLBuffer, siteCount: UInt32,
                          minMass: Float, scoreAlpha: Float, pairCount: UInt32,
                          in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Compute Densify Score Pairs"
        encoder.setComputePipelineState(scorePairsPipeline)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(massBuffer, offset: 0, index: 1)
        encoder.setBuffer(energyBuffer, offset: 0, index: 2)
        encoder.setBuffer(pairsBuffer, offset: 0, index: 3)

        var siteCountVar = siteCount
        var minMassVar = minMass
        var scoreAlphaVar = scoreAlpha
        var pairCountVar = pairCount

        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&minMassVar, length: MemoryLayout<Float>.stride, index: 5)
        encoder.setBytes(&scoreAlphaVar, length: MemoryLayout<Float>.stride, index: 6)
        encoder.setBytes(&pairCountVar, length: MemoryLayout<UInt32>.stride, index: 7)

        let tgs = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (Int(pairCount) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }

    /// Split sites (densification)
    func encodeSplit(sitesBuffer: MTLBuffer, adamBuffer: MTLBuffer,
                     splitIndicesBuffer: MTLBuffer, numToSplit: UInt32,
                     massBuffer: MTLBuffer,
                     errWBuffer: MTLBuffer, errWxBuffer: MTLBuffer, errWyBuffer: MTLBuffer,
                     errWxxBuffer: MTLBuffer, errWxyBuffer: MTLBuffer, errWyyBuffer: MTLBuffer,
                     currentSiteCount: UInt32,
                     targetTexture: MTLTexture,
                     in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Split Sites"
        encoder.setComputePipelineState(splitPipeline)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(adamBuffer, offset: 0, index: 1)
        encoder.setBuffer(splitIndicesBuffer, offset: 0, index: 2)

        var numToSplitVar = numToSplit
        encoder.setBytes(&numToSplitVar, length: MemoryLayout<UInt32>.stride, index: 3)

        encoder.setBuffer(massBuffer, offset: 0, index: 4)
        encoder.setBuffer(errWBuffer, offset: 0, index: 5)
        encoder.setBuffer(errWxBuffer, offset: 0, index: 6)
        encoder.setBuffer(errWyBuffer, offset: 0, index: 7)
        encoder.setBuffer(errWxxBuffer, offset: 0, index: 8)
        encoder.setBuffer(errWxyBuffer, offset: 0, index: 9)
        encoder.setBuffer(errWyyBuffer, offset: 0, index: 10)

        var currentCountVar = currentSiteCount
        encoder.setBytes(&currentCountVar, length: MemoryLayout<UInt32>.stride, index: 11)

        encoder.setTexture(targetTexture, index: 0)

        let tgs = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (Int(numToSplit) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }
}
