import Metal

/// Encoder for tiled gradient computation using a threadgroup hash reduction.
final class TiledGradientEncoder {
    private let gradPipeline: MTLComputePipelineState
    private let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
    private let tileHashSize = 256  // Must match kTileHashSize in sad_common.metal

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let gradFunc = library.makeFunction(name: "computeGradientsTiled") else {
            throw NSError(domain: "TiledGradientEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing tiled gradient kernel"])
        }
        self.gradPipeline = try device.makeComputePipelineState(function: gradFunc)
    }

    private func encodeGradients(encoder: MTLComputeCommandEncoder,
                                 cand0: MTLTexture, cand1: MTLTexture,
                                 targetTexture: MTLTexture, maskTexture: MTLTexture,
                                 renderTexture: MTLTexture,
                                 ssimWeight: Float,
                                 gradBuffers: [MTLBuffer],
                                 sitesBuffer: MTLBuffer, invScaleSq: Float, siteCount: UInt32,
                                 removalDeltaBuffer: MTLBuffer, computeRemoval: Bool) {
        guard gradBuffers.count == 10 else {
            fatalError("TiledGradientEncoder requires exactly 10 gradient buffers")
        }

        encoder.setComputePipelineState(gradPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(targetTexture, index: 2)
        encoder.setTexture(renderTexture, index: 3)
        encoder.setTexture(maskTexture, index: 4)

        for i in 0..<10 {
            encoder.setBuffer(gradBuffers[i], offset: 0, index: i)
        }
        encoder.setBuffer(sitesBuffer, offset: 0, index: 10)

        var invScaleSqVar = invScaleSq
        var siteCountVar = siteCount
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 11)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBuffer(removalDeltaBuffer, offset: 0, index: 13)
        var computeRemovalVar: UInt32 = computeRemoval ? 1 : 0
        encoder.setBytes(&computeRemovalVar, length: MemoryLayout<UInt32>.stride, index: 14)
        var ssimWeightVar = ssimWeight
        encoder.setBytes(&ssimWeightVar, length: MemoryLayout<Float>.stride, index: 15)

        let keyMemSize = tileHashSize * MemoryLayout<UInt32>.stride
        let gradMemSize = tileHashSize * MemoryLayout<Int32>.stride
        encoder.setThreadgroupMemoryLength(keyMemSize, index: 0)
        for i in 0..<10 {
            encoder.setThreadgroupMemoryLength(gradMemSize, index: 1 + i)
        }
        encoder.setThreadgroupMemoryLength(gradMemSize, index: 11)

        let width = targetTexture.width
        let height = targetTexture.height
        let threadGroups = MTLSize(
            width: (width + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (height + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadgroupSize)
    }

    /// Compute gradients using per-tile reduction (threadgroup hash).
    func encodeGradients(cand0: MTLTexture, cand1: MTLTexture, targetTexture: MTLTexture,
                         maskTexture: MTLTexture, renderTexture: MTLTexture, ssimWeight: Float,
                         gradBuffers: [MTLBuffer],  // 10 buffers
                         sitesBuffer: MTLBuffer, invScaleSq: Float, siteCount: UInt32,
                         removalDeltaBuffer: MTLBuffer, computeRemoval: Bool,
                         in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Compute Gradients (Tiled)"
        encodeGradients(encoder: encoder,
                        cand0: cand0, cand1: cand1,
                        targetTexture: targetTexture, maskTexture: maskTexture, renderTexture: renderTexture,
                        ssimWeight: ssimWeight,
                        gradBuffers: gradBuffers,
                        sitesBuffer: sitesBuffer, invScaleSq: invScaleSq, siteCount: siteCount,
                        removalDeltaBuffer: removalDeltaBuffer, computeRemoval: computeRemoval)
        encoder.endEncoding()
    }

    func encodeGradients(cand0: MTLTexture, cand1: MTLTexture, targetTexture: MTLTexture,
                         maskTexture: MTLTexture, renderTexture: MTLTexture, ssimWeight: Float,
                         gradBuffers: [MTLBuffer],
                         sitesBuffer: MTLBuffer, invScaleSq: Float, siteCount: UInt32,
                         removalDeltaBuffer: MTLBuffer, computeRemoval: Bool,
                         in encoder: MTLComputeCommandEncoder) {
        encodeGradients(encoder: encoder,
                        cand0: cand0, cand1: cand1,
                        targetTexture: targetTexture, maskTexture: maskTexture, renderTexture: renderTexture,
                        ssimWeight: ssimWeight,
                        gradBuffers: gradBuffers,
                        sitesBuffer: sitesBuffer, invScaleSq: invScaleSq, siteCount: siteCount,
                        removalDeltaBuffer: removalDeltaBuffer, computeRemoval: computeRemoval)
    }
}
