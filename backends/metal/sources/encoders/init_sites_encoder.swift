import Metal

/// Encoder for GPU gradient-weighted site initialization
final class InitSitesEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "initGradientWeighted") else {
            throw NSError(domain: "InitSitesEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing initGradientWeighted kernel"])
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Initialize sites using gradient-weighted probing
    func encode(sitesBuffer: MTLBuffer, numSites: UInt32,
                seedCounterBuffer: MTLBuffer,
                targetTexture: MTLTexture, maskTexture: MTLTexture,
                gradThreshold: Float, maxAttempts: UInt32,
                initLogTau: Float, initRadius: Float,
                in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Init Gradient Weighted"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var numSitesVar = numSites
        var gradThreshVar = gradThreshold
        var maxAttemptsVar = maxAttempts

        encoder.setBytes(&numSitesVar, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBuffer(seedCounterBuffer, offset: 0, index: 2)
        encoder.setTexture(targetTexture, index: 0)
        encoder.setTexture(maskTexture, index: 1)
        encoder.setBytes(&gradThreshVar, length: MemoryLayout<Float>.stride, index: 3)
        encoder.setBytes(&maxAttemptsVar, length: MemoryLayout<UInt32>.stride, index: 4)
        var initLogTauVar = initLogTau
        var initRadiusVar = initRadius
        encoder.setBytes(&initLogTauVar, length: MemoryLayout<Float>.stride, index: 5)
        encoder.setBytes(&initRadiusVar, length: MemoryLayout<Float>.stride, index: 6)

        let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (Int(numSites) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
