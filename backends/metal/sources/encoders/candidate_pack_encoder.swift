import Metal

final class CandidatePackEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "packCandidateSites") else {
            throw NSError(domain: "CandidatePackEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing packCandidateSites kernel"])
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(sitesBuffer: MTLBuffer, packedBuffer: MTLBuffer,
                siteCount: UInt32, in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Pack Candidate Sites"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(packedBuffer, offset: 0, index: 1)

        var siteCountVar = siteCount
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)

        let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (Int(siteCount) + 255) / 256,
                                   height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
