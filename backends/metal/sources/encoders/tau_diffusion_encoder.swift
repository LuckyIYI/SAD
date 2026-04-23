import Metal

final class TauDiffusionEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let funcObj = library.makeFunction(name: "diffuseTauGradientsAtSite") else {
            throw NSError(domain: "TauDiffusionEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing tau diffusion kernel"])
        }
        self.pipeline = try device.makeComputePipelineState(function: funcObj)
    }

    func encode(cand0: MTLTexture, cand1: MTLTexture,
                sitesBuffer: MTLBuffer,
                gradRaw: MTLBuffer, gradIn: MTLBuffer, gradOut: MTLBuffer,
                siteCount: UInt32, lambda: Float, candDownscale: UInt32,
                in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Diffuse Tau Gradients (Site)"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(gradRaw, offset: 0, index: 1)
        encoder.setBuffer(gradIn, offset: 0, index: 2)
        encoder.setBuffer(gradOut, offset: 0, index: 3)

        var siteCountVar = siteCount
        var lambdaVar = lambda
        var candDownscaleVar = candDownscale
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&lambdaVar, length: MemoryLayout<Float>.stride, index: 5)
        encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 6)

        let tgs = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (Int(siteCount) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(tg, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }
}
