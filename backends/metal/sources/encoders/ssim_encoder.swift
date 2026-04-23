import Metal

/// Encoder for computing SSIM between rendered and target images.
final class SSIMEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "computeSSIM") else {
            throw NSError(domain: "SSIMEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing computeSSIM kernel"])
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Compute SSIM accumulator (sum) - result must be divided by (width * height) on CPU.
    func encode(rendered: MTLTexture, target: MTLTexture, mask: MTLTexture,
                ssimBuffer: MTLBuffer, in commandBuffer: MTLCommandBuffer) {
        let width = rendered.width
        let height = rendered.height

        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: ssimBuffer, range: 0..<MemoryLayout<Float>.stride, value: 0)
            blit.endEncoding()
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Compute SSIM"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(rendered, index: 0)
        encoder.setTexture(target, index: 1)
        encoder.setTexture(mask, index: 2)
        encoder.setBuffer(ssimBuffer, offset: 0, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
