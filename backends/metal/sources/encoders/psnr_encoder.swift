import Metal

/// Encoder for computing PSNR between rendered and target images
final class PSNREncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "computePSNR") else {
            throw NSError(domain: "PSNREncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing computePSNR kernel"])
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Compute PSNR accumulator (MSE sum) - result must be divided by (width * height * 3) on CPU
    func encode(rendered: MTLTexture, target: MTLTexture, mask: MTLTexture,
                mseBuffer: MTLBuffer, in commandBuffer: MTLCommandBuffer) {
        let width = rendered.width
        let height = rendered.height

        // Clear MSE buffer first using blit
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: mseBuffer, range: 0..<MemoryLayout<Float>.stride, value: 0)
            blit.endEncoding()
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Compute PSNR"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(rendered, index: 0)
        encoder.setTexture(target, index: 1)
        encoder.setTexture(mask, index: 2)
        encoder.setBuffer(mseBuffer, offset: 0, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
