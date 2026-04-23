import Metal

/// Encoder for clearing float buffers on GPU (replaces CPU memset)
final class ClearEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "clearAtomicBuffer") else {
            throw NSError(domain: "ClearEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing clearAtomicBuffer kernel"])
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Clear a float buffer to zero
    func encode(buffer: MTLBuffer, count: UInt32, in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Clear Buffer"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        var countVar = count
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.stride, index: 1)

        let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (Int(count) + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }

    /// Clear multiple buffers in a single command buffer
    func encode(buffers: [MTLBuffer], count: UInt32, in commandBuffer: MTLCommandBuffer) {
        for buffer in buffers {
            encode(buffer: buffer, count: count, in: commandBuffer)
        }
    }
}
