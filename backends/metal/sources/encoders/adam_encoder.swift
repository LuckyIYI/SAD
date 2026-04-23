import Metal

/// Parameters for Adam optimizer
struct AdamParams {
    var lrPos: Float
    var lrTau: Float
    var lrRadius: Float
    var lrColor: Float
    var lrDir: Float
    var lrAniso: Float
    var beta1: Float
    var beta2: Float
    var eps: Float
    var t: UInt32
    var width: UInt32
    var height: UInt32
}

/// Encoder for Adam optimizer update step
final class AdamEncoder {
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let function = library.makeFunction(name: "adamUpdate") else {
            throw NSError(domain: "AdamEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing adamUpdate kernel"])
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Encode Adam update step
    func encode(sitesBuffer: MTLBuffer, adamBuffer: MTLBuffer,
                gradBuffers: [MTLBuffer],  // 10 gradient buffers
                params: AdamParams, siteCount: Int,
                in commandBuffer: MTLCommandBuffer) {
        guard gradBuffers.count == 10 else {
            fatalError("AdamEncoder requires exactly 10 gradient buffers")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Adam Update"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(adamBuffer, offset: 0, index: 1)

        for i in 0..<10 {
            encoder.setBuffer(gradBuffers[i], offset: 0, index: i + 2)
        }

        var lrPos = params.lrPos
        var lrTau = params.lrTau
        var lrRadius = params.lrRadius
        var lrColor = params.lrColor
        var lrDir = params.lrDir
        var lrAniso = params.lrAniso
        var beta1 = params.beta1
        var beta2 = params.beta2
        var eps = params.eps
        var t = params.t
        var width = params.width
        var height = params.height

        encoder.setBytes(&lrPos, length: MemoryLayout<Float>.stride, index: 12)
        encoder.setBytes(&lrTau, length: MemoryLayout<Float>.stride, index: 13)
        encoder.setBytes(&lrRadius, length: MemoryLayout<Float>.stride, index: 14)
        encoder.setBytes(&lrColor, length: MemoryLayout<Float>.stride, index: 15)
        encoder.setBytes(&lrDir, length: MemoryLayout<Float>.stride, index: 16)
        encoder.setBytes(&lrAniso, length: MemoryLayout<Float>.stride, index: 17)
        encoder.setBytes(&beta1, length: MemoryLayout<Float>.stride, index: 18)
        encoder.setBytes(&beta2, length: MemoryLayout<Float>.stride, index: 19)
        encoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 20)
        encoder.setBytes(&t, length: MemoryLayout<UInt32>.stride, index: 21)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.stride, index: 22)
        encoder.setBytes(&height, length: MemoryLayout<UInt32>.stride, index: 23)

        let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (siteCount + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
