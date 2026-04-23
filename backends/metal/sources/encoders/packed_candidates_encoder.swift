import Metal

final class PackedCandidatesEncoder {
    private let initPipeline: MTLComputePipelineState
    private let updatePipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let initFunc = library.makeFunction(name: "initCandidates"),
              let updateFunc = library.makeFunction(name: "updateCandidatesPacked") else {
            throw NSError(domain: "PackedCandidatesEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing packed candidate kernels"])
        }
        self.initPipeline = try device.makeComputePipelineState(function: initFunc)
        self.updatePipeline = try device.makeComputePipelineState(function: updateFunc)
    }

    func encodeInit(cand0: MTLTexture, cand1: MTLTexture,
                    siteCount: UInt32, seed: UInt32, perPixelMode: Bool,
                    in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Init Candidates"
        encoder.setComputePipelineState(initPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)

        var siteCountVar = siteCount
        var seedVar = seed
        var perPixelVar = perPixelMode

        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 0)
        encoder.setBytes(&seedVar, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&perPixelVar, length: MemoryLayout<Bool>.stride, index: 2)

        let width = cand0.width
        let height = cand0.height
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }

    func encodeUpdate(cand0In: MTLTexture, cand1In: MTLTexture,
                      cand0Out: MTLTexture, cand1Out: MTLTexture,
                      sitesBuffer: MTLBuffer, quant: PackedSiteQuant, siteCount: UInt32,
                      step: UInt32, stepHigh: UInt32, invScaleSq: Float,
                      radiusScale: Float, radiusProbes: UInt32, injectCount: UInt32,
                      candDownscale: UInt32, targetWidth: UInt32, targetHeight: UInt32,
                      in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Update Candidates Packed"
        encoder.setComputePipelineState(updatePipeline)
        encoder.setTexture(cand0In, index: 0)
        encoder.setTexture(cand1In, index: 1)
        encoder.setTexture(cand0Out, index: 2)
        encoder.setTexture(cand1Out, index: 3)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var quantVar = quant
        var siteCountVar = siteCount
        var stepVar = step
        var invScaleSqVar = invScaleSq
        var stepHighVar = stepHigh
        var radiusScaleVar = radiusScale
        var radiusProbesVar = radiusProbes
        var injectCountVar = injectCount
        var candDownscaleVar = candDownscale
        var targetWidthVar = targetWidth
        var targetHeightVar = targetHeight

        encoder.setBytes(&quantVar, length: MemoryLayout<PackedSiteQuant>.stride, index: 1)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&stepVar, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 4)
        encoder.setBytes(&stepHighVar, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&radiusScaleVar, length: MemoryLayout<Float>.stride, index: 6)
        encoder.setBytes(&radiusProbesVar, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&injectCountVar, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&targetWidthVar, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&targetHeightVar, length: MemoryLayout<UInt32>.stride, index: 11)

        let width = cand0In.width
        let height = cand0In.height
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
