import Metal

final class CompactCandidatesEncoder {
    private let updatePipeline: MTLComputePipelineState
    private let dummyBuffer: MTLBuffer

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let updateFunc = library.makeFunction(name: "updateCandidatesCompact") else {
            throw NSError(domain: "CompactCandidatesEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing compact candidate kernel"])
        }
        self.updatePipeline = try device.makeComputePipelineState(function: updateFunc)
        self.dummyBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    }

    private func encodeUpdate(encoder: MTLComputeCommandEncoder,
                              cand0In: MTLTexture, cand1In: MTLTexture,
                              cand0Out: MTLTexture, cand1Out: MTLTexture,
                              packedSitesBuffer: MTLBuffer, siteCount: UInt32,
                              step: UInt32, stepHigh: UInt32, invScaleSq: Float,
                              radiusScale: Float, radiusProbes: UInt32, injectCount: UInt32,
                              candDownscale: UInt32, targetWidth: UInt32, targetHeight: UInt32,
                              hilbertOrder: MTLBuffer?, hilbertPos: MTLBuffer?,
                              hilbertProbeCount: UInt32, hilbertWindow: UInt32) {
        encoder.setComputePipelineState(updatePipeline)
        encoder.setTexture(cand0In, index: 0)
        encoder.setTexture(cand1In, index: 1)
        encoder.setTexture(cand0Out, index: 2)
        encoder.setTexture(cand1Out, index: 3)
        encoder.setBuffer(packedSitesBuffer, offset: 0, index: 0)
        encoder.setBuffer(hilbertOrder ?? dummyBuffer, offset: 0, index: 8)
        encoder.setBuffer(hilbertPos ?? dummyBuffer, offset: 0, index: 9)

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
        var hilbertProbeVar = hilbertProbeCount
        var hilbertWindowVar = hilbertWindow

        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&stepVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 3)
        encoder.setBytes(&stepHighVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&radiusScaleVar, length: MemoryLayout<Float>.stride, index: 5)
        encoder.setBytes(&radiusProbesVar, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&injectCountVar, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&hilbertProbeVar, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&hilbertWindowVar, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&targetWidthVar, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&targetHeightVar, length: MemoryLayout<UInt32>.stride, index: 14)

        let width = cand0In.width
        let height = cand0In.height
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(width: (width + 15) / 16,
                                   height: (height + 15) / 16,
                                   depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }

    func encodeUpdate(cand0In: MTLTexture, cand1In: MTLTexture,
                      cand0Out: MTLTexture, cand1Out: MTLTexture,
                      packedSitesBuffer: MTLBuffer, siteCount: UInt32,
                      step: UInt32, stepHigh: UInt32, invScaleSq: Float,
                      radiusScale: Float, radiusProbes: UInt32, injectCount: UInt32,
                      candDownscale: UInt32, targetWidth: UInt32, targetHeight: UInt32,
                      hilbertOrder: MTLBuffer? = nil, hilbertPos: MTLBuffer? = nil,
                      hilbertProbeCount: UInt32 = 0, hilbertWindow: UInt32 = 0,
                      in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Update Candidates Compact"
        encodeUpdate(encoder: encoder,
                     cand0In: cand0In, cand1In: cand1In,
                     cand0Out: cand0Out, cand1Out: cand1Out,
                     packedSitesBuffer: packedSitesBuffer, siteCount: siteCount,
                     step: step, stepHigh: stepHigh, invScaleSq: invScaleSq,
                     radiusScale: radiusScale, radiusProbes: radiusProbes,
                     injectCount: injectCount,
                     candDownscale: candDownscale, targetWidth: targetWidth, targetHeight: targetHeight,
                     hilbertOrder: hilbertOrder, hilbertPos: hilbertPos,
                     hilbertProbeCount: hilbertProbeCount, hilbertWindow: hilbertWindow)
        encoder.endEncoding()
    }

    func encodeUpdate(cand0In: MTLTexture, cand1In: MTLTexture,
                      cand0Out: MTLTexture, cand1Out: MTLTexture,
                      packedSitesBuffer: MTLBuffer, siteCount: UInt32,
                      step: UInt32, stepHigh: UInt32, invScaleSq: Float,
                      radiusScale: Float, radiusProbes: UInt32, injectCount: UInt32,
                      candDownscale: UInt32, targetWidth: UInt32, targetHeight: UInt32,
                      hilbertOrder: MTLBuffer? = nil, hilbertPos: MTLBuffer? = nil,
                      hilbertProbeCount: UInt32 = 0, hilbertWindow: UInt32 = 0,
                      in encoder: MTLComputeCommandEncoder) {
        encodeUpdate(encoder: encoder,
                     cand0In: cand0In, cand1In: cand1In,
                     cand0Out: cand0Out, cand1Out: cand1Out,
                     packedSitesBuffer: packedSitesBuffer, siteCount: siteCount,
                     step: step, stepHigh: stepHigh, invScaleSq: invScaleSq,
                     radiusScale: radiusScale, radiusProbes: radiusProbes,
                     injectCount: injectCount,
                     candDownscale: candDownscale, targetWidth: targetWidth, targetHeight: targetHeight,
                     hilbertOrder: hilbertOrder, hilbertPos: hilbertPos,
                     hilbertProbeCount: hilbertProbeCount, hilbertWindow: hilbertWindow)
    }
}
