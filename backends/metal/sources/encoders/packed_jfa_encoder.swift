import Metal

final class PackedJFAEncoder {
    private let device: MTLDevice
    private let clearPipeline: MTLComputePipelineState
    private let seedPipeline: MTLComputePipelineState
    private let floodPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let clearFunc = library.makeFunction(name: "jfaClearCandidates"),
              let seedFunc = library.makeFunction(name: "jfaSeedPacked"),
              let floodFunc = library.makeFunction(name: "jfaFloodPacked") else {
            throw NSError(domain: "PackedJFAEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing packed JFA kernels"])
        }
        self.clearPipeline = try device.makeComputePipelineState(function: clearFunc)
        self.seedPipeline = try device.makeComputePipelineState(function: seedFunc)
        self.floodPipeline = try device.makeComputePipelineState(function: floodFunc)
    }

    func encodeJFA(cand0: MTLTexture, cand1: MTLTexture,
                   sitesBuffer: MTLBuffer, quant: PackedSiteQuant, siteCount: UInt32,
                   invScaleSq: Float, candDownscale: UInt32,
                   targetWidth: UInt32, targetHeight: UInt32,
                   in commandBuffer: MTLCommandBuffer) {
        let width = cand0.width
        let height = cand0.height
        let maxDim = max(width, height)

        var numPasses = 0
        var step = 1
        while step < maxDim {
            step *= 2
            numPasses += 1
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "JFA Clear"
            encoder.setComputePipelineState(clearPipeline)
            encoder.setTexture(cand0, index: 0)
            encoder.setTexture(cand1, index: 1)
            let threadGroupSize2D = MTLSize(width: 16, height: 16, depth: 1)
            let threadGroups2D = MTLSize(
                width: (width + 15) / 16,
                height: (height + 15) / 16,
                depth: 1)
            encoder.dispatchThreadgroups(threadGroups2D, threadsPerThreadgroup: threadGroupSize2D)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "JFA Seed Packed"
            encoder.setComputePipelineState(seedPipeline)
            encoder.setTexture(cand0, index: 0)
            encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

            var quantVar = quant
            var siteCountVar = siteCount
            var candDownscaleVar = candDownscale
            var targetWidthVar = targetWidth
            var targetHeightVar = targetHeight
            encoder.setBytes(&quantVar, length: MemoryLayout<PackedSiteQuant>.stride, index: 1)
            encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&targetWidthVar, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&targetHeightVar, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadGroupSize = min(256, Int(siteCount))
            let threadGroups = (Int(siteCount) + threadGroupSize - 1) / threadGroupSize
            encoder.dispatchThreadgroups(MTLSize(width: threadGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        var stepSize = step / 2
        var readFromCand0 = true

        let threadGroupSize2D = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups2D = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1)

        while stepSize >= 1 {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { break }
            encoder.label = "JFA Flood Packed (step=\(stepSize))"
            encoder.setComputePipelineState(floodPipeline)

            if readFromCand0 {
                encoder.setTexture(cand0, index: 0)
                encoder.setTexture(cand1, index: 1)
            } else {
                encoder.setTexture(cand1, index: 0)
                encoder.setTexture(cand0, index: 1)
            }

            encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

            var quantVar = quant
            var siteCountVar = siteCount
            var stepSizeVar = UInt32(stepSize)
            var invScaleSqVar = invScaleSq
            var candDownscaleVar = candDownscale
            var targetWidthVar = targetWidth
            var targetHeightVar = targetHeight

            encoder.setBytes(&quantVar, length: MemoryLayout<PackedSiteQuant>.stride, index: 1)
            encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&stepSizeVar, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 4)
            encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&targetWidthVar, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&targetHeightVar, length: MemoryLayout<UInt32>.stride, index: 7)

            encoder.dispatchThreadgroups(threadGroups2D, threadsPerThreadgroup: threadGroupSize2D)
            encoder.endEncoding()

            readFromCand0 = !readFromCand0
            stepSize /= 2
        }

        // The first flood pass writes cand1, so odd pass counts finish in cand1.
        if numPasses % 2 == 1 {
            if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
                blitEncoder.label = "JFA Copy Result"
                blitEncoder.copy(from: cand1,
                                sourceSlice: 0, sourceLevel: 0,
                                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                                sourceSize: MTLSize(width: width, height: height, depth: 1),
                                to: cand0,
                                destinationSlice: 0, destinationLevel: 0,
                                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
                blitEncoder.endEncoding()
            }
        }
    }

    func encodeSeed(cand0: MTLTexture,
                    sitesBuffer: MTLBuffer, quant: PackedSiteQuant, siteCount: UInt32,
                    candDownscale: UInt32, targetWidth: UInt32, targetHeight: UInt32,
                    in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "JFA Seed Packed Only"
        encoder.setComputePipelineState(seedPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var quantVar = quant
        var siteCountVar = siteCount
        var candDownscaleVar = candDownscale
        var targetWidthVar = targetWidth
        var targetHeightVar = targetHeight
        encoder.setBytes(&quantVar, length: MemoryLayout<PackedSiteQuant>.stride, index: 1)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&targetWidthVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&targetHeightVar, length: MemoryLayout<UInt32>.stride, index: 5)

        let threadGroupSize = min(256, Int(siteCount))
        let threadGroups = (Int(siteCount) + threadGroupSize - 1) / threadGroupSize
        encoder.dispatchThreadgroups(MTLSize(width: threadGroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
