import Metal

/// Encoder for Jump Flood Algorithm - true flood propagation for 4 closest sites
final class JFAEncoder {
    private let device: MTLDevice
    private let clearPipeline: MTLComputePipelineState
    private let seedPipeline: MTLComputePipelineState
    private let floodPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let clearFunc = library.makeFunction(name: "jfaClearCandidates"),
              let seedFunc = library.makeFunction(name: "jfaSeed"),
              let floodFunc = library.makeFunction(name: "jfaFlood") else {
            throw NSError(domain: "JFAEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing JFA kernels"])
        }
        self.clearPipeline = try device.makeComputePipelineState(function: clearFunc)
        self.seedPipeline = try device.makeComputePipelineState(function: seedFunc)
        self.floodPipeline = try device.makeComputePipelineState(function: floodFunc)
    }

    /// Run full JFA: seed + log2(maxDim) flood passes
    /// Writes 4 closest sites into cand0. cand1 is used as ping-pong buffer.
    /// After completion, result is in cand0.
    func encodeJFA(cand0: MTLTexture, cand1: MTLTexture,
                   sitesBuffer: MTLBuffer, siteCount: UInt32,
                   invScaleSq: Float, candDownscale: UInt32,
                   in commandBuffer: MTLCommandBuffer) {
        let width = cand0.width
        let height = cand0.height
        let maxDim = max(width, height)

        // Compute number of flood passes (log2 of maxDim, rounded up)
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
            encoder.label = "JFA Seed"
            encoder.setComputePipelineState(seedPipeline)
            encoder.setTexture(cand0, index: 0)
            encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

            var siteCountVar = siteCount
            var candDownscaleVar = candDownscale
            encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 2)

            let threadGroupSize = min(256, Int(siteCount))
            let threadGroups = (Int(siteCount) + threadGroupSize - 1) / threadGroupSize
            encoder.dispatchThreadgroups(MTLSize(width: threadGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1))
            encoder.endEncoding()
        }

        // --- Flood passes: step sizes N/2, N/4, ..., 1 ---
        var stepSize = step / 2  // Start from half of next power of 2
        var readFromCand0 = true

        let threadGroupSize2D = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups2D = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1)

        while stepSize >= 1 {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { break }
            encoder.label = "JFA Flood (step=\(stepSize))"
            encoder.setComputePipelineState(floodPipeline)

            // Ping-pong between cand0 and cand1
            if readFromCand0 {
                encoder.setTexture(cand0, index: 0)  // read
                encoder.setTexture(cand1, index: 1)  // write
            } else {
                encoder.setTexture(cand1, index: 0)  // read
                encoder.setTexture(cand0, index: 1)  // write
            }

            encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

            var siteCountVar = siteCount
            var stepSizeVar = UInt32(stepSize)
            var invScaleSqVar = invScaleSq

            var candDownscaleVar = candDownscale
            encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBytes(&stepSizeVar, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 3)
            encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 4)

            encoder.dispatchThreadgroups(threadGroups2D, threadsPerThreadgroup: threadGroupSize2D)
            encoder.endEncoding()

            readFromCand0 = !readFromCand0
            stepSize /= 2
        }

        // --- Ensure result ends up in cand0 ---
        // The first flood pass writes cand1, so odd pass counts finish in cand1.
        if numPasses % 2 == 1 {
            // Result is in cand1, copy to cand0
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

    /// Seed only: write the home-pixel site id into cand0.x without clearing.
    func encodeSeed(cand0: MTLTexture,
                    sitesBuffer: MTLBuffer, siteCount: UInt32,
                    candDownscale: UInt32,
                    in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "JFA Seed Only"
        encoder.setComputePipelineState(seedPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var siteCountVar = siteCount
        var candDownscaleVar = candDownscale
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&candDownscaleVar, length: MemoryLayout<UInt32>.stride, index: 2)

        let threadGroupSize = min(256, Int(siteCount))
        let threadGroups = (Int(siteCount) + threadGroupSize - 1) / threadGroupSize
        encoder.dispatchThreadgroups(MTLSize(width: threadGroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
