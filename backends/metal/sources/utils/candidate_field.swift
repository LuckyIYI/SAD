import Metal

struct CandidateTextures {
    var cand0A: MTLTexture
    var cand1A: MTLTexture
    var cand0B: MTLTexture
    var cand1B: MTLTexture
}

func jumpStepForIndex(_ stepIndex: UInt32, width: Int, height: Int) -> UInt32 {
    let maxDim = UInt32(max(width, height))
    var pow2: UInt32 = 1
    while pow2 < maxDim {
        pow2 <<= 1
    }
    if pow2 <= 1 {
        return 1
    }
    var stages: UInt32 = 0
    var tmp = pow2
    while tmp > 1 {
        tmp >>= 1
        stages += 1
    }
    let stage: UInt32
    if stages > 0 {
        stage = stepIndex >= stages ? (stages - 1) : stepIndex
    } else {
        stage = 0
    }
    let step = pow2 >> (stage + 1)
    return max(step, 1)
}

func packJumpStep(_ stepIndex: UInt32, width: Int, height: Int) -> UInt32 {
    let jumpStep = min(jumpStepForIndex(stepIndex, width: width, height: height), 0xffff)
    return (jumpStep << 16) | (stepIndex & 0xffff)
}

func makeCandidateTextures(device: MTLDevice, width: Int, height: Int, downscale: Int = 1) -> CandidateTextures {
    let scale = max(1, downscale)
    let candWidth = max(1, (width + scale - 1) / scale)
    let candHeight = max(1, (height + scale - 1) / scale)
    let candDesc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Uint,
        width: candWidth,
        height: candHeight,
        mipmapped: false)
    candDesc.usage = [.shaderRead, .shaderWrite]

    return CandidateTextures(
        cand0A: device.makeTexture(descriptor: candDesc)!,
        cand1A: device.makeTexture(descriptor: candDesc)!,
        cand0B: device.makeTexture(descriptor: candDesc)!,
        cand1B: device.makeTexture(descriptor: candDesc)!
    )
}

func packCandidateSites(encoder: CandidatePackEncoder,
                        commandBuffer: MTLCommandBuffer,
                        sitesBuffer: MTLBuffer,
                        packedBuffer: MTLBuffer,
                        siteCount: UInt32) {
    guard siteCount > 0 else { return }
    encoder.encode(sitesBuffer: sitesBuffer, packedBuffer: packedBuffer,
                   siteCount: siteCount, in: commandBuffer)
}

func updateCandidatesCompact(encoder: CompactCandidatesEncoder,
                             commandBuffer: MTLCommandBuffer,
                             candidates: inout CandidateTextures,
                             packedSitesBuffer: MTLBuffer,
                             siteCount: UInt32,
                             width: Int,
                             height: Int,
                             targetWidth: Int,
                             targetHeight: Int,
                             candDownscale: Int,
                             invScaleSq: Float,
                             radiusScale: Float,
                             radiusProbes: UInt32,
                             injectCount: UInt32,
                             hilbertOrder: MTLBuffer? = nil,
                             hilbertPos: MTLBuffer? = nil,
                             hilbertProbeCount: UInt32 = 0,
                             hilbertWindow: UInt32 = 0,
                             passes: Int,
                             jumpPassIndex: inout UInt32) {
    guard passes > 0 else { return }
    for _ in 0..<passes {
        let step = packJumpStep(jumpPassIndex, width: width, height: height)
        let stepHigh = jumpPassIndex >> 16
        encoder.encodeUpdate(cand0In: candidates.cand0A, cand1In: candidates.cand1A,
                             cand0Out: candidates.cand0B, cand1Out: candidates.cand1B,
                             packedSitesBuffer: packedSitesBuffer, siteCount: siteCount,
                             step: step, stepHigh: stepHigh, invScaleSq: invScaleSq,
                             radiusScale: radiusScale,
                             radiusProbes: radiusProbes,
                             injectCount: injectCount,
                             candDownscale: UInt32(max(1, candDownscale)),
                             targetWidth: UInt32(max(1, targetWidth)),
                             targetHeight: UInt32(max(1, targetHeight)),
                             hilbertOrder: hilbertOrder,
                             hilbertPos: hilbertPos,
                             hilbertProbeCount: hilbertProbeCount,
                             hilbertWindow: hilbertWindow,
                             in: commandBuffer)
        jumpPassIndex &+= 1
        swap(&candidates.cand0A, &candidates.cand0B)
        swap(&candidates.cand1A, &candidates.cand1B)
    }
}

func updateCandidatesPacked(encoder: PackedCandidatesEncoder,
                            commandBuffer: MTLCommandBuffer,
                            candidates: inout CandidateTextures,
                            sitesBuffer: MTLBuffer,
                            quant: PackedSiteQuant,
                            siteCount: UInt32,
                            width: Int,
                            height: Int,
                            targetWidth: Int,
                            targetHeight: Int,
                            candDownscale: Int,
                            invScaleSq: Float,
                            radiusScale: Float,
                            radiusProbes: UInt32,
                            injectCount: UInt32,
                            passes: Int,
                            jumpPassIndex: inout UInt32) {
    guard passes > 0 else { return }
    for _ in 0..<passes {
        let step = packJumpStep(jumpPassIndex, width: width, height: height)
        let stepHigh = jumpPassIndex >> 16
        encoder.encodeUpdate(cand0In: candidates.cand0A, cand1In: candidates.cand1A,
                             cand0Out: candidates.cand0B, cand1Out: candidates.cand1B,
                             sitesBuffer: sitesBuffer, quant: quant, siteCount: siteCount,
                             step: step, stepHigh: stepHigh, invScaleSq: invScaleSq,
                             radiusScale: radiusScale,
                             radiusProbes: radiusProbes,
                             injectCount: injectCount,
                             candDownscale: UInt32(max(1, candDownscale)),
                             targetWidth: UInt32(max(1, targetWidth)),
                             targetHeight: UInt32(max(1, targetHeight)),
                             in: commandBuffer)
        jumpPassIndex &+= 1
        swap(&candidates.cand0A, &candidates.cand0B)
        swap(&candidates.cand1A, &candidates.cand1B)
    }
}
