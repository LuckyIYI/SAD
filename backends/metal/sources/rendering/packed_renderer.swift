import Foundation
import Metal

struct PackedRenderParams {
    var candidatePasses: Int
    var useJFA: Bool
    var jfaRounds: Int
    var candRadiusScale: Float
    var candRadiusProbes: UInt32
    var candInjectCount: UInt32
    var candDownscale: Int
    var outputVoronoi: Bool
}

func renderPackedInference(packedSites: [PackedInferenceSite],
                           quant: PackedSiteQuant,
                           width: Int,
                           height: Int,
                           device: MTLDevice,
                           commandQueue: MTLCommandQueue,
                           library: MTLLibrary,
                           params: PackedRenderParams) -> (render: MTLTexture, voronoi: MTLTexture?, centroids: MTLTexture)? {
    guard !packedSites.isEmpty else {
        print("No active sites to render (packed inference).")
        return nil
    }

    let packedCandidatesEncoder = try! PackedCandidatesEncoder(device: device, library: library)
    let packedRenderEncoder = try! PackedRenderEncoder(device: device, library: library)
    let packedSeedEncoder = try! PackedJFAEncoder(device: device, library: library)
    let packedJFAEncoder = params.useJFA ? packedSeedEncoder : nil

    let sitesBuffer = device.makeBuffer(length: MemoryLayout<PackedInferenceSite>.stride * packedSites.count,
                                        options: .storageModeShared)!
    memcpy(sitesBuffer.contents(), packedSites, MemoryLayout<PackedInferenceSite>.stride * packedSites.count)

    var candidates = makeCandidateTextures(device: device,
                                           width: width,
                                           height: height,
                                           downscale: params.candDownscale)

    let renderDesc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Float,
        width: width,
        height: height,
        mipmapped: false)
    renderDesc.usage = [.shaderRead, .shaderWrite]
    renderDesc.storageMode = .shared
    let renderTexture = device.makeTexture(descriptor: renderDesc)!
    let voronoiTexture = device.makeTexture(descriptor: renderDesc)!
    let centroidsTexture = device.makeTexture(descriptor: renderDesc)!

    let invScaleSq = 1.0 as Float / (Float(max(width, height)) * Float(max(width, height)))
    let centroidRadius: Float = 2.0
    let nSitesU = UInt32(packedSites.count)

    guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }

    let seed: UInt32 = 12345
    packedCandidatesEncoder.encodeInit(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                       siteCount: nSitesU, seed: seed, perPixelMode: false,
                                       in: commandBuffer)
    packedSeedEncoder.encodeSeed(cand0: candidates.cand0A,
                                 sitesBuffer: sitesBuffer, quant: quant, siteCount: nSitesU,
                                 candDownscale: UInt32(max(1, params.candDownscale)),
                                 targetWidth: UInt32(max(1, width)),
                                 targetHeight: UInt32(max(1, height)),
                                 in: commandBuffer)

    let passes = max(0, params.candidatePasses)
    if passes > 0 {
        let prefix = params.useJFA ? "JFAx\(max(1, params.jfaRounds)) + " : ""
        print("Building packed candidate field (\(prefix)\(passes) VPT passes)...")
    }

    if passes > 0 {
        let rounds = params.useJFA ? max(1, params.jfaRounds) : 1
        let basePasses = passes / rounds
        let remainder = passes % rounds

        var passIndex: UInt32 = 0
        for round in 0..<rounds {
            if params.useJFA, let jfa = packedJFAEncoder {
                jfa.encodeJFA(cand0: candidates.cand0A, cand1: candidates.cand0B,
                              sitesBuffer: sitesBuffer, quant: quant, siteCount: nSitesU,
                              invScaleSq: invScaleSq,
                              candDownscale: UInt32(max(1, params.candDownscale)),
                              targetWidth: UInt32(max(1, width)),
                              targetHeight: UInt32(max(1, height)),
                              in: commandBuffer)
            }

            let passesThis = basePasses + (round < remainder ? 1 : 0)
            updateCandidatesPacked(encoder: packedCandidatesEncoder,
                                   commandBuffer: commandBuffer,
                                   candidates: &candidates,
                                   sitesBuffer: sitesBuffer,
                                   quant: quant,
                                   siteCount: nSitesU,
                                   width: candidates.cand0A.width,
                                   height: candidates.cand0A.height,
                                   targetWidth: width,
                                   targetHeight: height,
                                   candDownscale: params.candDownscale,
                                   invScaleSq: invScaleSq,
                                   radiusScale: params.candRadiusScale,
                                   radiusProbes: params.candRadiusProbes,
                                   injectCount: params.candInjectCount,
                                   passes: passesThis,
                                   jumpPassIndex: &passIndex)
        }
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    guard let renderBuffer = commandQueue.makeCommandBuffer() else { return nil }

    packedRenderEncoder.encodeRender(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                     output: renderTexture,
                                     sitesBuffer: sitesBuffer, quant: quant,
                                     invScaleSq: invScaleSq, siteCount: nSitesU,
                                     in: renderBuffer)

    packedRenderEncoder.encodeCentroids(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                        output: centroidsTexture,
                                        sitesBuffer: sitesBuffer, quant: quant,
                                        dotRadius: centroidRadius, siteCount: nSitesU,
                                        in: renderBuffer)

    if params.outputVoronoi {
        packedRenderEncoder.encodeColoring(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                           output: voronoiTexture,
                                           sitesBuffer: sitesBuffer, quant: quant,
                                           invScaleSq: invScaleSq, siteCount: nSitesU,
                                           in: renderBuffer)
    }

    renderBuffer.commit()
    renderBuffer.waitUntilCompleted()

    return (renderTexture, params.outputVoronoi ? voronoiTexture : nil, centroidsTexture)
}

func renderPackedWithCandidates(cand0: MTLTexture,
                                cand1: MTLTexture,
                                packedSites: [PackedInferenceSite],
                                quant: PackedSiteQuant,
                                device: MTLDevice,
                                commandQueue: MTLCommandQueue,
                                library: MTLLibrary,
                                outputVoronoi: Bool) -> (render: MTLTexture, voronoi: MTLTexture?, centroids: MTLTexture)? {
    guard !packedSites.isEmpty else {
        print("No sites to render (packed inference).")
        return nil
    }

    let packedRenderEncoder = try! PackedRenderEncoder(device: device, library: library)

    let sitesBuffer = device.makeBuffer(length: MemoryLayout<PackedInferenceSite>.stride * packedSites.count,
                                        options: .storageModeShared)!
    memcpy(sitesBuffer.contents(), packedSites, MemoryLayout<PackedInferenceSite>.stride * packedSites.count)

    let width = cand0.width
    let height = cand0.height
    let renderDesc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Float,
        width: width,
        height: height,
        mipmapped: false)
    renderDesc.usage = [.shaderRead, .shaderWrite]
    renderDesc.storageMode = .shared
    let renderTexture = device.makeTexture(descriptor: renderDesc)!
    let voronoiTexture = device.makeTexture(descriptor: renderDesc)!
    let centroidsTexture = device.makeTexture(descriptor: renderDesc)!

    let invScaleSq = 1.0 as Float / (Float(max(width, height)) * Float(max(width, height)))
    let centroidRadius: Float = 2.0
    let nSitesU = UInt32(packedSites.count)

    guard let renderBuffer = commandQueue.makeCommandBuffer() else { return nil }

    packedRenderEncoder.encodeRender(cand0: cand0, cand1: cand1,
                                     output: renderTexture,
                                     sitesBuffer: sitesBuffer, quant: quant,
                                     invScaleSq: invScaleSq, siteCount: nSitesU,
                                     in: renderBuffer)

    packedRenderEncoder.encodeCentroids(cand0: cand0, cand1: cand1,
                                        output: centroidsTexture,
                                        sitesBuffer: sitesBuffer, quant: quant,
                                        dotRadius: centroidRadius, siteCount: nSitesU,
                                        in: renderBuffer)

    if outputVoronoi {
        packedRenderEncoder.encodeColoring(cand0: cand0, cand1: cand1,
                                           output: voronoiTexture,
                                           sitesBuffer: sitesBuffer, quant: quant,
                                           invScaleSq: invScaleSq, siteCount: nSitesU,
                                           in: renderBuffer)
    }

    renderBuffer.commit()
    renderBuffer.waitUntilCompleted()

    return (renderTexture, outputVoronoi ? voronoiTexture : nil, centroidsTexture)
}
