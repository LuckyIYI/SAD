import Foundation
import Metal

struct RenderOptions {
    var sitesPath: String
    var outputPath: String?
    var outputVoronoiPath: String?
    var renderTargetPath: String?
    var widthOverride: Int?
    var heightOverride: Int?
    var candidatePasses: Int
    var useJFA: Bool
    var jfaRounds: Int
    var candRadiusScale: Float
    var candRadiusProbes: UInt32
    var candInjectCount: UInt32
    var candDownscale: Int
    var candHilbertWindow: UInt32
    var candHilbertProbes: UInt32
    var usePackedInference: Bool
}

func renderVoronoiFromSites(_ options: RenderOptions) {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Metal is not supported on this device")
        return
    }
    print("Using Metal device: \(device.name)")

    let ext = URL(fileURLWithPath: options.sitesPath).pathExtension.lowercased()
    var sites: [VoronoiSite]
    var width: Int
    var height: Int

    if ext == "json" {
        let loaded = loadSitesFromJSON(path: options.sitesPath)
        sites = loaded.sites
        width = loaded.width
        height = loaded.height
    } else {
        let loaded = loadSitesFromTXT(path: options.sitesPath)
        sites = loaded.sites
        width = options.widthOverride ?? loaded.width ?? 0
        height = options.heightOverride ?? loaded.height ?? 0
        if width <= 0 || height <= 0 {
            fatalError("TXT sites require --width and --height or a header line with image size.")
        }
    }

    let nSitesU = UInt32(sites.count)
    print("Loaded \(sites.count) sites")
    print("Render size: \(width)x\(height)")

    let base = (options.sitesPath as NSString).deletingPathExtension

    guard let commandQueue = device.makeCommandQueue() else {
        print("Failed to create command queue")
        return
    }
    guard let library = loadMetalLibrary(device: device) else {
        return
    }

    if options.usePackedInference {
        let packed = makePackedInferenceSites(sites, width: width, height: height)
        let params = PackedRenderParams(
            candidatePasses: options.candidatePasses,
            useJFA: options.useJFA,
            jfaRounds: options.jfaRounds,
            candRadiusScale: options.candRadiusScale,
            candRadiusProbes: options.candRadiusProbes,
            candInjectCount: options.candInjectCount,
            candDownscale: options.candDownscale,
            outputVoronoi: options.outputVoronoiPath != nil
        )
        guard let result = renderPackedInference(packedSites: packed.sites, quant: packed.quant,
                                                width: width, height: height,
                                                device: device, commandQueue: commandQueue,
                                                library: library, params: params) else {
            return
        }

        let imgPath = options.outputPath ?? (base + "_render.png")
        saveTexture(result.render, path: imgPath)
        print("Saved: \(imgPath)")

        if let vPath = options.outputVoronoiPath, let voronoi = result.voronoi {
            saveTexture(voronoi, path: vPath)
            print("Saved: \(vPath)")
        }
        return
    }

    let initCandidatesEncoder = try! PackedCandidatesEncoder(device: device, library: library)
    let candidatePackEncoder = try! CandidatePackEncoder(device: device, library: library)
    let compactCandidatesEncoder = try! CompactCandidatesEncoder(device: device, library: library)
    let renderEncoder = try! RenderEncoder(device: device, library: library)
    let seedEncoder = try! JFAEncoder(device: device, library: library)
    let jfaEncoder = options.useJFA ? seedEncoder : nil
    let psnrEncoder = try! PSNREncoder(device: device, library: library)

    let sitesBuffer = device.makeBuffer(length: MemoryLayout<VoronoiSite>.stride * sites.count, options: .storageModeShared)!
    memcpy(sitesBuffer.contents(), sites, MemoryLayout<VoronoiSite>.stride * sites.count)

    var candidates = makeCandidateTextures(device: device,
                                           width: width,
                                           height: height,
                                           downscale: options.candDownscale)
    let packedCandidatesBuffer = device.makeBuffer(
        length: MemoryLayout<PackedCandidateSite>.stride * sites.count,
        options: .storageModeShared
    )!
    let usesVptHilbert = options.candHilbertProbes > 0 && options.candHilbertWindow > 0
    let hilbertEncoder = usesVptHilbert ? try? HilbertEncoder(device: device, library: library) : nil
    var hilbertResources = usesVptHilbert
        ? makeHilbertResources(device: device, library: library, siteCapacity: sites.count)
        : nil

    let renderDesc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Float,
        width: width,
        height: height,
        mipmapped: false)
    renderDesc.usage = [.shaderRead, .shaderWrite]
    renderDesc.storageMode = .shared
    let renderTexture = device.makeTexture(descriptor: renderDesc)!
    let voronoiTexture = device.makeTexture(descriptor: renderDesc)!

    let scale = Float(max(width, height))
    let invScaleSq = 1.0 as Float / (scale * scale)

    guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

    let seed = UInt32(12345)
    let hilbertBuffers: (order: MTLBuffer?, pos: MTLBuffer?, probes: UInt32, window: UInt32)
    if usesVptHilbert {
        guard let hilbertEncoder = hilbertEncoder, var resources = hilbertResources else {
            fatalError("Hilbert probes require HilbertEncoder.")
        }
        updateHilbertResources(resources: &resources,
                               encoder: hilbertEncoder,
                               commandBuffer: commandBuffer,
                               sitesBuffer: sitesBuffer,
                               siteCount: sites.count,
                               width: width,
                               height: height)
        hilbertResources = resources
        hilbertBuffers = (resources.order, resources.pos,
                          options.candHilbertProbes, options.candHilbertWindow)
    } else {
        hilbertBuffers = (nil, nil, 0, 0)
    }
    initCandidatesEncoder.encodeInit(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                      siteCount: nSitesU, seed: seed, perPixelMode: false,
                                      in: commandBuffer)
    seedEncoder.encodeSeed(cand0: candidates.cand0A,
                           sitesBuffer: sitesBuffer, siteCount: nSitesU,
                           candDownscale: UInt32(max(1, options.candDownscale)),
                           in: commandBuffer)

    let passes = max(0, options.candidatePasses)
    if passes > 0 {
        packCandidateSites(encoder: candidatePackEncoder, commandBuffer: commandBuffer,
                           sitesBuffer: sitesBuffer, packedBuffer: packedCandidatesBuffer,
                           siteCount: nSitesU)
        let prefix = options.useJFA ? "JFAx\(max(1, options.jfaRounds)) + " : ""
        print("Building candidate field (\(prefix)\(passes) VPT passes)...")
    }

    if passes > 0 {
        let rounds = options.useJFA ? max(1, options.jfaRounds) : 1
        let basePasses = passes / rounds
        let remainder = passes % rounds

        var passIndex: UInt32 = 0
        for round in 0..<rounds {
            if options.useJFA, let jfa = jfaEncoder {
                jfa.encodeJFA(cand0: candidates.cand0A, cand1: candidates.cand1A,
                              sitesBuffer: sitesBuffer, siteCount: nSitesU,
                              invScaleSq: invScaleSq,
                              candDownscale: UInt32(max(1, options.candDownscale)),
                              in: commandBuffer)
            }

            let passesThis = basePasses + (round < remainder ? 1 : 0)
            updateCandidatesCompact(encoder: compactCandidatesEncoder, commandBuffer: commandBuffer,
                                    candidates: &candidates,
                                    packedSitesBuffer: packedCandidatesBuffer, siteCount: nSitesU,
                                    width: candidates.cand0A.width, height: candidates.cand0A.height,
                                    targetWidth: width, targetHeight: height,
                                    candDownscale: options.candDownscale,
                                    invScaleSq: invScaleSq,
                                    radiusScale: options.candRadiusScale,
                                    radiusProbes: options.candRadiusProbes,
                                    injectCount: options.candInjectCount,
                                    hilbertOrder: hilbertBuffers.order,
                                    hilbertPos: hilbertBuffers.pos,
                                    hilbertProbeCount: hilbertBuffers.probes,
                                    hilbertWindow: hilbertBuffers.window,
                                    passes: passesThis,
                                    jumpPassIndex: &passIndex)
        }
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    guard let renderBuffer = commandQueue.makeCommandBuffer() else { return }

    renderEncoder.encodeRender(cand0: candidates.cand0A, cand1: candidates.cand1A, output: renderTexture,
                               sitesBuffer: sitesBuffer, invScaleSq: invScaleSq, siteCount: nSitesU,
                               in: renderBuffer)

    if options.outputVoronoiPath != nil {
        renderEncoder.encodeColoring(cand0: candidates.cand0A, cand1: candidates.cand1A, output: voronoiTexture,
                                     sitesBuffer: sitesBuffer, invScaleSq: invScaleSq, siteCount: nSitesU,
                                     in: renderBuffer)
    }

    if let targetPath = options.renderTargetPath {
        if let target = loadTargetImage(path: targetPath, maskPath: nil, maxDimension: max(width, height), device: device),
           target.width == width, target.height == height {
            let psnrBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
            psnrEncoder.encode(rendered: renderTexture, target: target.texture,
                               mask: target.maskTexture,
                               mseBuffer: psnrBuffer, in: renderBuffer)
            renderBuffer.addCompletedHandler { _ in
                let mse = psnrBuffer.contents().load(as: Float.self) / Float(max(1.0, target.maskSum) * 3.0)
                let psnr = mse > 0 ? 20.0 * log10(1.0 / sqrt(mse)) : 100.0
                print(String(format: "Render PSNR: %.2f dB", psnr))
            }
        } else {
            print("Render target size mismatch; skipping PSNR.")
        }
    }

    renderBuffer.commit()
    renderBuffer.waitUntilCompleted()

    let imgPath = options.outputPath ?? (base + "_render.png")
    saveTexture(renderTexture, path: imgPath)
    print("Saved: \(imgPath)")

    if let vPath = options.outputVoronoiPath {
        saveTexture(voronoiTexture, path: vPath)
        print("Saved: \(vPath)")
    }
}
