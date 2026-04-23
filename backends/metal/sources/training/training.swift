import Metal
import Foundation
import Dispatch

private struct CandidateUpdatePlan {
    let shouldUpdate: Bool
    let passes: Int
}

private func candidateUpdatePlan(iter: Int, options: TrainingOptions,
                                 effectivePruneStartIter: Int) -> CandidateUpdatePlan {
    if options.initPerPixel {
        if iter < effectivePruneStartIter {
            return CandidateUpdatePlan(shouldUpdate: false, passes: 0)
        }
    }

    let shouldUpdate = (options.candUpdateFreq > 0) ? (iter % options.candUpdateFreq == 0) : false
    return CandidateUpdatePlan(shouldUpdate: shouldUpdate,
                               passes: shouldUpdate ? max(1, options.candUpdatePasses) : 0)
}

private func initModeSummary(_ mode: InitMode) -> String {
    switch mode {
    case .fromSites:
        return "sites"
    case .perPixel:
        return "per-pixel"
    case .gradientWeighted:
        return "gradient"
    }
}

private func printTrainingOverview(device: MTLDevice,
                                   options: TrainingOptions,
                                   target: TargetImage,
                                   actualSites: Int,
                                   activeSites: Int,
                                   plan: SiteCapacityPlan) {
    let masked = options.maskPath == nil ? "no" : "yes"
    let bpp = (options.targetBpp ?? 0.0) > 0.0
        ? String(format: " | target-bpp=%.3f", options.targetBpp ?? 0.0)
        : ""
    print("Training | backend=metal | device=\(device.name) | image=\(target.width)x\(target.height) | sites=\(activeSites)/\(actualSites) | iters=\(options.iterations) | log-freq=\(options.logFreq) | mask=\(masked) | out=\(options.outputDir)\(bpp)")

    let densify = options.densifyEnabled
        ? "on cap=\(plan.maxSitesCapacity)"
        : "off"
    let pruneEnd = options.pruneEndIter ?? (options.iterations - 1)
    let prune = options.prunePercentile > 0.0
        ? String(format: "on %.3f @%d-%d/%d", options.prunePercentile, options.pruneStartIter, pruneEnd, max(1, options.pruneFreq))
        : "off"
    let hilbert = options.candHilbertProbes > 0 && options.candHilbertWindow > 0
        ? " hilbert=\(options.candHilbertProbes)x\(options.candHilbertWindow)"
        : ""
    print("Schedule | init=\(initModeSummary(options.initMode)) | densify=\(densify) | prune=\(prune) | cand=freq \(options.candUpdateFreq), passes \(options.candUpdatePasses), downscale \(options.candDownscale)x\(hilbert)")
}

func trainVoronoi(_ options: TrainingOptions) {
    var options = options
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Metal is not supported on this device")
        return
    }

    guard let target = loadTargetImage(path: options.targetPath,
                                       maskPath: options.maskPath,
                                       maxDimension: options.maxDim,
                                       device: device) else {
        print("Failed to load target image")
        return
    }

    guard let commandQueue = device.makeCommandQueue() else {
        print("Failed to create command queue")
        return
    }

    guard let library = loadMetalLibrary(device: device) else {
        return
    }

    let clearEncoder = try! ClearEncoder(device: device, library: library)
    let psnrEncoder = try! PSNREncoder(device: device, library: library)
    let ssimEncoder = try! SSIMEncoder(device: device, library: library)
    let adamEncoder = try! AdamEncoder(device: device, library: library)
    let initCandidatesEncoder = try! PackedCandidatesEncoder(device: device, library: library)
    let seedEncoder = try! JFAEncoder(device: device, library: library)
    let candidatePackEncoder = try! CandidatePackEncoder(device: device, library: library)
    let compactCandidatesEncoder = try! CompactCandidatesEncoder(device: device, library: library)
    let renderEncoder = try! RenderEncoder(device: device, library: library)
    let tauDiffusionEncoder = try! TauDiffusionEncoder(device: device, library: library)
    let tiledGradientEncoder = try! TiledGradientEncoder(device: device, library: library)
    let densifyEncoder = try! DensifyEncoder(device: device, library: library)
    let pruneEncoder = try! PruneEncoder(device: device, library: library)
    let usesVptHilbert = options.candHilbertProbes > 0 && options.candHilbertWindow > 0
    let hilbertEncoder = usesVptHilbert
        ? try? HilbertEncoder(device: device, library: library)
        : nil

    let sites = initializeSites(options: options, target: target,
                                device: device, commandQueue: commandQueue,
                                library: library)

    var actualNSites = sites.count
    var activeSitesEstimate = sites.reduce(0) { $0 + ($1.position.x >= 0.0 ? 1 : 0) }

    let numPixels = target.width * target.height
    if let targetBpp = options.targetBpp, targetBpp > 0.0 {
        let pruneEndIter = options.pruneEndIter ?? (options.iterations - 1)
        let maxSitesCapacity: Int
        if options.maxSites > 0 {
            maxSitesCapacity = options.maxSites
        } else if options.densifyEnabled {
            maxSitesCapacity = min(numPixels * 2, max(options.nSites * 8, 8192))
        } else {
            maxSitesCapacity = numPixels * 2
        }
        let maxSites = max(maxSitesCapacity, actualNSites)
        let solve = solveTargetBpp(
            targetBpp: targetBpp,
            width: target.width,
            height: target.height,
            initSites: actualNSites,
            maxSites: maxSites,
            iters: options.iterations,
            densifyEnabled: options.densifyEnabled,
            densifyStart: options.densifyStart,
            densifyEnd: options.densifyEnd,
            densifyFreq: max(1, options.densifyFreq),
            baseDensify: options.densifyPercentile,
            pruneDuringDensify: options.pruneDuringDensify,
            pruneStart: options.pruneStartIter,
            pruneEnd: pruneEndIter,
            pruneFreq: max(1, options.pruneFreq),
            basePrune: options.prunePercentile,
            maxSplitIndices: 65536
        )
        options.densifyPercentile = solve.densifyPercentile
        options.prunePercentile = solve.prunePercentile
    }

    let plan = planSiteCapacity(options: options, initialSiteCount: actualNSites, numPixels: numPixels)
    printTrainingOverview(device: device, options: options, target: target,
                          actualSites: actualNSites, activeSites: activeSitesEstimate,
                          plan: plan)

    if plan.bufferCapacity != plan.requestedCapacity {
        print("Warning: expanding site buffer capacity from \(plan.requestedCapacity) to \(plan.bufferCapacity) to fit initial sites.")
    }

    let buffers = makeTrainingBuffers(device: device, plan: plan, initialSites: sites)
    let radixSortPairs: RadixSortUInt2? = plan.needsPairs
        ? try! RadixSortUInt2(device: device, library: library, paddedCount: plan.scorePairsCount)
        : nil
    var hilbertResources: HilbertResources? = nil

    var nSitesU = UInt32(actualNSites)
    var candidates = makeCandidateTextures(device: device,
                                           width: target.width,
                                           height: target.height,
                                           downscale: options.candDownscale)
    let renderTextures = makeRenderTextures(device: device, width: target.width, height: target.height)

    var viewer: LiveViewer? = nil
    if options.showViewer {
        DispatchQueue.main.sync {
            viewer = LiveViewer(device: device, library: library,
                                width: target.width, height: target.height)
            viewer?.show()
        }
    }

    let computeSSIMMetric = options.ssimMetric || options.ssimWeight > 0.0
    let psnrBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
    let ssimBuffer = computeSSIMMetric
        ? device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        : nil

    let scale = Float(max(target.width, target.height))
    let invScaleSq = 1.0 / (scale * scale)
    let centroidRadius: Float = 2.0
    let maskedPixels = max(1.0, target.maskSum)
    let deltaNormPerPixel: Float = 1.0 / maskedPixels
    let tauDiffusePasses = 4

    func resolveHilbertBuffers(_ commandBuffer: MTLCommandBuffer)
        -> (order: MTLBuffer?, pos: MTLBuffer?, probes: UInt32, window: UInt32) {
        guard usesVptHilbert else {
            return (nil, nil, 0, 0)
        }
        guard let hilbertEncoder = hilbertEncoder else {
            fatalError("Hilbert probes require HilbertEncoder.")
        }
        if hilbertResources == nil {
            hilbertResources = makeHilbertResources(device: device,
                                                     library: library,
                                                     siteCapacity: plan.bufferCapacity)
        }
        guard var resources = hilbertResources else {
            return (nil, nil, 0, 0)
        }
        let siteCount = Int(nSitesU)
        if !resources.ready || resources.siteCount != siteCount {
            updateHilbertResources(resources: &resources,
                                   encoder: hilbertEncoder,
                                   commandBuffer: commandBuffer,
                                   sitesBuffer: buffers.sitesBuffer,
                                   siteCount: siteCount,
                                   width: target.width,
                                   height: target.height)
        }
        hilbertResources = resources
        return (resources.order, resources.pos,
                options.candHilbertProbes, options.candHilbertWindow)
    }
    let tauDiffuseLambda: Float = 0.05

    let lrPos = options.lrPosBase * options.lrScale
    let lrTau = options.lrTauBase * options.lrScale
    let lrRadius = options.lrRadiusBase * options.lrScale
    let lrColor = options.lrColorBase * options.lrScale
    let lrDir = options.lrDirBase * options.lrScale
    let lrAniso = options.lrAnisoBase * options.lrScale

    let seed: UInt32 = 0

    if let commandBuffer = commandQueue.makeCommandBuffer() {
        initCandidatesEncoder.encodeInit(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                          siteCount: nSitesU, seed: seed,
                                          perPixelMode: options.initPerPixel, in: commandBuffer)
    seedEncoder.encodeSeed(cand0: candidates.cand0A,
                           sitesBuffer: buffers.sitesBuffer, siteCount: nSitesU,
                           candDownscale: UInt32(max(1, options.candDownscale)),
                           in: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    print("Logs | Iter | PSNR | Active | speed | elapsed")
    let startTime = Date()
    var bestPSNR: Float = 0
    var finalPSNR: Float = 0
    var bestSSIM: Float = 0
    var finalSSIM: Float = 0

    // Allow a small number of in-flight command buffers so the CPU can encode the next
    // iteration while the GPU is working, without unbounded queue growth.
    let inflightSemaphore = DispatchSemaphore(value: 3)

    let effectivePruneStartIter: Int
    if options.densifyEnabled && !options.pruneDuringDensify && options.pruneStartIter < options.densifyEnd {
        if options.densifyEnd == Int.max {
            effectivePruneStartIter = Int.max
            print("Note: Pruning disabled during densification (use `--prune-during-densify` to override).")
        } else {
            effectivePruneStartIter = options.densifyEnd
            print("Note: Adjusting prune start from \(options.pruneStartIter) -> \(effectivePruneStartIter) to avoid pruning during densification.")
        }
    } else {
        effectivePruneStartIter = options.pruneStartIter
    }

    let effectivePruneEndIter = options.pruneEndIter ?? (options.iterations - 1)
    var jumpPassIndex: UInt32 = 0

    for iter in 0..<options.iterations {
        if let captureIter = options.traceFrame, iter == captureIter {
            let captureManager = MTLCaptureManager.shared()
            let captureDescriptor = MTLCaptureDescriptor()
            captureDescriptor.captureObject = commandQueue
            captureDescriptor.destination = .gpuTraceDocument

            let outputURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("metal_trace_iter_\(captureIter).gputrace")
            captureDescriptor.outputURL = outputURL

            do {
                try captureManager.startCapture(with: captureDescriptor)
                print("Started Metal trace capture at iteration \(captureIter) -> \(outputURL.path)")
            } catch {
                print("Failed to start Metal trace capture: \(error)")
            }
        }

        inflightSemaphore.wait()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inflightSemaphore.signal()
            continue
        }
        commandBuffer.addCompletedHandler { _ in
            inflightSemaphore.signal()
        }

        let shouldLog = (iter % options.logFreq == 0) || (iter == options.iterations - 1)
        let shouldUpdateViewer = viewer != nil &&
            (iter % max(options.viewerFreq, 1) == 0 || iter == options.iterations - 1)
        let needsLossRender = options.ssimWeight > 0.0

        let candidatePlan = candidateUpdatePlan(iter: iter, options: options,
                                                effectivePruneStartIter: effectivePruneStartIter)
        let desiredSplits = max(0, Int(Float(activeSitesEstimate) * options.densifyPercentile))
        let shouldDensify = options.densifyEnabled &&
            iter >= options.densifyStart &&
            iter <= options.densifyEnd &&
            (iter % options.densifyFreq == 0) &&
            actualNSites < buffers.bufferCapacity &&
            desiredSplits > 0
        let useConcurrentCandGrad = candidatePlan.shouldUpdate &&
            candidatePlan.passes == 1 &&
            !shouldDensify
        let hilbertBuffers = candidatePlan.shouldUpdate
            ? resolveHilbertBuffers(commandBuffer)
            : (order: nil as MTLBuffer?, pos: nil as MTLBuffer?, probes: 0 as UInt32, window: 0 as UInt32)

        if candidatePlan.shouldUpdate {
            packCandidateSites(encoder: candidatePackEncoder, commandBuffer: commandBuffer,
                               sitesBuffer: buffers.sitesBuffer,
                               packedBuffer: buffers.packedCandidatesBuffer,
                               siteCount: nSitesU)
        }

        if candidatePlan.shouldUpdate && !useConcurrentCandGrad {
            updateCandidatesCompact(encoder: compactCandidatesEncoder, commandBuffer: commandBuffer,
                                    candidates: &candidates,
                                    packedSitesBuffer: buffers.packedCandidatesBuffer, siteCount: nSitesU,
                                    width: candidates.cand0A.width, height: candidates.cand0A.height,
                                    targetWidth: target.width, targetHeight: target.height,
                                    candDownscale: options.candDownscale,
                                    invScaleSq: invScaleSq,
                                    radiusScale: options.candRadiusScale,
                                    radiusProbes: options.candRadiusProbes,
                                    injectCount: options.candInjectCount,
                                    hilbertOrder: hilbertBuffers.order,
                                    hilbertPos: hilbertBuffers.pos,
                                    hilbertProbeCount: hilbertBuffers.probes,
                                    hilbertWindow: hilbertBuffers.window,
                                    passes: candidatePlan.passes,
                                    jumpPassIndex: &jumpPassIndex)
        }

        if shouldDensify,
           let massBuf = buffers.massBuffer,
           let energyBuf = buffers.energyBuffer,
           let ewBuf = buffers.errWBuffer,
           let ewxBuf = buffers.errWxBuffer,
           let ewyBuf = buffers.errWyBuffer,
           let ewxxBuf = buffers.errWxxBuffer,
           let ewxyBuf = buffers.errWxyBuffer,
           let ewyyBuf = buffers.errWyyBuffer,
           let pairsBuf = buffers.scorePairsBuffer,
           let splitIdsBuf = buffers.splitIndicesBuffer {
            clearEncoder.encode(buffers: [massBuf, energyBuf, ewBuf, ewxBuf, ewyBuf, ewxxBuf, ewxyBuf, ewyyBuf],
                               count: nSitesU, in: commandBuffer)

            densifyEncoder.encodeStats(
                cand0: candidates.cand0A, cand1: candidates.cand1A,
                targetTexture: target.texture, maskTexture: target.maskTexture,
                sitesBuffer: buffers.sitesBuffer, invScaleSq: invScaleSq, siteCount: nSitesU,
                massBuffer: massBuf, energyBuffer: energyBuf,
                errWBuffer: ewBuf, errWxBuffer: ewxBuf, errWyBuffer: ewyBuf,
                errWxxBuffer: ewxxBuf, errWxyBuffer: ewxyBuf, errWyyBuffer: ewyyBuf,
                in: commandBuffer)

            densifyEncoder.encodeScorePairs(
                sitesBuffer: buffers.sitesBuffer, massBuffer: massBuf, energyBuffer: energyBuf,
                pairsBuffer: pairsBuf, siteCount: nSitesU,
                minMass: 1.0, scoreAlpha: options.densifyScoreAlpha,
                pairCount: UInt32(buffers.scorePairsCount), in: commandBuffer)

            radixSortPairs?.encode(data: pairsBuf, maxKeyExclusive: UInt32.max, in: commandBuffer)

            if desiredSplits > 0 {
                if actualNSites < buffers.bufferCapacity {
                    let available = (buffers.bufferCapacity - actualNSites)
                    let numToSplit = min(desiredSplits, min(available, buffers.maxSplitIndicesCapacity))
                    if numToSplit > 0 {
                        pruneEncoder.encodeWriteSplitIndices(
                            sortedPairsBuffer: pairsBuf, splitIndicesBuffer: splitIdsBuf,
                            numToSplit: UInt32(numToSplit), in: commandBuffer)

                        densifyEncoder.encodeSplit(
                            sitesBuffer: buffers.sitesBuffer, adamBuffer: buffers.adamBuffer,
                            splitIndicesBuffer: splitIdsBuf, numToSplit: UInt32(numToSplit),
                            massBuffer: massBuf,
                            errWBuffer: ewBuf, errWxBuffer: ewxBuf, errWyBuffer: ewyBuf,
                            errWxxBuffer: ewxxBuf, errWxyBuffer: ewxyBuf, errWyyBuffer: ewyyBuf,
                            currentSiteCount: UInt32(actualNSites),
                            targetTexture: target.texture, in: commandBuffer)

                        actualNSites += numToSplit
                        nSitesU = UInt32(actualNSites)
                        activeSitesEstimate += numToSplit
                    }
                }
            }
        }

        let shouldPruneThisIter = options.prunePercentile > 0 &&
            iter >= effectivePruneStartIter &&
            iter < effectivePruneEndIter &&
            iter % options.pruneFreq == 0 &&
            buffers.scorePairsBuffer != nil &&
            buffers.pruneIndicesBuffer != nil

        if shouldPruneThisIter {
            clearEncoder.encode(buffer: buffers.removalDeltaBuffer, count: nSitesU, in: commandBuffer)
        }

        if needsLossRender {
            renderEncoder.encodeRender(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                       output: renderTextures.render,
                                       sitesBuffer: buffers.sitesBuffer, invScaleSq: invScaleSq,
                                       siteCount: nSitesU, in: commandBuffer)
        }

            if useConcurrentCandGrad, let encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent) {
            encoder.label = "Candidates + Gradients (Concurrent)"

            let step = packJumpStep(jumpPassIndex,
                                    width: candidates.cand0A.width,
                                    height: candidates.cand0A.height)
            let stepHigh = jumpPassIndex >> 16
            compactCandidatesEncoder.encodeUpdate(
                cand0In: candidates.cand0A, cand1In: candidates.cand1A,
                cand0Out: candidates.cand0B, cand1Out: candidates.cand1B,
                packedSitesBuffer: buffers.packedCandidatesBuffer, siteCount: nSitesU,
                step: step, stepHigh: stepHigh, invScaleSq: invScaleSq,
                radiusScale: options.candRadiusScale,
                radiusProbes: options.candRadiusProbes,
                injectCount: options.candInjectCount,
                candDownscale: UInt32(max(1, options.candDownscale)),
                targetWidth: UInt32(max(1, target.width)),
                targetHeight: UInt32(max(1, target.height)),
                hilbertOrder: hilbertBuffers.order,
                hilbertPos: hilbertBuffers.pos,
                hilbertProbeCount: hilbertBuffers.probes,
                hilbertWindow: hilbertBuffers.window,
                in: encoder)
            tiledGradientEncoder.encodeGradients(
                cand0: candidates.cand0A, cand1: candidates.cand1A,
                targetTexture: target.texture, maskTexture: target.maskTexture,
                renderTexture: renderTextures.render, ssimWeight: options.ssimWeight,
                gradBuffers: buffers.gradBuffers, sitesBuffer: buffers.sitesBuffer,
                invScaleSq: invScaleSq, siteCount: nSitesU,
                removalDeltaBuffer: buffers.removalDeltaBuffer, computeRemoval: shouldPruneThisIter,
                in: encoder)
            encoder.endEncoding()

            jumpPassIndex &+= 1
            swap(&candidates.cand0A, &candidates.cand0B)
            swap(&candidates.cand1A, &candidates.cand1B)
        } else {
            if useConcurrentCandGrad {
                updateCandidatesCompact(encoder: compactCandidatesEncoder, commandBuffer: commandBuffer,
                                        candidates: &candidates,
                                        packedSitesBuffer: buffers.packedCandidatesBuffer, siteCount: nSitesU,
                                        width: candidates.cand0A.width, height: candidates.cand0A.height,
                                        targetWidth: target.width, targetHeight: target.height,
                                        candDownscale: options.candDownscale,
                                        invScaleSq: invScaleSq,
                                        radiusScale: options.candRadiusScale,
                                        radiusProbes: options.candRadiusProbes,
                                        injectCount: options.candInjectCount,
                                        hilbertOrder: hilbertBuffers.order,
                                        hilbertPos: hilbertBuffers.pos,
                                        hilbertProbeCount: hilbertBuffers.probes,
                                        hilbertWindow: hilbertBuffers.window,
                                        passes: candidatePlan.passes,
                                        jumpPassIndex: &jumpPassIndex)
            }
            tiledGradientEncoder.encodeGradients(
                cand0: candidates.cand0A, cand1: candidates.cand1A,
                targetTexture: target.texture, maskTexture: target.maskTexture,
                renderTexture: renderTextures.render, ssimWeight: options.ssimWeight,
                gradBuffers: buffers.gradBuffers, sitesBuffer: buffers.sitesBuffer,
                invScaleSq: invScaleSq, siteCount: nSitesU,
                removalDeltaBuffer: buffers.removalDeltaBuffer, computeRemoval: shouldPruneThisIter,
                in: commandBuffer)
        }

        if tauDiffusePasses > 0 && tauDiffuseLambda > 0.0 {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                let bytes = MemoryLayout<Float>.stride * buffers.bufferCapacity
                blit.copy(from: buffers.gradBuffers[2], sourceOffset: 0,
                          to: buffers.tauGradRawBuffer, destinationOffset: 0,
                          size: bytes)
                blit.endEncoding()
            }

            let blend = Float(iter) / Float(options.iterations)
            let lambda = tauDiffuseLambda * (0.1 + 0.9 * blend)
            var currentIn: MTLBuffer = buffers.gradBuffers[2]
            var currentOut: MTLBuffer = buffers.tauGradTmpBuffer
            for _ in 0..<tauDiffusePasses {
                tauDiffusionEncoder.encode(
                    cand0: candidates.cand0A, cand1: candidates.cand1A,
                    sitesBuffer: buffers.sitesBuffer,
                    gradRaw: buffers.tauGradRawBuffer,
                    gradIn: currentIn, gradOut: currentOut,
                    siteCount: nSitesU,
                    lambda: lambda,
                    candDownscale: UInt32(options.candDownscale),
                    in: commandBuffer)
                swap(&currentIn, &currentOut)
            }
        }

        let adamParams = AdamParams(
            lrPos: lrPos, lrTau: lrTau, lrRadius: lrRadius, lrColor: lrColor,
            lrDir: lrDir,
            lrAniso: lrAniso,
            beta1: options.beta1, beta2: options.beta2, eps: options.eps,
            t: UInt32(iter + 1), width: UInt32(target.width), height: UInt32(target.height)
        )
        adamEncoder.encode(sitesBuffer: buffers.sitesBuffer, adamBuffer: buffers.adamBuffer,
                          gradBuffers: buffers.gradBuffers, params: adamParams,
                          siteCount: actualNSites, in: commandBuffer)

        if options.prunePercentile > 0,
           iter >= effectivePruneStartIter && iter < effectivePruneEndIter && iter % options.pruneFreq == 0,
           let pairsBuf = buffers.scorePairsBuffer,
           let pruneIdsBuf = buffers.pruneIndicesBuffer {
            pruneEncoder.encodeScorePairs(
                sitesBuffer: buffers.sitesBuffer, removalDeltaBuffer: buffers.removalDeltaBuffer,
                pairsBuffer: pairsBuf,
                siteCount: nSitesU, deltaNorm: deltaNormPerPixel,
                pairCount: UInt32(buffers.scorePairsCount), in: commandBuffer)

            radixSortPairs?.encode(data: pairsBuf, maxKeyExclusive: UInt32.max, in: commandBuffer)

            let desiredPrunes = max(0, Int(Float(activeSitesEstimate) * options.prunePercentile))
            let numToPrune = min(desiredPrunes, buffers.maxSplitIndicesCapacity)
            if numToPrune > 0 {
                pruneEncoder.encodeWriteSplitIndices(
                    sortedPairsBuffer: pairsBuf, splitIndicesBuffer: pruneIdsBuf,
                    numToSplit: UInt32(numToPrune), in: commandBuffer)
                pruneEncoder.encodePruneSites(
                    sitesBuffer: buffers.sitesBuffer, indicesBuffer: pruneIdsBuf,
                    count: UInt32(numToPrune), in: commandBuffer)
                activeSitesEstimate = max(0, activeSitesEstimate - numToPrune)
            }
        }

        if shouldLog || shouldUpdateViewer {
            renderEncoder.encodeRender(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                       output: renderTextures.render,
                                       sitesBuffer: buffers.sitesBuffer, invScaleSq: invScaleSq,
                                       siteCount: nSitesU, in: commandBuffer)
            if shouldUpdateViewer {
                renderEncoder.encodeColoring(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                             output: renderTextures.voronoi,
                                             sitesBuffer: buffers.sitesBuffer, invScaleSq: invScaleSq,
                                             siteCount: nSitesU, in: commandBuffer)
            }
            if shouldLog {
                psnrEncoder.encode(rendered: renderTextures.render, target: target.texture,
                                  mask: target.maskTexture,
                                  mseBuffer: psnrBuffer, in: commandBuffer)
                if computeSSIMMetric, let ssimBuffer = ssimBuffer {
                    ssimEncoder.encode(rendered: renderTextures.render, target: target.texture,
                                       mask: target.maskTexture,
                                       ssimBuffer: ssimBuffer, in: commandBuffer)
                }
            }
        }

        commandBuffer.commit()

        // Only block the CPU when we need to read back results (logging), update the live
        // viewer (uses a separate command queue), or stop a Metal capture.
        let shouldStopCapture = (options.traceFrame == iter)
        if shouldLog || shouldUpdateViewer || shouldStopCapture {
            commandBuffer.waitUntilCompleted()
        }

        if let captureIter = options.traceFrame, iter == captureIter {
            let captureManager = MTLCaptureManager.shared()
            captureManager.stopCapture()
            print("Stopped Metal trace capture at iteration \(captureIter)")
        }

        if shouldLog {
            let mse = psnrBuffer.contents().load(as: Float.self) / Float(max(1.0, target.maskSum) * 3.0)
            let psnr = mse > 0 ? 20.0 * log10(1.0 / sqrt(mse)) : 100.0
            var ssim: Float = 0.0
            if computeSSIMMetric, let ssimBuffer = ssimBuffer {
                ssim = ssimBuffer.contents().load(as: Float.self) / Float(max(1.0, target.maskSum))
            }

            let activeSites = countActiveSites(sitesBuffer: buffers.sitesBuffer, siteCount: actualNSites)
            let elapsed = Date().timeIntervalSince(startTime)
            let itsPerSec = Float(iter + 1) / Float(elapsed)

            if computeSSIMMetric {
                print(String(format: "Iter %4d | PSNR: %.2f dB | SSIM: %.4f | Active: %5d/%d | %.1f it/s | %.1fs",
                             iter, psnr, ssim, activeSites, actualNSites, itsPerSec, elapsed))
            } else {
                print(String(format: "Iter %4d | PSNR: %.2f dB | Active: %5d/%d | %.1f it/s | %.1fs",
                             iter, psnr, activeSites, actualNSites, itsPerSec, elapsed))
            }

            finalPSNR = psnr
            if psnr > bestPSNR {
                bestPSNR = psnr
            }
            if computeSSIMMetric {
                finalSSIM = ssim
                if ssim > bestSSIM {
                    bestSSIM = ssim
                }
            }
        }

        if shouldUpdateViewer, let viewer = viewer {
            viewer.update(image: renderTextures.render, ids: renderTextures.voronoi)
        }
    }

    let export = exportActiveSites(sitesBuffer: buffers.sitesBuffer, siteCount: actualNSites)
    let trainTime = Date().timeIntervalSince(startTime)
    let avgSpeed = Float(options.iterations) / Float(max(trainTime, 1e-6))

    print(String(format: "\nFinal PSNR: %.2f dB (best: %.2f dB)", finalPSNR, bestPSNR))
    if computeSSIMMetric {
        print(String(format: "Final SSIM: %.4f (best: %.4f)", finalSSIM, bestSSIM))
    }

    let preRefreshPSNR = finalPSNR
    let preRefreshSSIM = finalSSIM
    print("Saving result...")

    let finalCandidatePasses = max(1, options.exportCandPasses ?? options.candUpdatePasses)
    if finalCandidatePasses > 0 {
        print("Refreshing candidates for final render...")
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            let hilbertBuffers = usesVptHilbert
                ? resolveHilbertBuffers(commandBuffer)
                : (order: nil as MTLBuffer?, pos: nil as MTLBuffer?, probes: 0 as UInt32, window: 0 as UInt32)
            packCandidateSites(encoder: candidatePackEncoder, commandBuffer: commandBuffer,
                               sitesBuffer: buffers.sitesBuffer,
                               packedBuffer: buffers.packedCandidatesBuffer,
                               siteCount: nSitesU)
            updateCandidatesCompact(encoder: compactCandidatesEncoder, commandBuffer: commandBuffer,
                                    candidates: &candidates,
                                    packedSitesBuffer: buffers.packedCandidatesBuffer, siteCount: nSitesU,
                                    width: candidates.cand0A.width, height: candidates.cand0A.height,
                                    targetWidth: target.width, targetHeight: target.height,
                                    candDownscale: options.candDownscale,
                                    invScaleSq: invScaleSq,
                                    radiusScale: options.candRadiusScale,
                                    radiusProbes: options.candRadiusProbes,
                                    injectCount: options.candInjectCount,
                                    hilbertOrder: hilbertBuffers.order,
                                    hilbertPos: hilbertBuffers.pos,
                                    hilbertProbeCount: hilbertBuffers.probes,
                                    hilbertWindow: hilbertBuffers.window,
                                    passes: finalCandidatePasses,
                                    jumpPassIndex: &jumpPassIndex)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }

    if let commandBuffer = commandQueue.makeCommandBuffer() {
        renderEncoder.encodeRender(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                   output: renderTextures.render,
                                   sitesBuffer: buffers.sitesBuffer, invScaleSq: invScaleSq,
                                   siteCount: nSitesU, in: commandBuffer)

        psnrEncoder.encode(rendered: renderTextures.render, target: target.texture,
                           mask: target.maskTexture,
                           mseBuffer: psnrBuffer, in: commandBuffer)
        if computeSSIMMetric, let ssimBuffer = ssimBuffer {
            ssimEncoder.encode(rendered: renderTextures.render, target: target.texture,
                               mask: target.maskTexture,
                               ssimBuffer: ssimBuffer, in: commandBuffer)
        }

        renderEncoder.encodeColoring(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                     output: renderTextures.voronoi,
                                     sitesBuffer: buffers.sitesBuffer, invScaleSq: invScaleSq,
                                     siteCount: nSitesU, in: commandBuffer)

        renderEncoder.encodeCentroids(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                      output: renderTextures.centroids,
                                      sitesBuffer: buffers.sitesBuffer, dotRadius: centroidRadius,
                                      siteCount: nSitesU, in: commandBuffer)

        let tauParams = TauHeatmapParams(
            minTau: export.minTau,
            meanTau: export.avgTau,
            maxTau: export.maxTau,
            dotRadius: 2.0
        )
        renderEncoder.encodeTauHeatmap(cand0: candidates.cand0A, cand1: candidates.cand1A,
                                       output: renderTextures.tauHeatmap,
                                       sitesBuffer: buffers.sitesBuffer,
                                       params: tauParams,
                                       siteCount: nSitesU, in: commandBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    let postMSE = psnrBuffer.contents().load(as: Float.self) / Float(max(1.0, target.maskSum) * 3.0)
    let postRefreshPSNR = postMSE > 0 ? 20.0 * log10(1.0 / sqrt(postMSE)) : 100.0
    print(String(format: "Final PSNR (pre-refresh): %.2f dB | post-refresh: %.2f dB",
                 preRefreshPSNR, postRefreshPSNR))
    if computeSSIMMetric, let ssimBuffer = ssimBuffer {
        let postRefreshSSIM = ssimBuffer.contents().load(as: Float.self) / Float(max(1.0, target.maskSum))
        print(String(format: "Final SSIM (pre-refresh): %.4f | post-refresh: %.4f",
                     preRefreshSSIM, postRefreshSSIM))
    }

    ensureDirectory(options.outputDir)
    let paths = makeOutputPaths(outputDir: options.outputDir, targetPath: options.targetPath)

    saveTexture(renderTextures.render, path: paths.imagePath)
    print("Saved: \(paths.imagePath)")

    saveTexture(renderTextures.voronoi, path: paths.cellPath)
    print("Saved: \(paths.cellPath)")

    saveTexture(renderTextures.centroids, path: paths.centroidsPath)
    print("Saved: \(paths.centroidsPath)")

    saveTexture(renderTextures.tauHeatmap, path: paths.tauHeatmapPath)
    print("Saved: \(paths.tauHeatmapPath)")

    let stats = TrainingStats(
        initialSites: actualNSites,
        finalSites: export.activeCount,
        finalPSNR: finalPSNR,
        bestPSNR: bestPSNR,
        finalSSIM: computeSSIMMetric ? finalSSIM : nil,
        bestSSIM: computeSSIMMetric ? bestSSIM : nil,
        totalTime: trainTime,
        avgSpeed: avgSpeed,
        minTau: export.minTau,
        maxTau: export.maxTau,
        avgTau: export.avgTau,
        minRadius: export.minRadius,
        maxRadius: export.maxRadius,
        avgRadius: export.avgRadius
    )

    let timestamp = makeTimestamp()
    let report = buildTextReport(options: options,
                                 width: target.width,
                                 height: target.height,
                                 effectivePruneEndIter: effectivePruneEndIter,
                                 paths: paths,
                                 timestamp: timestamp,
                                 stats: stats)

    if let reportData = report.data(using: .utf8) {
        try? reportData.write(to: URL(fileURLWithPath: paths.reportPath))
        print("Saved: \(paths.reportPath)")
    }

    let mdReport = buildMarkdownReport(options: options,
                                       width: target.width,
                                       height: target.height,
                                       timestamp: timestamp,
                                       stats: stats)
    if let mdData = mdReport.data(using: .utf8) {
        try? mdData.write(to: URL(fileURLWithPath: paths.mdPath))
        print("Saved: \(paths.mdPath)")
    }

    writeSitesTXT(sites: export.sites, width: target.width, height: target.height, path: paths.sitesPath)
    print("Saved: \(paths.sitesPath)")

    let neighbors: [[Int]]
    if options.exportNeighbors {
        let sitesPtrFinal = buffers.sitesBuffer.contents().bindMemory(to: VoronoiSite.self, capacity: actualNSites)
        neighbors = computeSiteNeighbors(
            cand0: candidates.cand0A,
            cand1: candidates.cand1A,
            sites: sitesPtrFinal,
            siteCount: actualNSites,
            activeMap: export.activeMap,
            activeCount: export.activeCount,
            invScaleSq: invScaleSq)
        print("Neighbors computed (unique edges: ~\(neighbors.reduce(0) { $0 + $1.count } / 2)).")
    } else {
        neighbors = Array(repeating: [], count: export.sites.count)
    }

    writeSitesJSON(sites: export.sites, width: target.width, height: target.height,
                   neighbors: neighbors, path: paths.jsonPath)
    print("Saved: \(paths.jsonPath)")

    var packedPSNRValue: Float? = nil
    if options.packedPSNR {
        let sitesPtr = buffers.sitesBuffer.contents().bindMemory(to: VoronoiSite.self, capacity: actualNSites)
        let fullSites = Array(UnsafeBufferPointer(start: sitesPtr, count: actualNSites))
        let packed = makePackedInferenceSites(fullSites, width: target.width, height: target.height)
        if let packedResult = renderPackedWithCandidates(cand0: candidates.cand0A,
                                                         cand1: candidates.cand1A,
                                                         packedSites: packed.sites,
                                                         quant: packed.quant,
                                                         device: device,
                                                         commandQueue: commandQueue,
                                                         library: library,
                                                         outputVoronoi: false) {
            saveTexture(packedResult.render, path: paths.packedPath)
            print("Saved: \(paths.packedPath)")

            if let commandBuffer = commandQueue.makeCommandBuffer() {
                psnrEncoder.encode(rendered: packedResult.render, target: target.texture,
                                   mask: target.maskTexture,
                                   mseBuffer: psnrBuffer, in: commandBuffer)
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
            }
            let mse = psnrBuffer.contents().load(as: Float.self) / Float(max(1.0, target.maskSum) * 3.0)
            packedPSNRValue = mse > 0 ? 20.0 * log10(1.0 / sqrt(mse)) : 100.0
            if let packedPSNRValue = packedPSNRValue {
                print(String(format: "Packed inference PSNR: %.2f dB", packedPSNRValue))
            }
        }
    }

    let sitesPruned = actualNSites - export.activeCount
    let prunedPercent = actualNSites > 0 ? Float(sitesPruned) / Float(actualNSites) * 100.0 : 0.0
    let totalWallTime = Date().timeIntervalSince(startTime)

    print("\n" + "=".padding(toLength: 50, withPad: "=", startingAt: 0))
    print("SUMMARY:")
    print("  Sites: \(actualNSites) -> \(export.activeCount) (\(String(format: "%.1f%%", prunedPercent)) pruned)")
    print("  Final PSNR: \(String(format: "%.2f dB", finalPSNR)) (best: \(String(format: "%.2f dB", bestPSNR)))")
    if computeSSIMMetric {
        print("  Final SSIM: \(String(format: "%.4f", finalSSIM)) (best: \(String(format: "%.4f", bestSSIM)))")
    }
    if let packedPSNRValue = packedPSNRValue {
        print("  Packed PSNR: \(String(format: "%.2f dB", packedPSNRValue))")
    }
    print("  Speed: \(String(format: "%.1f it/s", avgSpeed))")
    print("  Time: \(String(format: "%.2f s", trainTime)) train | \(String(format: "%.2f s", totalWallTime)) total")
    print("  Compression: \(String(format: "%.2fx", Float(target.width * target.height) / Float(export.activeCount)))")
    print("=".padding(toLength: 50, withPad: "=", startingAt: 0))
}
