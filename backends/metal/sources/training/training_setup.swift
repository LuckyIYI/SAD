import Foundation
import Metal
import CoreGraphics

struct TargetImage {
    let width: Int
    let height: Int
    let texture: MTLTexture
    let floatData: [Float]
    let maskTexture: MTLTexture
    let maskSum: Float
}

private func createMaskTexture(width: Int, height: Int, device: MTLDevice, fill: Float) -> MTLTexture? {
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Float,
        width: width,
        height: height,
        mipmapped: false)
    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
        return nil
    }

    var floatData = [Float](repeating: 0, count: width * height * 4)
    for i in 0..<(width * height) {
        let base = i * 4
        floatData[base] = fill
        floatData[base + 1] = fill
        floatData[base + 2] = fill
        floatData[base + 3] = 1.0
    }

    let region = MTLRegionMake2D(0, 0, width, height)
    texture.replace(region: region, mipmapLevel: 0, withBytes: floatData,
                    bytesPerRow: width * 4 * MemoryLayout<Float>.stride)
    return texture
}

private func loadMaskTexture(path: String, targetWidth: Int, targetHeight: Int,
                             maxDimension: Int, device: MTLDevice) -> (MTLTexture, Float)? {
    guard let (cgImage, width, height) = loadImage(path: path, maxDimension: maxDimension, device: device),
          let texture = createTextureFromImage(cgImage, device: device) else {
        print("Failed to load mask image: \(path)")
        return nil
    }

    if width != targetWidth || height != targetHeight {
        print("Mask dimensions \(width)x\(height) do not match target \(targetWidth)x\(targetHeight)")
        return nil
    }

    let bytesPerRow = width * 4 * MemoryLayout<Float>.stride
    var floatData = [Float](repeating: 0, count: width * height * 4)
    let region = MTLRegionMake2D(0, 0, width, height)
    texture.getBytes(&floatData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

    var maskSum: Float = 0.0
    for i in 0..<(width * height) {
        if floatData[i * 4] > 0.0 {
            maskSum += 1.0
        }
    }

    return (texture, maskSum)
}

func loadTargetImage(path: String, maskPath: String?, maxDimension: Int, device: MTLDevice) -> TargetImage? {
    guard let (cgImage, width, height) = loadImage(path: path, maxDimension: maxDimension, device: device),
          let texture = createTextureFromImage(cgImage, device: device) else {
        return nil
    }

    let bytesPerRow = width * 4 * MemoryLayout<Float>.stride
    var floatData = [Float](repeating: 0, count: width * height * 4)
    let region = MTLRegionMake2D(0, 0, width, height)
    texture.getBytes(&floatData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

    let maskTexture: MTLTexture
    let maskSum: Float
    if let maskPath = maskPath {
        guard let (loadedMask, maskTotal) = loadMaskTexture(
            path: maskPath,
            targetWidth: width,
            targetHeight: height,
            maxDimension: maxDimension,
            device: device
        ) else {
            return nil
        }
        maskTexture = loadedMask
        maskSum = maskTotal
    } else {
        guard let defaultMask = createMaskTexture(width: width, height: height, device: device, fill: 1.0) else {
            return nil
        }
        maskTexture = defaultMask
        maskSum = Float(width * height)
    }

    return TargetImage(width: width, height: height, texture: texture, floatData: floatData,
                       maskTexture: maskTexture, maskSum: maskSum)
}

struct SiteCapacityPlan {
    let densifyEnabled: Bool
    let needsPairs: Bool
    let needsPrune: Bool
    let maxSitesCapacity: Int
    let requestedCapacity: Int
    let bufferCapacity: Int
    let scorePairsCount: Int
    let maxSplitIndicesCapacity: Int
}

func planSiteCapacity(options: TrainingOptions, initialSiteCount: Int, numPixels: Int) -> SiteCapacityPlan {
    let needsPairs = options.densifyEnabled || options.prunePercentile > 0.0
    let needsPrune = options.prunePercentile > 0.0

    let maxSitesCapacity: Int
    if options.maxSites > 0 {
        maxSitesCapacity = options.maxSites
    } else if options.densifyEnabled {
        maxSitesCapacity = min(numPixels * 2, max(options.nSites * 8, 8192))
    } else {
        maxSitesCapacity = numPixels * 2
    }

    let requestedCapacity = options.densifyEnabled ? maxSitesCapacity : initialSiteCount
    let bufferCapacity = max(initialSiteCount, requestedCapacity)
    let scorePairsCount = needsPairs ? max(1, bufferCapacity) : 0
    let maxSplitIndicesCapacity = 65536

    return SiteCapacityPlan(
        densifyEnabled: options.densifyEnabled,
        needsPairs: needsPairs,
        needsPrune: needsPrune,
        maxSitesCapacity: maxSitesCapacity,
        requestedCapacity: requestedCapacity,
        bufferCapacity: bufferCapacity,
        scorePairsCount: scorePairsCount,
        maxSplitIndicesCapacity: maxSplitIndicesCapacity
    )
}

struct TrainingBuffers {
    let sitesBuffer: MTLBuffer
    let packedCandidatesBuffer: MTLBuffer
    let adamBuffer: MTLBuffer
    let tauGradRawBuffer: MTLBuffer
    let tauGradTmpBuffer: MTLBuffer
    let gradBuffers: [MTLBuffer]
    let removalDeltaBuffer: MTLBuffer
    let massBuffer: MTLBuffer?
    let energyBuffer: MTLBuffer?
    let errWBuffer: MTLBuffer?
    let errWxBuffer: MTLBuffer?
    let errWyBuffer: MTLBuffer?
    let errWxxBuffer: MTLBuffer?
    let errWxyBuffer: MTLBuffer?
    let errWyyBuffer: MTLBuffer?
    let scorePairsBuffer: MTLBuffer?
    let splitIndicesBuffer: MTLBuffer?
    let pruneIndicesBuffer: MTLBuffer?
    let bufferCapacity: Int
    let scorePairsCount: Int
    let maxSplitIndicesCapacity: Int
}

func makeTrainingBuffers(device: MTLDevice,
                         plan: SiteCapacityPlan,
                         initialSites: [VoronoiSite]) -> TrainingBuffers {
    let bufferCapacity = plan.bufferCapacity

    let sitesBuffer = device.makeBuffer(
        length: MemoryLayout<VoronoiSite>.stride * bufferCapacity,
        options: .storageModeShared
    )!
    let packedCandidatesBuffer = device.makeBuffer(
        length: MemoryLayout<PackedCandidateSite>.stride * bufferCapacity,
        options: .storageModeShared
    )!
    let adamBuffer = device.makeBuffer(
        length: MemoryLayout<AdamState>.stride * bufferCapacity,
        options: .storageModeShared
    )!
    let tauGradRawBuffer = device.makeBuffer(
        length: MemoryLayout<Float>.stride * bufferCapacity,
        options: .storageModeShared
    )!
    let tauGradTmpBuffer = device.makeBuffer(
        length: MemoryLayout<Float>.stride * bufferCapacity,
        options: .storageModeShared
    )!
    let removalDeltaBuffer = device.makeBuffer(
        length: MemoryLayout<Float>.stride * bufferCapacity,
        options: .storageModeShared
    )!

    let gradBuffers = (0..<10).map { _ in
        device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                          options: .storageModeShared)!
    }

    for buffer in gradBuffers {
        memset(buffer.contents(), 0, MemoryLayout<Float>.stride * bufferCapacity)
    }
    memset(removalDeltaBuffer.contents(), 0, MemoryLayout<Float>.stride * bufferCapacity)

    let sitesPtr = sitesBuffer.contents().bindMemory(to: VoronoiSite.self, capacity: bufferCapacity)
    for i in 0..<initialSites.count {
        sitesPtr[i] = initialSites[i]
    }

    let adamPtr = adamBuffer.contents().bindMemory(to: AdamState.self, capacity: bufferCapacity)
    for i in 0..<bufferCapacity {
        adamPtr[i] = AdamState()
    }

    var massBuffer: MTLBuffer? = nil
    var energyBuffer: MTLBuffer? = nil
    var errWBuffer: MTLBuffer? = nil
    var errWxBuffer: MTLBuffer? = nil
    var errWyBuffer: MTLBuffer? = nil
    var errWxxBuffer: MTLBuffer? = nil
    var errWxyBuffer: MTLBuffer? = nil
    var errWyyBuffer: MTLBuffer? = nil
    var scorePairsBuffer: MTLBuffer? = nil
    var splitIndicesBuffer: MTLBuffer? = nil
    var pruneIndicesBuffer: MTLBuffer? = nil

    if plan.needsPairs {
        scorePairsBuffer = device.makeBuffer(
            length: MemoryLayout<SIMD2<UInt32>>.stride * plan.scorePairsCount,
            options: .storageModeShared
        )
    }

    if plan.needsPrune {
        pruneIndicesBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride * plan.maxSplitIndicesCapacity,
            options: .storageModeShared
        )
    }

    if plan.densifyEnabled {
        massBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                                       options: .storageModeShared)
        energyBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                                         options: .storageModeShared)
        errWBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                                       options: .storageModeShared)
        errWxBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                                        options: .storageModeShared)
        errWyBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                                        options: .storageModeShared)
        errWxxBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                                         options: .storageModeShared)
        errWxyBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                                         options: .storageModeShared)
        errWyyBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferCapacity,
                                         options: .storageModeShared)
        splitIndicesBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride * plan.maxSplitIndicesCapacity,
            options: .storageModeShared
        )
    }

    return TrainingBuffers(
        sitesBuffer: sitesBuffer,
        packedCandidatesBuffer: packedCandidatesBuffer,
        adamBuffer: adamBuffer,
        tauGradRawBuffer: tauGradRawBuffer,
        tauGradTmpBuffer: tauGradTmpBuffer,
        gradBuffers: gradBuffers,
        removalDeltaBuffer: removalDeltaBuffer,
        massBuffer: massBuffer,
        energyBuffer: energyBuffer,
        errWBuffer: errWBuffer,
        errWxBuffer: errWxBuffer,
        errWyBuffer: errWyBuffer,
        errWxxBuffer: errWxxBuffer,
        errWxyBuffer: errWxyBuffer,
        errWyyBuffer: errWyyBuffer,
        scorePairsBuffer: scorePairsBuffer,
        splitIndicesBuffer: splitIndicesBuffer,
        pruneIndicesBuffer: pruneIndicesBuffer,
        bufferCapacity: bufferCapacity,
        scorePairsCount: plan.scorePairsCount,
        maxSplitIndicesCapacity: plan.maxSplitIndicesCapacity
    )
}

struct RenderTextures {
    let render: MTLTexture
    let voronoi: MTLTexture
    let tauHeatmap: MTLTexture
}

func makeRenderTextures(device: MTLDevice, width: Int, height: Int) -> RenderTextures {
    let renderDesc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Float,
        width: width,
        height: height,
        mipmapped: false)
    renderDesc.usage = [.shaderRead, .shaderWrite]
    renderDesc.storageMode = .shared

    return RenderTextures(
        render: device.makeTexture(descriptor: renderDesc)!,
        voronoi: device.makeTexture(descriptor: renderDesc)!,
        tauHeatmap: device.makeTexture(descriptor: renderDesc)!
    )
}

struct BppSolveResult {
    let densifyPercentile: Float
    let prunePercentile: Float
    let finalSites: Int
    let achievedBpp: Float
}

func simulateFinalSites(initSites: Int,
                        maxSites: Int,
                        iters: Int,
                        densifyEnabled: Bool,
                        densifyStart: Int,
                        densifyEnd: Int,
                        densifyFreq: Int,
                        densifyPercentile: Float,
                        pruneDuringDensify: Bool,
                        pruneStart: Int,
                        pruneEnd: Int,
                        pruneFreq: Int,
                        prunePercentile: Float,
                        maxSplitIndices: Int) -> Int {
    var actualSites = initSites
    var activeEstimate = initSites
    let maxSitesClamped = max(maxSites, initSites)

    var effectivePruneStart = pruneStart
    if densifyEnabled && !pruneDuringDensify && pruneStart < densifyEnd {
        effectivePruneStart = densifyEnd
    }

    if iters <= 0 {
        return activeEstimate
    }

    for it in 0..<iters {
        if densifyEnabled &&
            densifyPercentile > 0.0 &&
            it >= densifyStart &&
            it <= densifyEnd &&
            it % max(1, densifyFreq) == 0 &&
            actualSites < maxSitesClamped {
            let desired = Int(Float(activeEstimate) * densifyPercentile)
            let available = maxSitesClamped - actualSites
            let numToSplit = min(desired, min(available, maxSplitIndices))
            if numToSplit > 0 {
                actualSites += numToSplit
                activeEstimate += numToSplit
            }
        }

        if prunePercentile > 0.0 &&
            it >= effectivePruneStart &&
            it < pruneEnd &&
            it % max(1, pruneFreq) == 0 {
            let desired = Int(Float(activeEstimate) * prunePercentile)
            let numToPrune = min(desired, maxSplitIndices)
            if numToPrune > 0 {
                activeEstimate = max(0, activeEstimate - numToPrune)
            }
        }
    }

    return activeEstimate
}

func solveTargetBpp(targetBpp: Float,
                    width: Int,
                    height: Int,
                    initSites: Int,
                    maxSites: Int,
                    iters: Int,
                    densifyEnabled: Bool,
                    densifyStart: Int,
                    densifyEnd: Int,
                    densifyFreq: Int,
                    baseDensify: Float,
                    pruneDuringDensify: Bool,
                    pruneStart: Int,
                    pruneEnd: Int,
                    pruneFreq: Int,
                    basePrune: Float,
                    maxSplitIndices: Int) -> BppSolveResult {
    let bitsPerSite: Float = 16.0 * 8.0
    let targetSites = max(1, Int((targetBpp * Float(width * height) / bitsPerSite).rounded()))
    let maxBase = max(baseDensify, basePrune)
    if maxBase <= 0.0 {
        let finalSites = simulateFinalSites(
            initSites: initSites,
            maxSites: maxSites,
            iters: iters,
            densifyEnabled: densifyEnabled,
            densifyStart: densifyStart,
            densifyEnd: densifyEnd,
            densifyFreq: densifyFreq,
            densifyPercentile: 0.0,
            pruneDuringDensify: pruneDuringDensify,
            pruneStart: pruneStart,
            pruneEnd: pruneEnd,
            pruneFreq: pruneFreq,
            prunePercentile: 0.0,
            maxSplitIndices: maxSplitIndices
        )
        let achievedBpp = Float(finalSites) * bitsPerSite / Float(width * height)
        return BppSolveResult(densifyPercentile: 0.0,
                              prunePercentile: 0.0,
                              finalSites: finalSites,
                              achievedBpp: achievedBpp)
    }

    let maxPct: Float = 0.95
    var sMax = maxPct / maxBase
    if sMax > 50.0 {
        sMax = 50.0
    }

    func evalSites(scale: Float) -> Int {
        let densify = densifyEnabled ? min(maxPct, baseDensify * scale) : 0.0
        let prune = min(maxPct, basePrune * scale)
        return simulateFinalSites(
            initSites: initSites,
            maxSites: maxSites,
            iters: iters,
            densifyEnabled: densifyEnabled,
            densifyStart: densifyStart,
            densifyEnd: densifyEnd,
            densifyFreq: densifyFreq,
            densifyPercentile: densify,
            pruneDuringDensify: pruneDuringDensify,
            pruneStart: pruneStart,
            pruneEnd: pruneEnd,
            pruneFreq: pruneFreq,
            prunePercentile: prune,
            maxSplitIndices: maxSplitIndices
        )
    }

    var bestScale: Float = 0.0
    var bestSites = evalSites(scale: 0.0)
    var bestErr = abs(bestSites - targetSites)
    let samples = 80
    for i in 0...samples {
        let s = sMax * Float(i) / Float(samples)
        let sites = evalSites(scale: s)
        let err = abs(sites - targetSites)
        if err < bestErr {
            bestErr = err
            bestScale = s
            bestSites = sites
        }
    }

    var step = sMax / Float(samples)
    for _ in 0..<20 {
        var improved = false
        let s0 = bestScale - step
        let s1 = bestScale + step
        if s0 >= 0.0 {
            let sites = evalSites(scale: s0)
            let err = abs(sites - targetSites)
            if err < bestErr {
                bestErr = err
                bestScale = s0
                bestSites = sites
                improved = true
            }
        }
        if s1 <= sMax {
            let sites = evalSites(scale: s1)
            let err = abs(sites - targetSites)
            if err < bestErr {
                bestErr = err
                bestScale = s1
                bestSites = sites
                improved = true
            }
        }
        if !improved {
            step *= 0.5
        }
    }

    let densify = densifyEnabled ? min(maxPct, baseDensify * bestScale) : 0.0
    let prune = min(maxPct, basePrune * bestScale)
    let achievedBpp = Float(bestSites) * bitsPerSite / Float(width * height)
    return BppSolveResult(densifyPercentile: densify,
                          prunePercentile: prune,
                          finalSites: bestSites,
                          achievedBpp: achievedBpp)
}
