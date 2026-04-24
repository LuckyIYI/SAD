import Foundation
import Metal

private func initSitesFromFile(path: String, targetWidth: Int, targetHeight: Int) -> [VoronoiSite] {
    let ext = URL(fileURLWithPath: path).pathExtension.lowercased()
    let loaded = ext == "json"
        ? loadSitesFromJSON(path: path)
        : {
            let txt = loadSitesFromTXT(path: path)
            return (sites: txt.sites, width: txt.width ?? targetWidth, height: txt.height ?? targetHeight)
        }()
    let loadedSites = loaded.sites

    let sites: [VoronoiSite]
    if loaded.width != targetWidth || loaded.height != targetHeight {
        print("Warning: Loaded sites dimensions (\(loaded.width)x\(loaded.height)) differ from target (\(targetWidth)x\(targetHeight))")

        let scaleX = Float(targetWidth) / Float(loaded.width)
        let scaleY = Float(targetHeight) / Float(loaded.height)
        sites = loadedSites.map { site in
            var scaled = site
            scaled.position.x *= scaleX
            scaled.position.y *= scaleY
            return scaled
        }
    } else {
        sites = loadedSites
    }

    return sites
}

private func initSitesGradientWeighted(options: TrainingOptions, target: TargetImage,
                                       device: MTLDevice, commandQueue: MTLCommandQueue,
                                       library: MTLLibrary) -> [VoronoiSite] {
    var sites = Array(repeating: VoronoiSite(
        position: SIMD2<Float>(-1, -1),
        log_tau: options.initLogTau,
        radius: options.initRadius,
        color: SIMD3<Float>(0.5, 0.5, 0.5),
        aniso_dir: SIMD2<Float>(1, 0),
        log_aniso: 0
    ), count: options.nSites)

    let initSeedBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    memset(initSeedBuffer.contents(), 0, MemoryLayout<UInt32>.stride)

    let tempSitesBuffer = device.makeBuffer(
        length: MemoryLayout<VoronoiSite>.stride * options.nSites,
        options: .storageModeShared
    )!
    let initSitesEncoder = try! InitSitesEncoder(device: device, library: library)
    if let commandBuffer = commandQueue.makeCommandBuffer() {
        initSitesEncoder.encode(sitesBuffer: tempSitesBuffer, numSites: UInt32(options.nSites),
                                seedCounterBuffer: initSeedBuffer,
                                targetTexture: target.texture, maskTexture: target.maskTexture,
                                gradThreshold: 0.01 * options.initGradientAlpha,
                                maxAttempts: 256,
                                initLogTau: options.initLogTau,
                                initRadius: options.initRadius,
                                in: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    let tempPtr = tempSitesBuffer.contents().bindMemory(to: VoronoiSite.self, capacity: options.nSites)
    for i in 0..<options.nSites {
        sites[i] = tempPtr[i]
    }

    return sites
}

private func initSitesPerPixel(options: TrainingOptions, target: TargetImage) -> [VoronoiSite] {
    let width = target.width
    let height = target.height

    var sites: [VoronoiSite] = []
    sites.reserveCapacity(width * height)
    for y in 0..<height {
        for x in 0..<width {
            let idx = (y * width + x) * 4
            let color = SIMD3<Float>(
                target.floatData[idx],
                target.floatData[idx + 1],
                target.floatData[idx + 2]
            )

            sites.append(VoronoiSite(
                position: SIMD2<Float>(Float(x), Float(y)),
                log_tau: options.initLogTau,
                radius: options.initRadius,
                color: color,
                aniso_dir: SIMD2<Float>(1, 0),
                log_aniso: 0
            ))
        }
    }
    return sites
}

func initializeSites(options: TrainingOptions, target: TargetImage,
                     device: MTLDevice, commandQueue: MTLCommandQueue,
                     library: MTLLibrary) -> [VoronoiSite] {
    switch options.initMode {
    case .fromSites(let path):
        return initSitesFromFile(path: path,
                                 targetWidth: target.width,
                                 targetHeight: target.height)
    case .perPixel:
        return initSitesPerPixel(options: options, target: target)
    case .gradientWeighted:
        return initSitesGradientWeighted(options: options, target: target,
                                         device: device, commandQueue: commandQueue,
                                         library: library)
    }
}
