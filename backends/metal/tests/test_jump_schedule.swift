import Dispatch
import Foundation
import Metal
import simd

struct VoronoiSite {
    var position: SIMD2<Float>
    var log_tau: Float
    var radius: Float
    var color: SIMD3<Float>
    var aniso_dir: SIMD2<Float>
    var log_aniso: Float
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

func voronoi_dmix2(_ site: VoronoiSite, _ uv: SIMD2<Float>, _ invScaleSq: Float) -> Float {
    let diff = uv - site.position
    let diff2 = simd_dot(diff, diff)
    let proj = simd_dot(site.aniso_dir, diff)
    let proj2 = proj * proj
    let perp2 = max(diff2 - proj2, 0.0)
    let l1 = exp(site.log_aniso)
    let l2 = 1.0 / l1
    let d2_aniso = l1 * proj2 + l2 * perp2
    let d2_norm = d2_aniso * invScaleSq
    let d2_safe = max(d2_norm, 1e-8)
    let inv_scale = sqrt(invScaleSq)
    let r_norm = site.radius * inv_scale
    return sqrt(d2_safe) - r_norm
}

func insertClosest8(_ bestIdx: inout [UInt32], _ bestD2: inout [Float],
                    candIdx: UInt32, uv: SIMD2<Float>, sites: [VoronoiSite],
                    invScaleSq: Float) {
    if candIdx >= sites.count { return }
    for i in 0..<8 {
        if bestIdx[i] == candIdx { return }
    }
    let site = sites[Int(candIdx)]
    if site.position.x < 0.0 { return }
    let dMix2 = voronoi_dmix2(site, uv, invScaleSq)
    let tau = max(exp(site.log_tau), 1e-4)
    let d2 = tau * dMix2

    var insertPos = 8
    if d2 < bestD2[0] { insertPos = 0 }
    else if d2 < bestD2[1] { insertPos = 1 }
    else if d2 < bestD2[2] { insertPos = 2 }
    else if d2 < bestD2[3] { insertPos = 3 }
    else if d2 < bestD2[4] { insertPos = 4 }
    else if d2 < bestD2[5] { insertPos = 5 }
    else if d2 < bestD2[6] { insertPos = 6 }
    else if d2 < bestD2[7] { insertPos = 7 }

    if insertPos < 8 {
        for i in stride(from: 7, to: insertPos, by: -1) {
            bestD2[i] = bestD2[i - 1]
            bestIdx[i] = bestIdx[i - 1]
        }
        bestD2[insertPos] = d2
        bestIdx[insertPos] = candIdx
    }
}

func computeTop8CPU(samples: [SIMD2<Int>], width: Int, height: Int, sites: [VoronoiSite]) -> [[UInt32]] {
    let scale = Float(max(width, height))
    let invScaleSq = 1.0 / (scale * scale)
    var out = Array(repeating: [UInt32](repeating: 0xffffffff, count: 8), count: samples.count)
    for (s, p) in samples.enumerated() {
        var bestIdx = [UInt32](repeating: 0xffffffff, count: 8)
        var bestD2 = [Float](repeating: Float.infinity, count: 8)
        let uv = SIMD2<Float>(Float(p.x), Float(p.y))
        for i in 0..<sites.count {
            insertClosest8(&bestIdx, &bestD2, candIdx: UInt32(i),
                           uv: uv, sites: sites, invScaleSq: invScaleSq)
        }
        out[s] = bestIdx
    }
    return out
}

func evaluateCandidates(cpuTop8: [[UInt32]], gpuC0: [UInt32], gpuC1: [UInt32], width: Int, height: Int, samples: [SIMD2<Int>]) {
    var matchCount = 0
    var totalHits = 0
    for (s, p) in samples.enumerated() {
        let base4 = (p.y * width + p.x) * 4
        var gpuSet = [UInt32]()
        gpuSet.reserveCapacity(8)
        gpuSet.append(gpuC0[base4 + 0])
        gpuSet.append(gpuC0[base4 + 1])
        gpuSet.append(gpuC0[base4 + 2])
        gpuSet.append(gpuC0[base4 + 3])
        gpuSet.append(gpuC1[base4 + 0])
        gpuSet.append(gpuC1[base4 + 1])
        gpuSet.append(gpuC1[base4 + 2])
        gpuSet.append(gpuC1[base4 + 3])

        var hits = 0
        for i in 0..<8 {
            let tgt = cpuTop8[s][i]
            if gpuSet.contains(tgt) {
                hits += 1
            }
        }
        totalHits += hits
        if hits == 8 { matchCount += 1 }
    }
    let count = samples.count
    let recall = Double(totalHits) / Double(count * 8)
    let full = Double(matchCount) / Double(count)
    print(String(format: "Recall@8: %.4f | Full match: %.4f", recall, full))
}

func measureCommandBuffer(_ commandBuffer: MTLCommandBuffer) -> (Double, Bool) {
    let start = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds
    let wallMs = Double(end - start) / 1_000_000.0

    let gpuStart = commandBuffer.gpuStartTime
    let gpuEnd = commandBuffer.gpuEndTime
    if gpuEnd > gpuStart && gpuStart > 0 {
        return ((gpuEnd - gpuStart) * 1000.0, true)
    }
    return (wallMs, false)
}

func readRGBA32UInt(_ tex: MTLTexture) -> [UInt32] {
    let width = tex.width
    let height = tex.height
    var data = [UInt32](repeating: 0, count: width * height * 4)
    let bytesPerRow = width * 4 * MemoryLayout<UInt32>.stride
    data.withUnsafeMutableBytes { ptr in
        let region = MTLRegionMake2D(0, 0, width, height)
        tex.getBytes(ptr.baseAddress!, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
    }
    return data
}

func encodeJfaFloodOnly(
    cand0: MTLTexture,
    cand1: MTLTexture,
    sitesBuffer: MTLBuffer,
    siteCount: UInt32,
    invScaleSq: Float,
    floodPipeline: MTLComputePipelineState,
    in commandBuffer: MTLCommandBuffer)
{
    let width = cand0.width
    let height = cand0.height
    let maxDim = max(width, height)

    var numPasses = 0
    var step = 1
    while step < maxDim {
        step <<= 1
        numPasses += 1
    }

    if numPasses == 0 { return }

    var stepSize = step / 2
    var readFromCand0 = true

    let threadGroupSize2D = MTLSize(width: 16, height: 16, depth: 1)
    let threadGroups2D = MTLSize(
        width: (width + 15) / 16,
        height: (height + 15) / 16,
        depth: 1)

    while stepSize >= 1 {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { break }
        encoder.label = "JFA Flood Only (step=\(stepSize))"
        encoder.setComputePipelineState(floodPipeline)
        if readFromCand0 {
            encoder.setTexture(cand0, index: 0)
            encoder.setTexture(cand1, index: 1)
        } else {
            encoder.setTexture(cand1, index: 0)
            encoder.setTexture(cand0, index: 1)
        }
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var siteCountVar = siteCount
        var stepSizeVar = UInt32(stepSize)
        var invScaleSqVar = invScaleSq
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&stepSizeVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 3)

        encoder.dispatchThreadgroups(threadGroups2D, threadsPerThreadgroup: threadGroupSize2D)
        encoder.endEncoding()

        readFromCand0.toggle()
        stepSize /= 2
    }

    // Odd number of passes leaves the result in cand1.
    if numPasses % 2 == 1 {
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.label = "JFA Copy Result"
            blitEncoder.copy(
                from: cand1,
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

func makeSites(width: Int, height: Int, count: Int, seed: UInt64) -> [VoronoiSite] {
    var rng = seed
    func next() -> UInt32 {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return UInt32(truncatingIfNeeded: rng)
    }
    let cellCount = max(1, width * height)
    let maxLayer = (count - 1) / cellCount
    let offsetDenom = Float(maxLayer + 2)
    var usedCells = Set<Int>()
    usedCells.reserveCapacity(min(count, cellCount))
    var layer = 0
    var sites = [VoronoiSite]()
    sites.reserveCapacity(count)
    while sites.count < count {
        if usedCells.count >= cellCount {
            usedCells.removeAll(keepingCapacity: true)
            layer += 1
        }
        var cell = Int(next() % UInt32(cellCount))
        while usedCells.contains(cell) {
            cell = Int(next() % UInt32(cellCount))
        }
        usedCells.insert(cell)
        let ix = cell % width
        let iy = cell / width
        let offset = (Float(layer) + 1.0) / offsetDenom
        let x = Float(ix) + offset
        let y = Float(iy) + offset
        let angle = Float(next() % 6283) / 1000.0
        let dir = SIMD2<Float>(cos(angle), sin(angle))
        let site = VoronoiSite(position: SIMD2<Float>(x, y),
                               log_tau: 0.0,
                               radius: 0.0,
                               color: SIMD3<Float>(0, 0, 0),
                               aniso_dir: dir,
                               log_aniso: 0.0)
        sites.append(site)
    }
    return sites
}

func runTest() throws {
    let domainSizes = [256, 1024, 2048]
    let passesToTest = [8, 12, 16]
    let siteCounts = [2048, 16384, 65536, 131072]
    let siteSeeds: [UInt64] = [1, 2]
    let initSeeds: [UInt32] = [1, 123]
    let sampleCount = 256
    struct Config {
        let name: String
        let radiusScale: Float
        let radiusProbes: UInt32
        let injectCount: UInt32
    }
    struct TrialMetrics {
        let recall: Double
        let full: Double
        let initMs: Double
        let jfaMs: Double
        let vptMs: Double
    }
    struct TestMode {
        let name: String
        let useJfa: Bool
    }
    let configs: [Config] = [
        Config(name: "R64 P0 I16", radiusScale: 64, radiusProbes: 0, injectCount: 16),
        Config(name: "R64 P0 I24", radiusScale: 64, radiusProbes: 0, injectCount: 24),
        Config(name: "R64 P0 I32", radiusScale: 64, radiusProbes: 0, injectCount: 32),
        Config(name: "R64 P0 I40", radiusScale: 64, radiusProbes: 0, injectCount: 40),
        Config(name: "R64 P4 I16", radiusScale: 64, radiusProbes: 4, injectCount: 16),
    ]
    let modes: [TestMode] = [
        TestMode(name: "Seed+VPT", useJfa: false),
        TestMode(name: "Seed+JFA+VPT", useJfa: true),
    ]

    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("Metal not supported")
    }

    let libURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("sad.metallib")
    let library = try device.makeLibrary(URL: libURL)
    let commandQueue = device.makeCommandQueue()!

    let initCandidatesEncoder = try PackedCandidatesEncoder(device: device, library: library)
    let candidatePackEncoder = try CandidatePackEncoder(device: device, library: library)
    let compactCandidatesEncoder = try CompactCandidatesEncoder(device: device, library: library)
    let jfaEncoder = try JFAEncoder(device: device, library: library)
    guard let jfaFloodFunc = library.makeFunction(name: "jfaFlood") else {
        throw NSError(domain: "Test", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Missing jfaFlood kernel"])
    }
    let jfaFloodPipeline = try device.makeComputePipelineState(function: jfaFloodFunc)

    var usesGpuTimestamps = true

    for size in domainSizes {
        let width = size
        let height = size
        print("\n=== Domain \(width)x\(height) ===")

        let candDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Uint,
            width: width,
            height: height,
            mipmapped: false)
        candDesc.usage = [.shaderRead, .shaderWrite]

        let scale = Float(max(width, height))
        let invScaleSq = 1.0 / (scale * scale)

        var rng = UInt64(12345)
        func nextSample() -> UInt32 {
            rng ^= rng << 13
            rng ^= rng >> 7
            rng ^= rng << 17
            return UInt32(truncatingIfNeeded: rng)
        }
        var samples = [SIMD2<Int>]()
        samples.reserveCapacity(sampleCount)
        for _ in 0..<sampleCount {
            let x = Int(nextSample() % UInt32(width))
            let y = Int(nextSample() % UInt32(height))
            samples.append(SIMD2<Int>(x, y))
        }

        for siteCount in siteCounts {
            print("\nSite count: \(siteCount)")
            var results = [String: [String: [Int: [TrialMetrics]]]]()
            for mode in modes {
                var perConfig = [String: [Int: [TrialMetrics]]]()
                for cfg in configs {
                    var perPass = [Int: [TrialMetrics]]()
                    for passes in passesToTest {
                        perPass[passes] = []
                    }
                    perConfig[cfg.name] = perPass
                }
                results[mode.name] = perConfig
            }

            for siteSeed in siteSeeds {
                let sites = makeSites(width: width, height: height, count: siteCount, seed: siteSeed)
                let sitesBuffer = device.makeBuffer(length: MemoryLayout<VoronoiSite>.stride * sites.count, options: .storageModeShared)!
                memcpy(sitesBuffer.contents(), sites, MemoryLayout<VoronoiSite>.stride * sites.count)
                let packedCandidatesBuffer = device.makeBuffer(
                    length: MemoryLayout<PackedCandidateSite>.stride * sites.count,
                    options: .storageModeShared
                )!
                let cpuTop8 = computeTop8CPU(samples: samples, width: width, height: height, sites: sites)

                for initSeed in initSeeds {
                    for cfg in configs {
                        for passes in passesToTest {
                            for mode in modes {
                                var cand0A = device.makeTexture(descriptor: candDesc)!
                                var cand1A = device.makeTexture(descriptor: candDesc)!
                                var cand0B = device.makeTexture(descriptor: candDesc)!
                                var cand1B = device.makeTexture(descriptor: candDesc)!

                                var initMs = 0.0
                                var jfaMs = 0.0
                                var vptMs = 0.0

                                if let commandBuffer = commandQueue.makeCommandBuffer() {
                                    initCandidatesEncoder.encodeInit(cand0: cand0A, cand1: cand1A,
                                                                      siteCount: UInt32(siteCount), seed: initSeed, perPixelMode: false,
                                                                      in: commandBuffer)
                                    jfaEncoder.encodeSeed(cand0: cand0A,
                                                          sitesBuffer: sitesBuffer, siteCount: UInt32(siteCount),
                                                          in: commandBuffer)
                                    let (ms, usedGpu) = measureCommandBuffer(commandBuffer)
                                    initMs = ms
                                    if !usedGpu { usesGpuTimestamps = false }
                                }

                                if mode.useJfa {
                                    if let commandBuffer = commandQueue.makeCommandBuffer() {
                                        encodeJfaFloodOnly(cand0: cand0A, cand1: cand0B,
                                                           sitesBuffer: sitesBuffer, siteCount: UInt32(siteCount),
                                                           invScaleSq: invScaleSq, floodPipeline: jfaFloodPipeline,
                                                           in: commandBuffer)
                                        let (ms, usedGpu) = measureCommandBuffer(commandBuffer)
                                        jfaMs = ms
                                        if !usedGpu { usesGpuTimestamps = false }
                                    }
                                }

                                if passes > 0 {
                                    if let commandBuffer = commandQueue.makeCommandBuffer() {
                                        packCandidateSites(encoder: candidatePackEncoder, commandBuffer: commandBuffer,
                                                           sitesBuffer: sitesBuffer,
                                                           packedBuffer: packedCandidatesBuffer,
                                                           siteCount: UInt32(siteCount))
                                        for pass in 0..<passes {
                                            let step = packJumpStep(UInt32(pass), width: width, height: height)
                                            let stepHigh = UInt32(pass) >> 16
                                            compactCandidatesEncoder.encodeUpdate(
                                                cand0In: cand0A, cand1In: cand1A,
                                                cand0Out: cand0B, cand1Out: cand1B,
                                                packedSitesBuffer: packedCandidatesBuffer, siteCount: UInt32(siteCount),
                                                step: step, stepHigh: stepHigh, invScaleSq: invScaleSq,
                                                radiusScale: cfg.radiusScale, radiusProbes: cfg.radiusProbes,
                                                injectCount: cfg.injectCount,
                                                candDownscale: 1,
                                                targetWidth: UInt32(width),
                                                targetHeight: UInt32(height),
                                                in: commandBuffer)
                                            swap(&cand0A, &cand0B)
                                            swap(&cand1A, &cand1B)
                                        }
                                        let (ms, usedGpu) = measureCommandBuffer(commandBuffer)
                                        vptMs = ms
                                        if !usedGpu { usesGpuTimestamps = false }
                                    }
                                }

                                let gpuC0 = readRGBA32UInt(cand0A)
                                let gpuC1 = readRGBA32UInt(cand1A)

                                var matchCount = 0
                                var totalHits = 0
                                for (s, p) in samples.enumerated() {
                                    let base4 = (p.y * width + p.x) * 4
                                    var gpuSet = [UInt32]()
                                    gpuSet.reserveCapacity(8)
                                    gpuSet.append(gpuC0[base4 + 0])
                                    gpuSet.append(gpuC0[base4 + 1])
                                    gpuSet.append(gpuC0[base4 + 2])
                                    gpuSet.append(gpuC0[base4 + 3])
                                    gpuSet.append(gpuC1[base4 + 0])
                                    gpuSet.append(gpuC1[base4 + 1])
                                    gpuSet.append(gpuC1[base4 + 2])
                                    gpuSet.append(gpuC1[base4 + 3])

                                    var hits = 0
                                    for i in 0..<8 {
                                        let tgt = cpuTop8[s][i]
                                        if gpuSet.contains(tgt) {
                                            hits += 1
                                        }
                                    }
                                    totalHits += hits
                                    if hits == 8 { matchCount += 1 }
                                }
                                let recall = Double(totalHits) / Double(sampleCount * 8)
                                let full = Double(matchCount) / Double(sampleCount)
                                let metrics = TrialMetrics(recall: recall, full: full,
                                                          initMs: initMs, jfaMs: jfaMs, vptMs: vptMs)
                                results[mode.name]?[cfg.name]?[passes]?.append(metrics)
                            }
                        }
                    }
                }
            }

            for mode in modes {
                print("\nMode: \(mode.name)")
                for cfg in configs {
                    let ops = 8 + Int(cfg.radiusProbes) + Int(cfg.injectCount)
                    print("Config \(cfg.name) | opsApprox \(ops)")
                    for passes in passesToTest {
                        let vals = results[mode.name]?[cfg.name]?[passes] ?? []
                        let denom = Double(max(vals.count, 1))
                        let recallMean = vals.map { $0.recall }.reduce(0, +) / denom
                        let fullMean = vals.map { $0.full }.reduce(0, +) / denom
                        let initMean = vals.map { $0.initMs }.reduce(0, +) / denom
                        let jfaMean = vals.map { $0.jfaMs }.reduce(0, +) / denom
                        let vptMean = vals.map { $0.vptMs }.reduce(0, +) / denom
                        let totalMean = initMean + jfaMean + vptMean
                        let perPass = passes > 0 ? vptMean / Double(passes) : 0.0
                        print(String(format: "  Passes: %d | Recall@8 mean: %.4f | Full match mean: %.4f (trials=%d)",
                                     passes, recallMean, fullMean, vals.count))
                        print(String(format: "           Total ms: %.3f | VPT ms/pass: %.3f | Init ms: %.3f | JFA ms: %.3f",
                                     totalMean, perPass, initMean, jfaMean))
                    }
                }
            }
        }
    }

    if !usesGpuTimestamps {
        print("\nNote: GPU timestamps unavailable, timings use CPU wall time.")
    }
}

@main
struct TestProgram {
    static func main() {
        do {
            try runTest()
        } catch {
            fatalError("Test failed: \(error)")
        }
    }
}
