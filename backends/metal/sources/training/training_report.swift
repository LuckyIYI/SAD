import Foundation
import Metal

struct ActiveSiteExport {
    let sites: [VoronoiSite]
    let activeMap: [Int]
    let activeCount: Int
    let minTau: Float
    let maxTau: Float
    let avgTau: Float
    let minRadius: Float
    let maxRadius: Float
    let avgRadius: Float
}

func countActiveSites(sitesBuffer: MTLBuffer, siteCount: Int) -> Int {
    let sitesPtr = sitesBuffer.contents().bindMemory(to: VoronoiSite.self, capacity: siteCount)
    var count = 0
    for i in 0..<siteCount where sitesPtr[i].position.x >= 0.0 {
        count += 1
    }
    return count
}

func exportActiveSites(sitesBuffer: MTLBuffer, siteCount: Int) -> ActiveSiteExport {
    let sitesPtr = sitesBuffer.contents().bindMemory(to: VoronoiSite.self, capacity: siteCount)
    var exportSites: [VoronoiSite] = []
    exportSites.reserveCapacity(siteCount)
    var activeMap = [Int](repeating: -1, count: siteCount)

    var minTau: Float = Float.infinity
    var maxTau: Float = -Float.infinity
    var sumTau: Float = 0.0
    var minRadius: Float = Float.infinity
    var maxRadius: Float = -Float.infinity
    var sumRadius: Float = 0.0
    var activeCount = 0

    for i in 0..<siteCount {
        let site = sitesPtr[i]
        if site.position.x >= 0.0 {
            activeMap[i] = activeCount
            exportSites.append(site)
            activeCount += 1

            let tau = exp(site.log_tau)
            minTau = min(minTau, tau)
            maxTau = max(maxTau, tau)
            sumTau += tau

            let radius = site.radius
            minRadius = min(minRadius, radius)
            maxRadius = max(maxRadius, radius)
            sumRadius += radius
        }
    }

    if activeCount == 0 {
        minTau = 0.0
        maxTau = 0.0
        minRadius = 0.0
        maxRadius = 0.0
    }

    let avgTau = activeCount > 0 ? (sumTau / Float(activeCount)) : 0.0
    let avgRadius = activeCount > 0 ? (sumRadius / Float(activeCount)) : 0.0

    return ActiveSiteExport(
        sites: exportSites,
        activeMap: activeMap,
        activeCount: activeCount,
        minTau: minTau,
        maxTau: maxTau,
        avgTau: avgTau,
        minRadius: minRadius,
        maxRadius: maxRadius,
        avgRadius: avgRadius
    )
}

struct TrainingOutputPaths {
    let baseFilename: String
    let imagePath: String
    let cellPath: String
    let centroidsPath: String
    let tauHeatmapPath: String
    let packedPath: String
    let reportPath: String
    let mdPath: String
    let sitesPath: String
    let jsonPath: String
}

func makeOutputPaths(outputDir: String, targetPath: String) -> TrainingOutputPaths {
    let originalFilename = (targetPath as NSString).lastPathComponent
    let baseFilename = (originalFilename as NSString).deletingPathExtension

    return TrainingOutputPaths(
        baseFilename: baseFilename,
        imagePath: "\(outputDir)/\(baseFilename).png",
        cellPath: "\(outputDir)/\(baseFilename)_cells.png",
        centroidsPath: "\(outputDir)/\(baseFilename)_centroids.png",
        tauHeatmapPath: "\(outputDir)/\(baseFilename)_tau_heatmap.png",
        packedPath: "\(outputDir)/\(baseFilename)_packed.png",
        reportPath: "\(outputDir)/\(baseFilename)_stats.txt",
        mdPath: "\(outputDir)/\(baseFilename)_stats.md",
        sitesPath: "\(outputDir)/\(baseFilename)_sites.txt",
        jsonPath: "\(outputDir)/\(baseFilename)_sites.json"
    )
}

func ensureDirectory(_ path: String) {
    try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
}

func makeTimestamp() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
    return formatter.string(from: Date())
}

struct TrainingStats {
    let initialSites: Int
    let finalSites: Int
    let finalPSNR: Float
    let bestPSNR: Float
    let finalSSIM: Float?
    let bestSSIM: Float?
    let totalTime: TimeInterval
    let avgSpeed: Float
    let minTau: Float
    let maxTau: Float
    let avgTau: Float
    let minRadius: Float
    let maxRadius: Float
    let avgRadius: Float
}

func buildTextReport(options: TrainingOptions,
                     width: Int,
                     height: Int,
                     effectivePruneEndIter: Int,
                     paths: TrainingOutputPaths,
                     timestamp: String,
                     stats: TrainingStats) -> String {
    let sitesPruned = stats.initialSites - stats.finalSites
    let prunedPercent = stats.initialSites > 0
        ? (Float(sitesPruned) / Float(stats.initialSites) * 100.0)
        : 0.0

    let ssimReportBlock: String
    if let finalSSIM = stats.finalSSIM, let bestSSIM = stats.bestSSIM {
        ssimReportBlock = "    - Final SSIM: \(String(format: "%.4f", finalSSIM))\n    - Best SSIM: \(String(format: "%.4f", bestSSIM))\n"
    } else {
        ssimReportBlock = ""
    }

    let initDesc: String
    if let path = options.initFromSitesPath {
        initDesc = "From previous sites (\(path))"
    } else if options.initPerPixel {
        initDesc = "Per-pixel (1 site per pixel)"
    } else {
        initDesc = "Gradient-weighted"
    }

    let lrPos = options.lrPosBase * options.lrScale
    let lrTau = options.lrTauBase * options.lrScale
    let lrRadius = options.lrRadiusBase * options.lrScale
    let lrColor = options.lrColorBase * options.lrScale

    return """
    ========================================
    SAD DIFFERENTIABLE FITTING - RUN STATISTICS
    ========================================
    Timestamp: \(timestamp)
    Target: \(options.targetPath)
    Image Size: \(width)x\(height)

    CONFIGURATION:
    - Initialization: \(initDesc)
    - Initial Sites: \(stats.initialSites)
    - Iterations: \(options.iterations)
    - Prune Percentile: \(options.prunePercentile)
    - Pruning Start: iter \(options.pruneStartIter)
    - Pruning End: iter \(effectivePruneEndIter)
    - Pruning Frequency: every \(options.pruneFreq) iterations

    LEARNING RATES:
    - Position: \(lrPos)
    - Log Tau: \(lrTau)
    - Radius: \(lrRadius)\(lrRadius == 0.0 ? " (DISABLED)" : "")
    - Color: \(lrColor)

    RESULTS:
    - Final Active Sites: \(stats.finalSites)
    - Sites Pruned: \(sitesPruned) (\(String(format: "%.1f%%", prunedPercent)))
    - Final PSNR: \(String(format: "%.2f dB", stats.finalPSNR))
    - Best PSNR: \(String(format: "%.2f dB", stats.bestPSNR))
    \(ssimReportBlock)
    - Total Time: \(String(format: "%.2fs", stats.totalTime))
    - Average Speed: \(String(format: "%.1f it/s", stats.avgSpeed))

    EFFICIENCY:
    - Sites per pixel: \(String(format: "%.3f", Float(stats.finalSites) / Float(width * height)))
    - Compression ratio: \(String(format: "%.2fx", Float(width * height) / Float(stats.finalSites)))

    SITE STATISTICS (Active Sites Only):
    - Tau (temperature):
      * Range: \(String(format: "%.1f", stats.minTau)) - \(String(format: "%.1f", stats.maxTau))
      * Average: \(String(format: "%.1f", stats.avgTau))
    - Radius:
      * Range: \(String(format: "%.2f", stats.minRadius)) - \(String(format: "%.2f", stats.maxRadius)) pixels
      * Average: \(String(format: "%.2f", stats.avgRadius)) pixels

    OUTPUT:
    - Image: \(paths.imagePath)
    - Cell visualization: \(paths.cellPath)
    - Report: \(paths.reportPath)
    ========================================

    """
}

func buildMarkdownReport(options: TrainingOptions,
                         width: Int,
                         height: Int,
                         timestamp: String,
                         stats: TrainingStats) -> String {
    let sitesPruned = stats.initialSites - stats.finalSites
    let prunedPercent = stats.initialSites > 0
        ? (Float(sitesPruned) / Float(stats.initialSites) * 100.0)
        : 0.0
    let totalPixels = width * height
    let compressionRatio = stats.finalSites > 0
        ? Float(totalPixels) / Float(stats.finalSites * 2)
        : 0.0

    let ssimRow: String
    if let finalSSIM = stats.finalSSIM {
        ssimRow = "\n| **SSIM** | \(String(format: "%.4f", finalSSIM)) |"
    } else {
        ssimRow = ""
    }

    let initDesc = options.initPerPixel ? "Per-pixel init" : "Gradient-weighted init"

    return """
    # SAD Results

    **Image:** `\(options.targetPath)` (\(width)x\(height))
    **Date:** \(timestamp)

    | Metric | Value |
    |--------|-------|
    | **PSNR** | \(String(format: "%.2f dB", stats.finalPSNR)) |\(ssimRow)
    | **Initial Sites** | \(stats.initialSites) |
    | **Final Sites** | \(stats.finalSites) |
    | **Sites Pruned** | \(sitesPruned) (\(String(format: "%.1f%%", prunedPercent))) |
    | **Compression Ratio** | \(String(format: "%.2fx", compressionRatio)) |

    > Compression assumes each site costs 2x a regular pixel

    ---

    **Configuration:** \(initDesc) - \(options.iterations) iterations - Prune percentile \(options.prunePercentile)

    """
}
