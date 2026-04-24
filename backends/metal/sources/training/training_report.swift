import Foundation
import Metal

struct ActiveSiteExport {
    let sites: [VoronoiSite]
    let activeCount: Int
    let minTau: Float
    let maxTau: Float
    let avgTau: Float
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

    var minTau: Float = Float.infinity
    var maxTau: Float = -Float.infinity
    var sumTau: Float = 0.0
    var activeCount = 0

    for i in 0..<siteCount {
        let site = sitesPtr[i]
        if site.position.x >= 0.0 {
            exportSites.append(site)
            activeCount += 1

            let tau = exp(site.log_tau)
            minTau = min(minTau, tau)
            maxTau = max(maxTau, tau)
            sumTau += tau
        }
    }

    if activeCount == 0 {
        minTau = 0.0
        maxTau = 0.0
    }

    let avgTau = activeCount > 0 ? (sumTau / Float(activeCount)) : 0.0

    return ActiveSiteExport(
        sites: exportSites,
        activeCount: activeCount,
        minTau: minTau,
        maxTau: maxTau,
        avgTau: avgTau
    )
}

struct TrainingOutputPaths {
    let baseFilename: String
    let imagePath: String
    let cellPath: String
    let tauHeatmapPath: String
    let packedPath: String
    let sitesPath: String
}

func makeOutputPaths(outputDir: String, targetPath: String) -> TrainingOutputPaths {
    let originalFilename = (targetPath as NSString).lastPathComponent
    let baseFilename = (originalFilename as NSString).deletingPathExtension

    return TrainingOutputPaths(
        baseFilename: baseFilename,
        imagePath: "\(outputDir)/\(baseFilename).png",
        cellPath: "\(outputDir)/\(baseFilename)_cells.png",
        tauHeatmapPath: "\(outputDir)/\(baseFilename)_tau_heatmap.png",
        packedPath: "\(outputDir)/\(baseFilename)_packed.png",
        sitesPath: "\(outputDir)/\(baseFilename)_sites.txt"
    )
}

func ensureDirectory(_ path: String) {
    try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
}
