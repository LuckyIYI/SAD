import Foundation
import Metal

// Sanitize float for JSON export (replace NaN/Inf with 0).
func sanitizeFloat(_ value: Float) -> Float {
    if value.isNaN || value.isInfinite {
        return 0.0
    }
    return value
}

func loadSitesFromJSON(path: String) -> (sites: [VoronoiSite], width: Int, height: Int) {
    let url = URL(fileURLWithPath: path)
    let data = try! Data(contentsOf: url)
    let obj = try! JSONSerialization.jsonObject(with: data, options: [])
    guard let dict = obj as? [String: Any] else {
        fatalError("Invalid JSON (expected object): \(path)")
    }
    guard let width = dict["image_width"] as? Int,
          let height = dict["image_height"] as? Int,
          let sitesArr = dict["sites"] as? [[String: Any]] else {
        fatalError("Invalid sites JSON schema: \(path)")
    }

    var sites: [VoronoiSite] = []
    sites.reserveCapacity(sitesArr.count)
    for s in sitesArr {
        guard let pos = s["pos"] as? [Double], pos.count == 2,
              let color = s["color"] as? [Double], color.count == 3,
              let logTau = s["log_tau"] as? Double,
              let radiusValue = (s["radius"] as? Double) ?? (s["radius_sq"] as? Double) else {
            continue
        }
        let anisoDir = (s["aniso_dir"] as? [Double]) ?? [1.0, 0.0]
        let logAniso = (s["log_aniso"] as? Double) ?? 0.0

        let dirX = Float(anisoDir.count > 0 ? anisoDir[0] : 1.0)
        let dirY = Float(anisoDir.count > 1 ? anisoDir[1] : 0.0)

        sites.append(VoronoiSite(
            position: SIMD2<Float>(Float(pos[0]), Float(pos[1])),
            log_tau: Float(logTau),
            radius: Float(radiusValue),
            color: SIMD3<Float>(Float(color[0]), Float(color[1]), Float(color[2])),
            aniso_dir: SIMD2<Float>(dirX, dirY),
            log_aniso: Float(logAniso)
        ))
    }

    if sites.isEmpty {
        fatalError("No sites found in JSON: \(path)")
    }

    return (sites, width, height)
}

func loadSitesFromTXT(path: String) -> (sites: [VoronoiSite], width: Int?, height: Int?) {
    let text = try! String(contentsOfFile: path, encoding: .utf8)
    var sites: [VoronoiSite] = []
    sites.reserveCapacity(16384)
    var headerWidth: Int?
    var headerHeight: Int?

    for rawLine in text.split(separator: "\n") {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        if line.isEmpty { continue }
        if line.hasPrefix("#") {
            let lower = line.lowercased()
            if lower.contains("image size") {
                let tokens = line.split { $0 == " " || $0 == "\t" || $0 == ":" }
                let numbers = tokens.compactMap { Int($0) }
                if numbers.count >= 2 {
                    headerWidth = numbers[0]
                    headerHeight = numbers[1]
                }
            }
            continue
        }
        let parts = line.split(separator: " ")
        if parts.count != 7 && parts.count != 10 { continue }
        let vals = parts.compactMap { Double($0) }
        if vals.count != parts.count { continue }

        let posX = Float(vals[0])
        let posY = Float(vals[1])
        let colR = Float(vals[2])
        let colG = Float(vals[3])
        let colB = Float(vals[4])
        let logTau = Float(vals[5])
        let radiusValue = Float(vals[6])

        var dir = SIMD2<Float>(1, 0)
        var logAniso: Float = 0
        if vals.count == 10 {
            dir = SIMD2<Float>(Float(vals[7]), Float(vals[8]))
            logAniso = Float(vals[9])
        }

        sites.append(VoronoiSite(
            position: SIMD2<Float>(posX, posY),
            log_tau: logTau,
            radius: radiusValue,
            color: SIMD3<Float>(colR, colG, colB),
            aniso_dir: dir,
            log_aniso: logAniso
        ))
    }

    if sites.isEmpty {
        fatalError("No sites found in TXT: \(path)")
    }
    return (sites, headerWidth, headerHeight)
}

func writeSitesTXT(sites: [VoronoiSite], width: Int, height: Int, path: String) {
    func formatFloat(_ value: Float) -> String {
        return String(format: "%.9g", Double(value))
    }

    var sitesData = "# SAD Sites (position_x, position_y, color_r, color_g, color_b, log_tau, radius, aniso_dir_x, aniso_dir_y, log_aniso)\n"
    sitesData += "# Image size: \(width) \(height)\n"
    sitesData += "# Total sites: \(sites.count)\n"
    sitesData += "# Active sites: \(sites.count)\n"

    for site in sites {
        let line = [
            formatFloat(site.position.x),
            formatFloat(site.position.y),
            formatFloat(site.color.x),
            formatFloat(site.color.y),
            formatFloat(site.color.z),
            formatFloat(site.log_tau),
            formatFloat(site.radius),
            formatFloat(site.aniso_dir.x),
            formatFloat(site.aniso_dir.y),
            formatFloat(site.log_aniso)
        ].joined(separator: " ")
        sitesData += "\(line)\n"
    }

    if let sitesFileData = sitesData.data(using: .utf8) {
        try? sitesFileData.write(to: URL(fileURLWithPath: path))
    }
}

func writeSitesJSON(sites: [VoronoiSite], width: Int, height: Int,
                    neighbors: [[Int]], path: String) {
    var jsonData = """
    {
      \"image_width\": \(width),
      \"image_height\": \(height),
      \"site_count\": \(sites.count),
      \"sites\": [

    """

    for (siteIndex, site) in sites.enumerated() {
        if siteIndex > 0 {
            jsonData += ",\n"
        }
        let neigh = neighbors[siteIndex].map(String.init).joined(separator: ", ")

        let pos_x = sanitizeFloat(site.position.x)
        let pos_y = sanitizeFloat(site.position.y)
        let col_r = sanitizeFloat(site.color.x)
        let col_g = sanitizeFloat(site.color.y)
        let col_b = sanitizeFloat(site.color.z)
        let log_tau = sanitizeFloat(site.log_tau)
        let radius = sanitizeFloat(site.radius)
        let aniso_x = sanitizeFloat(site.aniso_dir.x)
        let aniso_y = sanitizeFloat(site.aniso_dir.y)
        let log_aniso = sanitizeFloat(site.log_aniso)

        jsonData += String(format: """
            {
              \"pos\": [%.6f, %.6f],
              \"color\": [%.6f, %.6f, %.6f],
              \"log_tau\": %.6f,
              \"radius\": %.6f,
              \"aniso_dir\": [%.6f, %.6f],
              \"log_aniso\": %.6f,
              \"neighbors\": [\(neigh)]
            }
        """, pos_x, pos_y, col_r, col_g, col_b,
             log_tau, radius, aniso_x, aniso_y, log_aniso)
    }

    jsonData += "\n  ]\n}\n"

    if let jsonFileData = jsonData.data(using: .utf8) {
        try? jsonFileData.write(to: URL(fileURLWithPath: path))
    }
}

func readRGBA32UIntTexture(_ texture: MTLTexture) -> [UInt32] {
    precondition(texture.pixelFormat == .rgba32Uint)
    let width = texture.width
    let height = texture.height
    let channels = 4
    var data = [UInt32](repeating: 0, count: width * height * channels)
    data.withUnsafeMutableBytes { raw in
        let bytesPerRow = width * channels * MemoryLayout<UInt32>.stride
        texture.getBytes(
            raw.baseAddress!,
            bytesPerRow: bytesPerRow,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0)
    }
    return data
}

func computeSiteNeighbors(
    cand0: MTLTexture,
    cand1: MTLTexture,
    sites: UnsafePointer<VoronoiSite>,
    siteCount: Int,
    activeMap: [Int],
    activeCount: Int,
    invScaleSq: Float
) -> [[Int]] {
    let width = cand0.width
    let height = cand0.height

    let cand0Data = readRGBA32UIntTexture(cand0)
    let cand1Data = readRGBA32UIntTexture(cand1)

    var posX = [Float](repeating: 0, count: siteCount)
    var posY = [Float](repeating: 0, count: siteCount)
    var dirX = [Float](repeating: 1, count: siteCount)
    var dirY = [Float](repeating: 0, count: siteCount)
    var tau = [Float](repeating: 0, count: siteCount)
    var l1 = [Float](repeating: 1, count: siteCount)
    var l2 = [Float](repeating: 1, count: siteCount)
    var rNorm = [Float](repeating: 0, count: siteCount)
    let invScale = sqrt(invScaleSq)

    for i in 0..<siteCount {
        if activeMap[i] < 0 { continue }
        let s = sites[i]
        posX[i] = s.position.x
        posY[i] = s.position.y
        dirX[i] = s.aniso_dir.x
        dirY[i] = s.aniso_dir.y
        tau[i] = expf(s.log_tau)
        let e = expf(s.log_aniso)
        l1[i] = e
        l2[i] = 1.0 / max(e, 1e-20)
        rNorm[i] = s.radius * invScale
    }

    var edgeSet = Set<UInt64>()
    edgeSet.reserveCapacity(max(activeCount * 8, 1024))

    let siteCountU = UInt32(siteCount)
    for y in 0..<height {
        let fy = Float(y)
        for x in 0..<width {
            let fx = Float(x)
            let p = (y * width + x) * 4

            var bestId = -1
            var bestLogit = -Float.infinity
            var secondId = -1
            var secondLogit = -Float.infinity

            @inline(__always)
            func considerCandidate(_ idU: UInt32) {
                if idU >= siteCountU { return }
                let id = Int(idU)
                let exportId = activeMap[id]
                if exportId < 0 { return }

                let dx = fx - posX[id]
                let dy = fy - posY[id]
                let diff2 = dx * dx + dy * dy
                let proj = dirX[id] * dx + dirY[id] * dy
                let proj2 = proj * proj
                let perp2 = max(diff2 - proj2, 0.0)
                let d2Aniso = l1[id] * proj2 + l2[id] * perp2
                let d2Norm = d2Aniso * invScaleSq
                let d2Safe = max(d2Norm, 1e-8)
                let dmix = sqrt(d2Safe) - rNorm[id]
                let logit = -tau[id] * dmix

                if logit > bestLogit {
                    secondLogit = bestLogit
                    secondId = bestId
                    bestLogit = logit
                    bestId = exportId
                } else if logit > secondLogit {
                    secondLogit = logit
                    secondId = exportId
                }
            }

            considerCandidate(cand0Data[p + 0])
            considerCandidate(cand0Data[p + 1])
            considerCandidate(cand0Data[p + 2])
            considerCandidate(cand0Data[p + 3])
            considerCandidate(cand1Data[p + 0])
            considerCandidate(cand1Data[p + 1])
            considerCandidate(cand1Data[p + 2])
            considerCandidate(cand1Data[p + 3])

            if bestId >= 0, secondId >= 0, bestId != secondId {
                let a = min(bestId, secondId)
                let b = max(bestId, secondId)
                let key = (UInt64(UInt32(a)) << 32) | UInt64(UInt32(b))
                edgeSet.insert(key)
            }
        }
    }

    var neighbors = Array(repeating: [Int](), count: activeCount)
    for key in edgeSet {
        let a = Int(UInt32(key >> 32))
        let b = Int(UInt32(key & 0xffffffff))
        if a >= 0, b >= 0, a < activeCount, b < activeCount {
            neighbors[a].append(b)
            neighbors[b].append(a)
        }
    }

    for i in 0..<activeCount {
        neighbors[i].sort()
    }
    return neighbors
}
