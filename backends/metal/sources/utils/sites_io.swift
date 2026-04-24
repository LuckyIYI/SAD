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
