import Foundation

struct PackedInferenceSite {
    var data: SIMD4<UInt32>
}

struct PackedSiteQuant {
    var logTauMin: Float
    var logTauScale: Float
    var radiusMin: Float
    var radiusScale: Float
    var colorRMin: Float
    var colorRScale: Float
    var colorGMin: Float
    var colorGScale: Float
    var colorBMin: Float
    var colorBScale: Float
}

private func isActiveSite(_ site: VoronoiSite) -> Bool {
    return site.position.x >= 0.0 && site.position.y >= 0.0
}

@inline(__always)
private func setActiveFlagWord0(_ word0: UInt32, active: Bool) -> UInt32 {
    let flag: UInt32 = active ? 0 : 1
    return (word0 & 0x3fffffff) | (flag << 30)
}

private func normalizedDir(_ dir: SIMD2<Float>) -> SIMD2<Float> {
    let len2 = dir.x * dir.x + dir.y * dir.y
    if len2 < 1.0e-8 {
        return SIMD2<Float>(1, 0)
    }
    return dir * (1.0 / sqrt(len2))
}

private func packUnorm16(_ value: Float, minValue: Float, scale: Float) -> UInt32 {
    if scale <= 0.0 {
        return 0
    }
    let n = min(max((value - minValue) / scale, 0.0), 1.0)
    return UInt32(n * 65535.0 + 0.5)
}

private func packUnorm16Max(_ value: Float, maxValue: Float) -> UInt32 {
    if maxValue <= 0.0 {
        return 0
    }
    let n = min(max(value / maxValue, 0.0), 1.0)
    return UInt32(n * 65535.0 + 0.5)
}

private func packUnorm15Max(_ value: Float, maxValue: Float) -> UInt32 {
    if maxValue <= 0.0 {
        return 0
    }
    let n = min(max(value / maxValue, 0.0), 1.0)
    return UInt32(n * 32767.0 + 0.5)
}

private func packUnorm11(_ value: Float, minValue: Float, scale: Float) -> UInt32 {
    if scale <= 0.0 {
        return 0
    }
    let n = min(max((value - minValue) / scale, 0.0), 1.0)
    return UInt32(n * 2047.0 + 0.5)
}

private func packUnorm10(_ value: Float, minValue: Float, scale: Float) -> UInt32 {
    if scale <= 0.0 {
        return 0
    }
    let n = min(max((value - minValue) / scale, 0.0), 1.0)
    return UInt32(n * 1023.0 + 0.5)
}

private func packAngle(_ dir: SIMD2<Float>) -> UInt32 {
    let d = normalizedDir(dir)
    let angle = atan2(d.y, d.x)
    let twoPi = Float.pi * 2.0
    let n = min(max((angle + Float.pi) / twoPi, 0.0), 1.0)
    return UInt32(n * 65535.0 + 0.5)
}

func makePackedInferenceSites(_ sites: [VoronoiSite],
                              width: Int,
                              height: Int) -> (sites: [PackedInferenceSite], quant: PackedSiteQuant) {
    let activeSites = sites.filter(isActiveSite)

    var minLogTau = Float.infinity
    var maxLogTau = -Float.infinity
    var minRadius = Float.infinity
    var maxRadius = -Float.infinity
    var minColorR = Float.infinity
    var maxColorR = -Float.infinity
    var minColorG = Float.infinity
    var maxColorG = -Float.infinity
    var minColorB = Float.infinity
    var maxColorB = -Float.infinity

    for site in activeSites {
        if site.log_tau.isFinite {
            minLogTau = min(minLogTau, site.log_tau)
            maxLogTau = max(maxLogTau, site.log_tau)
        }
        if site.radius.isFinite {
            minRadius = min(minRadius, site.radius)
            maxRadius = max(maxRadius, site.radius)
        }
        if site.color.x.isFinite {
            minColorR = min(minColorR, site.color.x)
            maxColorR = max(maxColorR, site.color.x)
        }
        if site.color.y.isFinite {
            minColorG = min(minColorG, site.color.y)
            maxColorG = max(maxColorG, site.color.y)
        }
        if site.color.z.isFinite {
            minColorB = min(minColorB, site.color.z)
            maxColorB = max(maxColorB, site.color.z)
        }
    }

    if !minLogTau.isFinite || !maxLogTau.isFinite {
        minLogTau = 0.0
        maxLogTau = 0.0
    }
    if !minRadius.isFinite || !maxRadius.isFinite {
        minRadius = 0.0
        maxRadius = 0.0
    }
    if !minColorR.isFinite || !maxColorR.isFinite {
        minColorR = 0.0
        maxColorR = 0.0
    }
    if !minColorG.isFinite || !maxColorG.isFinite {
        minColorG = 0.0
        maxColorG = 0.0
    }
    if !minColorB.isFinite || !maxColorB.isFinite {
        minColorB = 0.0
        maxColorB = 0.0
    }
    let logTauScale = max(maxLogTau - minLogTau, 1.0e-6)
    let radiusScale = max(maxRadius - minRadius, 1.0e-6)
    let colorRScale = max(maxColorR - minColorR, 1.0e-6)
    let colorGScale = max(maxColorG - minColorG, 1.0e-6)
    let colorBScale = max(maxColorB - minColorB, 1.0e-6)

    let quant = PackedSiteQuant(
        logTauMin: minLogTau,
        logTauScale: logTauScale,
        radiusMin: minRadius,
        radiusScale: radiusScale,
        colorRMin: minColorR,
        colorRScale: colorRScale,
        colorGMin: minColorG,
        colorGScale: colorGScale,
        colorBMin: minColorB,
        colorBScale: colorBScale
    )

    let maxX = Float(max(width - 1, 0))
    let maxY = Float(max(height - 1, 0))

    var packedSites: [PackedInferenceSite] = []
    packedSites.reserveCapacity(sites.count)

    for site in sites {
        let active = isActiveSite(site)
        let posX = min(max(site.position.x, 0.0), maxX)
        let posY = min(max(site.position.y, 0.0), maxY)
        let px = packUnorm15Max(posX, maxValue: maxX)
        let py = packUnorm15Max(posY, maxValue: maxY)
        var w0 = px | (py << 15)
        w0 = setActiveFlagWord0(w0, active: active)

        let r = packUnorm11(site.color.x, minValue: quant.colorRMin, scale: quant.colorRScale)
        let g = packUnorm11(site.color.y, minValue: quant.colorGMin, scale: quant.colorGScale)
        let b = packUnorm10(site.color.z, minValue: quant.colorBMin, scale: quant.colorBScale)
        let w1 = r | (g << 11) | (b << 22)

        let logTau = site.log_tau.isFinite ? site.log_tau : minLogTau
        let radius = site.radius.isFinite ? site.radius : minRadius
        let lt = packUnorm16(logTau, minValue: minLogTau, scale: logTauScale)
        let rd = packUnorm16(radius, minValue: minRadius, scale: radiusScale)
        let w2 = lt | (rd << 16)

        let angleBits = packAngle(site.aniso_dir)
        let logAniso = site.log_aniso.isFinite ? site.log_aniso : 0.0
        let logAnisoBits = UInt32(Float16(logAniso).bitPattern)
        let w3 = (angleBits & 0xffff) | (logAnisoBits << 16)

        packedSites.append(PackedInferenceSite(data: SIMD4<UInt32>(w0, w1, w2, w3)))
    }

    return (packedSites, quant)
}
