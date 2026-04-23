import Foundation

enum InitMode {
    case gradientWeighted
    case perPixel
    case fromSites(String)
}

struct TrainingOptions {
    var targetPath: String
    var maskPath: String?
    var iterations: Int
    var targetBpp: Float?

    var initMode: InitMode
    var nSites: Int
    var initGradientAlpha: Float
    var initLogTau: Float
    var initRadius: Float
    var maxDim: Int

    var prunePercentile: Float
    var pruneStartIter: Int
    var pruneFreq: Int
    var pruneEndIter: Int?
    var pruneDuringDensify: Bool

    var densifyEnabled: Bool
    var densifyStart: Int
    var densifyFreq: Int
    var densifyEnd: Int
    var densifyPercentile: Float
    var densifyScoreAlpha: Float
    var maxSites: Int

    var candUpdateFreq: Int
    var candUpdatePasses: Int
    var candDownscale: Int
    var candRadiusScale: Float
    var candRadiusProbes: UInt32
    var candInjectCount: UInt32
    var candHilbertWindow: UInt32
    var candHilbertProbes: UInt32
    var exportCandPasses: Int?

    var lrScale: Float
    var lrPosBase: Float
    var lrTauBase: Float
    var lrRadiusBase: Float
    var lrColorBase: Float
    var lrDirBase: Float
    var lrAnisoBase: Float
    var beta1: Float
    var beta2: Float
    var eps: Float

    var ssimWeight: Float
    var ssimMetric: Bool

    var showViewer: Bool
    var viewerFreq: Int
    var logFreq: Int
    var traceFrame: Int?

    var outputDir: String
    var exportNeighbors: Bool
    var packedPSNR: Bool
}

extension TrainingOptions {
    var initFromSitesPath: String? {
        if case .fromSites(let path) = initMode { return path }
        return nil
    }

    var initPerPixel: Bool {
        if case .perPixel = initMode { return true }
        return false
    }
}
