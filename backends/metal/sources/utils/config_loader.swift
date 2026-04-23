import Foundation

struct TrainingConfig: Decodable {
    var DEFAULT_SITES: Int?
    var DEFAULT_MAX_SITES: Int?
    var DEFAULT_ITERS: Int?
    var DEFAULT_TARGET_BPP: Float?
    var PRUNE_PERCENTILE: Float?
    var PRUNE_DURING_DENSIFY: Bool?
    var PRUNE_START: Int?
    var PRUNE_FREQ: Int?
    var PRUNE_END: Int?
    var DENSIFY_START: Int?
    var DENSIFY_FREQ: Int?
    var DENSIFY_END: Int?
    var DENSIFY_PERCENTILE: Float?
    var DENSIFY_SCORE_ALPHA: Float?
    var CAND_UPDATE_FREQ: Int?
    var CAND_UPDATE_PASSES: Int?
    var CAND_DOWNSCALE: Int?
    var CAND_RADIUS_SCALE: Float?
    var CAND_RADIUS_PROBES: Int?
    var CAND_INJECT_COUNT: Int?
    var CAND_HILBERT_WINDOW: Int?
    var CAND_HILBERT_PROBES: Int?
    var LR_POS_BASE: Float?
    var LR_TAU_BASE: Float?
    var LR_RADIUS_BASE: Float?
    var LR_COLOR_BASE: Float?
    var LR_DIR_BASE: Float?
    var LR_ANISO_BASE: Float?
    var INIT_LOG_TAU: Float?
    var INIT_RADIUS: Float?
    var BETA1: Float?
    var BETA2: Float?
    var EPS: Float?
    var MAX_DIM: Int?
}

func loadTrainingConfig(path: String) -> TrainingConfig? {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        return nil
    }
    return try? JSONDecoder().decode(TrainingConfig.self, from: data)
}
