import Foundation

// Site parameters structure (must match Metal shader).
struct VoronoiSite {
    var position: SIMD2<Float>
    var log_tau: Float
    var radius: Float
    var color: SIMD3<Float>
    var aniso_dir: SIMD2<Float> = SIMD2<Float>(1, 0)
    var log_aniso: Float = 0
}

struct AdamState {
    var m_pos: SIMD2<Float> = SIMD2<Float>(0, 0)
    var v_pos: Float = 0
    var m_log_tau: Float = 0
    var v_log_tau: Float = 0
    var m_radius: Float = 0
    var v_radius: Float = 0
    var m_color: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    var v_color: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    var m_dir: SIMD2<Float> = SIMD2<Float>(0, 0)
    var v_dir: Float = 0
    var m_log_aniso: Float = 0
    var v_log_aniso: Float = 0
    var _pad: SIMD2<Float> = SIMD2<Float>(0, 0)
}
