#include "../sad_common.metal"

kernel void packCandidateSites(
    constant VoronoiSite *sites [[buffer(0)]],
    device PackedCandidateSite *packed [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= siteCount) return;
    VoronoiSite site = sites[gid];

    PackedCandidateSite out;
    if (site.position.x < 0.0f) {
        out.a = half4(half(-1.0f), half(-1.0f), half(0.0f), half(0.0f));
        out.b = half4(half(0.0f), half(0.0f), half(0.0f), half(0.0f));
    } else {
        out.a = half4(half(site.position.x), half(site.position.y),
                      half(site.log_tau), half(site.radius));
        out.b = half4(half(site.aniso_dir.x), half(site.aniso_dir.y),
                      half(site.log_aniso), half(0.0f));
    }

    packed[gid] = out;
}
