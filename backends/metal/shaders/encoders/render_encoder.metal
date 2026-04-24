#include "../sad_common.metal"

kernel void renderVoronoi(
    texture2d<uint, access::read>   candidates0 [[texture(0)]],
    texture2d<uint, access::read>   candidates1 [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant VoronoiSite *sites [[buffer(0)]],
    constant float &inv_scale_sq [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = float2(gid);

    uint candIds[8];
    uint2 outSize = uint2(width, height);
    loadCandidateIds(candidates0, candidates1, gid, outSize, candIds);

    // Compute logits and weights (skip inactive sites)
    float logits[8];
    float max_logit = -INFINITY;

    for (uint i = 0; i < 8; ++i) {
        uint idx = candIds[i];
        if (idx >= siteCount) {
            logits[i] = -INFINITY;
            continue;
        }
        VoronoiSite site = sites[idx];
        // Skip inactive sites (marked with negative position)
        if (site.position.x < 0.0f) {
            logits[i] = -INFINITY;
            continue;
        }
        float tau = exp(site.log_tau);
        float dmix2 = voronoi_dmix2(site, uv, inv_scale_sq);
        logits[i] = -tau * dmix2;
        max_logit = max(max_logit, logits[i]);
    }

    // Early exit if all candidates are invalid/inactive
    if (max_logit == -INFINITY) {
        output.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);  // Black pixel
        return;
    }

    // Softmax
    float weights[8];
    float sum_w = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        weights[i] = exp(logits[i] - max_logit);
        sum_w += weights[i];
    }
    float inv_sum = 1.0f / max(sum_w, 1e-8f);

    // Blend colors (skip inactive sites to avoid NaN)
    float3 color = float3(0.0f);
    for (uint i = 0; i < 8; ++i) {
        float w = weights[i] * inv_sum;
        if (!isnan(w) && !isinf(w) && candIds[i] < siteCount && sites[candIds[i]].position.x >= 0.0f) {
            color += w * sites[candIds[i]].color;
        }
    }

    output.write(float4(color, 1.0f), gid);


}

kernel void renderVoronoiColoring(
    texture2d<uint, access::read>   candidates0 [[texture(0)]],
    texture2d<uint, access::read>   candidates1 [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant VoronoiSite *sites [[buffer(0)]],
    constant float &inv_scale_sq [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = float2(gid);

    uint candIds[8];
    uint2 outSize = uint2(width, height);
    loadCandidateIds(candidates0, candidates1, gid, outSize, candIds);

    // Compute softmax logits (same as actual rendering)
    float logits[8];
    float max_logit = -INFINITY;

    for (uint i = 0; i < 8; ++i) {
        uint idx = candIds[i];
        if (idx >= siteCount) {
            logits[i] = -INFINITY;
            continue;
        }
        VoronoiSite site = sites[idx];
        // Skip inactive sites (marked with negative position)
        if (site.position.x < 0.0f) {
            logits[i] = -INFINITY;
            continue;
        }
        float tau = exp(site.log_tau);
        float dmix2 = voronoi_dmix2(site, uv, inv_scale_sq);
        logits[i] = -tau * dmix2;
        max_logit = max(max_logit, logits[i]);
    }

    // Softmax normalization
    float weights[8];
    float sum_w = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        weights[i] = exp(logits[i] - max_logit);
        sum_w += weights[i];
    }
    float inv_sum = 1.0f / max(sum_w, 1e-8f);

    // Blend hashed ID colors using softmax weights
    float3 color = float3(0.0f);
    for (uint i = 0; i < 8; ++i) {
        float w = weights[i] * inv_sum;
        if (!isnan(w) && !isinf(w) && candIds[i] < siteCount && sites[candIds[i]].position.x >= 0.0f) {
            float3 id_color = hashColor(candIds[i]);
            color += w * id_color;
        }
    }

    output.write(float4(color, 1.0f), gid);
}

kernel void renderCentroidsTauHeatmap(
    texture2d<uint, access::read>   candidates0 [[texture(0)]],
    texture2d<uint, access::read>   candidates1 [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant VoronoiSite *sites [[buffer(0)]],
    constant TauHeatmapParams &params [[buffer(1)]],
    constant uint &siteCount [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = float2(gid);

    uint candIds[8];
    uint2 outSize = uint2(width, height);
    loadCandidateIds(candidates0, candidates1, gid, outSize, candIds);

    float radius = params.dotRadius;
    float3 accColor = float3(0.0f);
    float accAlpha = 0.0f;

    for (uint i = 0; i < 8; ++i) {
        uint idx = candIds[i];
        if (idx >= siteCount) continue;
        VoronoiSite site = sites[idx];
        if (site.position.x < 0.0f) continue;

        float dist = length(uv - site.position);
        float alpha = 1.0f - smoothstep(radius - 1.0f, radius + 1.0f, dist);
        if (alpha <= 0.0f) continue;

        // Tau color: blue (min) -> white (mean) -> red (max)
        float tau = exp(site.log_tau);
        float3 color;
        if (tau <= params.meanTau) {
            float t = (params.meanTau > params.minTau)
                ? (tau - params.minTau) / (params.meanTau - params.minTau)
                : 0.5f;
            color = mix(float3(0.0f, 0.0f, 1.0f), float3(1.0f, 1.0f, 1.0f), pow(clamp(t, 0.0f, 1.0f), 0.2f));
        } else {
            float t = (params.maxTau > params.meanTau)
                ? (tau - params.meanTau) / (params.maxTau - params.meanTau)
                : 0.5f;
            color = mix(float3(1.0f, 1.0f, 1.0f), float3(1.0f, 0.0f, 0.0f), pow(clamp(t, 0.0f, 1.0f), 0.2f));
        }

        float blend = alpha * (1.0f - accAlpha);
        accColor += color * blend;
        accAlpha += blend;
    }

    float3 bgColor = float3(0.05f);
    float3 finalColor = accColor + bgColor * (1.0f - accAlpha);
    output.write(float4(finalColor, 1.0f), gid);
}
