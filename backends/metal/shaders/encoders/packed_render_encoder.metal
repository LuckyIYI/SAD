#include "../sad_common.metal"

kernel void renderVoronoiPacked(
    texture2d<uint, access::read>   candidates0 [[texture(0)]],
    texture2d<uint, access::read>   candidates1 [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant PackedInferenceSite *sites [[buffer(0)]],
    constant PackedSiteQuant &quant [[buffer(1)]],
    constant float &inv_scale_sq [[buffer(2)]],
    constant uint &siteCount [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;
    float2 dims = float2(float(width - 1), float(height - 1));

    float2 uv = float2(gid);

    uint candIds[8];
    uint2 outSize = uint2(width, height);
    loadCandidateIds(candidates0, candidates1, gid, outSize, candIds);

    float weights[8];
    float3 pred;
    if (!computePackedWeights(candIds, sites, quant, siteCount, uv, inv_scale_sq, dims, weights, pred)) {
        output.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);
        return;
    }

    output.write(float4(pred, 1.0f), gid);
}

kernel void renderVoronoiColoringPacked(
    texture2d<uint, access::read>   candidates0 [[texture(0)]],
    texture2d<uint, access::read>   candidates1 [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant PackedInferenceSite *sites [[buffer(0)]],
    constant PackedSiteQuant &quant [[buffer(1)]],
    constant float &inv_scale_sq [[buffer(2)]],
    constant uint &siteCount [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;
    float2 dims = float2(float(width - 1), float(height - 1));

    float2 uv = float2(gid);

    uint candIds[8];
    uint2 outSize = uint2(width, height);
    loadCandidateIds(candidates0, candidates1, gid, outSize, candIds);

    float weights[8];
    float3 pred;
    if (!computePackedWeights(candIds, sites, quant, siteCount, uv, inv_scale_sq, dims, weights, pred)) {
        output.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);
        return;
    }

    float3 color = float3(0.0f);
    for (uint i = 0; i < 8; ++i) {
        if (candIds[i] < siteCount) {
            float3 id_color = hashColor(candIds[i]);
            color += weights[i] * id_color;
        }
    }

    output.write(float4(color, 1.0f), gid);
}

kernel void renderCentroidsTauHeatmapPacked(
    texture2d<uint, access::read>   candidates0 [[texture(0)]],
    texture2d<uint, access::read>   candidates1 [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant PackedInferenceSite *sites [[buffer(0)]],
    constant PackedSiteQuant &quant [[buffer(1)]],
    constant TauHeatmapParams &params [[buffer(2)]],
    constant uint &siteCount [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height) return;
    float2 dims = float2(float(width - 1), float(height - 1));

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
        PackedInferenceSite packed = sites[idx];
        if (!packedActive(packed)) continue;
        VoronoiSite site = decodePackedSite(packed, dims, quant);

        float dist = length(uv - site.position);
        float alpha = 1.0f - smoothstep(radius - 1.0f, radius + 1.0f, dist);
        if (alpha <= 0.0f) continue;

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
