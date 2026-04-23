#include "sad_common.metal"

struct ViewerVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex ViewerVertexOut viewerVertex(uint vid [[vertex_id]]) {
    float2 pos;
    if (vid == 0) { pos = float2(-1.0, -1.0); }
    else if (vid == 1) { pos = float2( 1.0, -1.0); }
    else if (vid == 2) { pos = float2(-1.0,  1.0); }
    else { pos = float2( 1.0,  1.0); }

    ViewerVertexOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    return out;
}

fragment float4 viewerFragment(ViewerVertexOut in [[stage_in]],
                               texture2d<float> tex [[texture(0)]],
                               sampler samp [[sampler(0)]]) {
    float2 uv = in.uv;
    uv.y = 1.0 - uv.y;
    return tex.sample(samp, uv);
}
