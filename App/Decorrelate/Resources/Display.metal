#include <metal_stdlib>
using namespace metal;

struct DisplayVertex {
    float4 position [[position]];
    float2 uv;
};

struct DisplayUniforms {
    float2 scale;      // shrinks the quad to the aspect-fit rect
};

/// A quad covering exactly the letterboxed image rect, as a 4-vertex triangle strip.
///
/// Not a scaled fullscreen triangle: that shape extends to NDC +3, so scaling it shrinks
/// the rect without bounding the geometry, and everything between the image rect and the
/// triangle's edge samples uv > 1 and comes out as clamp-to-edge smear instead of
/// letterbox.
vertex DisplayVertex display_vertex(uint vid [[vertex_id]],
                                    constant DisplayUniforms &u [[buffer(0)]])
{
    float2 corner = float2((vid & 1) ? 1.0 : 0.0, (vid & 2) ? 1.0 : 0.0);
    DisplayVertex out;
    out.position = float4((corner * 2.0 - 1.0) * u.scale, 0.0, 1.0);
    out.uv = float2(corner.x, 1.0 - corner.y);
    return out;
}

fragment float4 display_fragment(DisplayVertex in [[stage_in]],
                                 texture2d<float> source [[texture(0)]],
                                 sampler smp [[sampler(0)]])
{
    return source.sample(smp, in.uv);
}
