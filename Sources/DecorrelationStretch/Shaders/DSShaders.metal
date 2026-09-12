#include <metal_stdlib>
using namespace metal;

// Must match DSColorSpaceFamily in Swift.
constant int DS_FAMILY_RGB = 0;
constant int DS_FAMILY_YUV = 1;
constant int DS_FAMILY_LAB = 2;

constant uint DS_THREADS_PER_GROUP = 256;
constant uint DS_ACCUMULATOR_COUNT = 10;   // n, sx, sy, sz, xx, xy, xz, yy, yz, zz

struct DSSpaceParams {
    float3 weights;
    int family;
};

struct DSApplyParams {
    float3x3 matrix;
    float3 offset;
    float amount;
};

struct DSStatsParams {
    float3 center;
    uint originX, originY, width, height;
    uint stride;
};

// MARK: - sRGB transfer

static inline float3 ds_srgb_to_linear(float3 c) {
    float3 lo = c * (1.0f / 12.92f);
    float3 hi = pow(max(c + 0.055f, 0.0f) * (1.0f / 1.055f), 2.4f);
    return select(lo, hi, c > 0.04045f);
}

static inline float3 ds_linear_to_srgb(float3 c) {
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(max(c, 0.0f), 1.0f / 2.4f) - 0.055f;
    return select(lo, hi, c > 0.0031308f);
}

// MARK: - CIELAB (D65)

constant float3 DS_D65 = float3(0.95047f, 1.0f, 1.08883f);
constant float DS_LAB_EPS = 216.0f / 24389.0f;
constant float DS_LAB_KAPPA = 24389.0f / 27.0f;

static inline float ds_lab_f(float t) {
    return t > DS_LAB_EPS ? pow(max(t, 0.0f), 1.0f / 3.0f) : (DS_LAB_KAPPA * t + 16.0f) / 116.0f;
}

static inline float ds_lab_f_inv(float t) {
    float t3 = t * t * t;
    return t3 > DS_LAB_EPS ? t3 : (116.0f * t - 16.0f) / DS_LAB_KAPPA;
}

static inline float3 ds_rgb_to_lab(float3 rgb) {
    float3 l = ds_srgb_to_linear(rgb);
    float3 xyz = float3(
        dot(l, float3(0.4124564f, 0.3575761f, 0.1804375f)),
        dot(l, float3(0.2126729f, 0.7151522f, 0.0721750f)),
        dot(l, float3(0.0193339f, 0.1191920f, 0.9503041f))
    ) / DS_D65;
    float fx = ds_lab_f(xyz.x), fy = ds_lab_f(xyz.y), fz = ds_lab_f(xyz.z);
    return float3(116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz));
}

static inline float3 ds_lab_to_rgb(float3 lab) {
    float fy = (lab.x + 16.0f) / 116.0f;
    float fx = fy + lab.y / 500.0f;
    float fz = fy - lab.z / 200.0f;
    float3 xyz = float3(ds_lab_f_inv(fx), ds_lab_f_inv(fy), ds_lab_f_inv(fz)) * DS_D65;
    float3 lin = float3(
        dot(xyz, float3( 3.2404542f, -1.5371385f, -0.4985314f)),
        dot(xyz, float3(-0.9692660f,  1.8760108f,  0.0415560f)),
        dot(xyz, float3( 0.0556434f, -0.2040259f,  1.0572252f))
    );
    return ds_linear_to_srgb(lin);
}

// MARK: - YUV (BT.601)

static inline float3 ds_rgb_to_yuv(float3 rgb) {
    return float3(
        dot(rgb, float3( 0.29900f,  0.58700f,  0.11400f)),
        dot(rgb, float3(-0.14713f, -0.28886f,  0.43600f)),
        dot(rgb, float3( 0.61500f, -0.51499f, -0.10001f))
    );
}

static inline float3 ds_yuv_to_rgb(float3 yuv) {
    return float3(
        yuv.x + 1.13983f * yuv.z,
        yuv.x - 0.39465f * yuv.y - 0.58060f * yuv.z,
        yuv.x + 2.03211f * yuv.y
    );
}

// MARK: - Working space

/// RGB into the weighted working space the covariance is measured in.
static inline float3 ds_to_space(float3 rgb, constant DSSpaceParams &sp) {
    float3 v;
    switch (sp.family) {
        case DS_FAMILY_YUV: v = ds_rgb_to_yuv(rgb); break;
        case DS_FAMILY_LAB: v = ds_rgb_to_lab(rgb); break;
        default:            v = rgb;                break;
    }
    return v * sp.weights;
}

static inline float3 ds_from_space(float3 w, constant DSSpaceParams &sp) {
    float3 v = w / sp.weights;
    switch (sp.family) {
        case DS_FAMILY_YUV: return ds_yuv_to_rgb(v);
        case DS_FAMILY_LAB: return ds_lab_to_rgb(v);
        default:            return v;
    }
}

// MARK: - Analysis pass

/// Accumulates first and second moments about `center` over the region of interest.
/// Each threadgroup tree-reduces into one 10-float record; the host sums the records.
kernel void ds_stats(texture2d<float, access::read> src      [[texture(0)]],
                     device float                  *partials [[buffer(0)]],
                     constant DSSpaceParams        &sp       [[buffer(1)]],
                     constant DSStatsParams        &st       [[buffer(2)]],
                     uint2  gid        [[thread_position_in_grid]],
                     uint   lane       [[thread_index_in_threadgroup]],
                     uint2  groupId    [[threadgroup_position_in_grid]],
                     uint2  groupCount [[threadgroups_per_grid]])
{
    threadgroup float acc[DS_THREADS_PER_GROUP][DS_ACCUMULATOR_COUNT];

    for (uint i = 0; i < DS_ACCUMULATOR_COUNT; ++i) { acc[lane][i] = 0.0f; }

    uint2 coord = uint2(st.originX, st.originY) + gid * st.stride;
    bool inside = gid.x * st.stride < st.width &&
                  gid.y * st.stride < st.height &&
                  coord.x < src.get_width() && coord.y < src.get_height();

    if (inside) {
        float3 w = ds_to_space(src.read(coord).rgb, sp) - st.center;
        acc[lane][0] = 1.0f;
        acc[lane][1] = w.x;  acc[lane][2] = w.y;  acc[lane][3] = w.z;
        acc[lane][4] = w.x * w.x;
        acc[lane][5] = w.x * w.y;
        acc[lane][6] = w.x * w.z;
        acc[lane][7] = w.y * w.y;
        acc[lane][8] = w.y * w.z;
        acc[lane][9] = w.z * w.z;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = DS_THREADS_PER_GROUP / 2; s > 0; s >>= 1) {
        if (lane < s) {
            for (uint i = 0; i < DS_ACCUMULATOR_COUNT; ++i) {
                acc[lane][i] += acc[lane + s][i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        uint slot = (groupId.y * groupCount.x + groupId.x) * DS_ACCUMULATOR_COUNT;
        for (uint i = 0; i < DS_ACCUMULATOR_COUNT; ++i) {
            partials[slot + i] = acc[0][i];
        }
    }
}

// MARK: - Apply pass

/// One affine multiply per pixel in the working space. This is the whole filter.
kernel void ds_apply(texture2d<float, access::read>  src [[texture(0)]],
                     texture2d<float, access::write> dst [[texture(1)]],
                     constant DSSpaceParams          &sp [[buffer(0)]],
                     constant DSApplyParams          &ap [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) { return; }

    float4 texel = src.read(gid);
    float3 stretched = ds_from_space(ap.matrix * ds_to_space(texel.rgb, sp) + ap.offset, sp);
    float3 rgb = clamp(mix(texel.rgb, stretched, ap.amount), 0.0f, 1.0f);
    dst.write(float4(rgb, texel.a), gid);
}
