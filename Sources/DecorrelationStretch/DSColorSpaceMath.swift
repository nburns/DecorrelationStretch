import simd

/// CPU implementations of the same conversions performed in DSShaders.metal.
///
/// These exist for two reasons: the coefficient-fitting tools need to work in the base
/// space without a GPU round trip, and having an independent implementation lets the
/// test suite check the shader against it. `DSColorSpaceMathTests` asserts the two agree;
/// any change here must be mirrored in the shader and vice versa.
public enum DSColorSpaceMath {

    // MARK: sRGB transfer

    public static func srgbToLinear(_ c: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z))
    }

    public static func linearToSrgb(_ c: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(linearToSrgb(c.x), linearToSrgb(c.y), linearToSrgb(c.z))
    }

    static func srgbToLinear(_ c: Float) -> Float {
        c > 0.04045 ? pow(max(c + 0.055, 0) / 1.055, 2.4) : c / 12.92
    }

    static func linearToSrgb(_ c: Float) -> Float {
        c > 0.0031308 ? 1.055 * pow(max(c, 0), 1 / 2.4) - 0.055 : c * 12.92
    }

    // MARK: CIELAB (D65)

    static let d65 = SIMD3<Float>(0.95047, 1.0, 1.08883)
    static let labEpsilon: Float = 216.0 / 24389.0
    static let labKappa: Float = 24389.0 / 27.0

    static func labF(_ t: Float) -> Float {
        t > labEpsilon ? pow(max(t, 0), 1.0 / 3.0) : (labKappa * t + 16) / 116
    }

    static func labFInverse(_ t: Float) -> Float {
        let t3 = t * t * t
        return t3 > labEpsilon ? t3 : (116 * t - 16) / labKappa
    }

    public static func rgbToLab(_ rgb: SIMD3<Float>) -> SIMD3<Float> {
        let l = srgbToLinear(rgb)
        let xyz = SIMD3<Float>(
            simd_dot(l, SIMD3(0.4124564, 0.3575761, 0.1804375)),
            simd_dot(l, SIMD3(0.2126729, 0.7151522, 0.0721750)),
            simd_dot(l, SIMD3(0.0193339, 0.1191920, 0.9503041))
        ) / d65
        let fx = labF(xyz.x), fy = labF(xyz.y), fz = labF(xyz.z)
        return SIMD3(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
    }

    public static func labToRgb(_ lab: SIMD3<Float>) -> SIMD3<Float> {
        let fy = (lab.x + 16) / 116
        let fx = fy + lab.y / 500
        let fz = fy - lab.z / 200
        let xyz = SIMD3<Float>(labFInverse(fx), labFInverse(fy), labFInverse(fz)) * d65
        let linear = SIMD3<Float>(
            simd_dot(xyz, SIMD3( 3.2404542, -1.5371385, -0.4985314)),
            simd_dot(xyz, SIMD3(-0.9692660,  1.8760108,  0.0415560)),
            simd_dot(xyz, SIMD3( 0.0556434, -0.2040259,  1.0572252))
        )
        return linearToSrgb(linear)
    }

    // MARK: YUV (BT.601)

    public static func rgbToYuv(_ rgb: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            simd_dot(rgb, SIMD3( 0.29900,  0.58700,  0.11400)),
            simd_dot(rgb, SIMD3(-0.14713, -0.28886,  0.43600)),
            simd_dot(rgb, SIMD3( 0.61500, -0.51499, -0.10001))
        )
    }

    public static func yuvToRgb(_ yuv: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            yuv.x + 1.13983 * yuv.z,
            yuv.x - 0.39465 * yuv.y - 0.58060 * yuv.z,
            yuv.x + 2.03211 * yuv.y
        )
    }

    // MARK: Working space

    /// RGB into the weighted working space, matching `ds_to_space` in the shader.
    public static func toSpace(_ rgb: SIMD3<Float>, _ space: DSColorSpace) -> SIMD3<Float> {
        let v: SIMD3<Float>
        switch space.family {
        case .rgb: v = rgb
        case .yuv: v = rgbToYuv(rgb)
        case .lab: v = rgbToLab(rgb)
        }
        return v * space.weights
    }

    /// Weighted working space back to RGB, matching `ds_from_space` in the shader.
    public static func fromSpace(_ w: SIMD3<Float>, _ space: DSColorSpace) -> SIMD3<Float> {
        let v = w / space.weights
        switch space.family {
        case .rgb: return v
        case .yuv: return yuvToRgb(v)
        case .lab: return labToRgb(v)
        }
    }
}
