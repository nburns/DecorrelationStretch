import simd

/// The base color space in which the covariance is measured and the stretch applied.
///
/// Choice of base space is the single biggest lever on the result: it decides which
/// directions in color space are the "short axes" that decorrelation stretch inflates.
public enum DSColorSpaceFamily: Int32, Sendable, CaseIterable {
    case rgb = 0
    case yuv = 1
    case lab = 2
}

/// A base space plus per-axis multipliers applied before the covariance is measured
/// and undone after the stretch.
///
/// Scaling an axis
/// changes the covariance, which changes the eigenvectors, which changes the rotation
/// the stretch is performed in — so the weights genuinely alter the output rather than
/// merely rescaling it.
///
/// Because the target sigmas are imposed in the *weighted* space and then divided back
/// out, a weight below 1 also raises that axis's final standard deviation. The net
/// perceptual effect is empirical; these are knobs to turn, not a formula to predict.
public struct DSColorSpace: Hashable, Sendable {
    public var family: DSColorSpaceFamily
    /// Multipliers on the three axes of `family`, in that space's own units.
    public var weights: SIMD3<Float>

    public init(family: DSColorSpaceFamily, weights: SIMD3<Float> = .one) {
        self.family = family
        self.weights = weights
    }

    /// Nominal per-axis extent of the unweighted space, used to interpret
    /// `DSTargetVariance.uniform` in units that make sense for each space.
    public var nominalScale: SIMD3<Float> {
        let base: SIMD3<Float>
        switch family {
        case .rgb: base = SIMD3(1, 1, 1)
        case .yuv: base = SIMD3(1, 0.872, 1.230)   // Y in 0...1, U/V full BT.601 swing
        case .lab: base = SIMD3(100, 128, 128)
        }
        return base * weights
    }

    /// A reasonable point to accumulate moments about before the true mean is known.
    /// Only affects conditioning, never the result.
    public var nominalCenter: SIMD3<Float> {
        let base: SIMD3<Float>
        switch family {
        case .rgb: base = SIMD3(0.5, 0.5, 0.5)
        case .yuv: base = SIMD3(0.5, 0, 0)
        case .lab: base = SIMD3(50, 0, 0)
        }
        return base * weights
    }
}

public extension DSColorSpace {
    static let rgb = DSColorSpace(family: .rgb)
    static let yuv = DSColorSpace(family: .yuv)
    static let lab = DSColorSpace(family: .lab)

    /// Luminance held back so the stretch spends its range on chroma. A good
    /// general-purpose starting point.
    static let chromaBoost = DSColorSpace(family: .yuv, weights: SIMD3(0.5, 1.5, 1.5))

    /// Weighted toward the a* (green-red) axis, for red and ochre pigment.
    static let redEmphasis = DSColorSpace(family: .lab, weights: SIMD3(0.6, 1.6, 0.8))

    /// Weighted toward the b* (blue-yellow) axis, for faint yellows.
    static let yellowEmphasis = DSColorSpace(family: .lab, weights: SIMD3(0.6, 0.8, 1.6))

    /// Weighted toward L*, for dark pigment on dark rock where the signal is tonal.
    static let tonalEmphasis = DSColorSpace(family: .lab, weights: SIMD3(1.5, 0.7, 0.7))
}
