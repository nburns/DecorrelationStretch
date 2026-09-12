import simd

/// How much variance the stretch should put on each axis of the working space.
public enum DSTargetVariance: Equatable, Sendable {
    /// `Σ_T = Σ`. Decorrelates without changing per-axis spread — MATLAB's
    /// `decorrstretch` default. Subtle; useful when you want the colors to stay
    /// broadly believable.
    case preserveOriginal
    /// `Σ_T = fraction · nominalScale · I`. Every axis is driven to the same
    /// standard deviation, which is what produces the dramatic, heavily saturated look.
    /// `fraction` is a proportion of the space's nominal extent; 0.059 corresponds to a
    /// standard deviation of 15 in 0...255 units.
    case uniform(fraction: Float)
}

/// The output of the analysis pass: an affine map applied to every pixel in the
/// weighted working space.
public struct DSTransform: Equatable, Sendable {
    public var matrix: simd_float3x3
    public var offset: SIMD3<Float>

    /// Eigenvalues of the measured covariance, descending. Near-zero values mean the
    /// color planes were close to linearly dependent.
    public var eigenvalues: SIMD3<Float>
    /// True when at least one eigenvalue fell below the conditioning tolerance and was
    /// clamped. The stretch still runs, but that axis carried no usable signal.
    public var wasRegularized: Bool
    /// Number of pixels the statistics were measured over.
    public var sampleCount: Int

    public static let identity = DSTransform(
        matrix: matrix_identity_float3x3,
        offset: .zero,
        eigenvalues: .one,
        wasRegularized: false,
        sampleCount: 0
    )

    public func mix(with other: DSTransform, t: Float) -> DSTransform {
        var result = other
        result.matrix = simd_float3x3(
            simd_mix(matrix.columns.0, other.matrix.columns.0, SIMD3(repeating: t)),
            simd_mix(matrix.columns.1, other.matrix.columns.1, SIMD3(repeating: t)),
            simd_mix(matrix.columns.2, other.matrix.columns.2, SIMD3(repeating: t))
        )
        result.offset = simd_mix(offset, other.offset, SIMD3(repeating: t))
        return result
    }
}
