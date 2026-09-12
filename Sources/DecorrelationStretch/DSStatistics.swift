import simd

/// First and second moments of the pixel cloud in the weighted working space,
/// accumulated *about a reference center* rather than about the origin.
///
/// Forming the covariance from origin-centered raw moments is the numerically weak
/// step in the classic algorithm — `Σx²` and `nµ²` are nearly equal, so subtracting
/// them cancels most of the significant digits. Accumulating about a center close to
/// the true mean (the previous frame's mean, or the midpoint of the space) keeps both
/// terms small and the cancellation never happens. The center is exact bookkeeping,
/// not an approximation: any center gives the same covariance in exact arithmetic.
public struct DSMoments: Sendable, Equatable {
    /// The point the moments are measured about.
    public var center: SIMD3<Double>
    public var count: Double
    /// `Σ(x − center)`
    public var sum: SIMD3<Double>
    /// Upper triangle of `Σ(x − center)(x − center)ᵀ`: xx, xy, xz, yy, yz, zz.
    public var sumOuter: SIMD6

    public struct SIMD6: Sendable, Equatable {
        public var xx, xy, xz, yy, yz, zz: Double
        public init(xx: Double = 0, xy: Double = 0, xz: Double = 0,
                    yy: Double = 0, yz: Double = 0, zz: Double = 0) {
            self.xx = xx; self.xy = xy; self.xz = xz
            self.yy = yy; self.yz = yz; self.zz = zz
        }
        public static func + (a: SIMD6, b: SIMD6) -> SIMD6 {
            SIMD6(xx: a.xx + b.xx, xy: a.xy + b.xy, xz: a.xz + b.xz,
                  yy: a.yy + b.yy, yz: a.yz + b.yz, zz: a.zz + b.zz)
        }
    }

    public init(center: SIMD3<Double> = .zero) {
        self.center = center
        self.count = 0
        self.sum = .zero
        self.sumOuter = SIMD6()
    }

    public mutating func add(_ value: SIMD3<Double>) {
        let d = value - center
        count += 1
        sum += d
        sumOuter = sumOuter + SIMD6(xx: d.x * d.x, xy: d.x * d.y, xz: d.x * d.z,
                                    yy: d.y * d.y, yz: d.y * d.z, zz: d.z * d.z)
    }

    /// Combines partials that share the same center.
    public mutating func combine(_ other: DSMoments) {
        precondition(other.center == center, "moments must share a center to combine")
        count += other.count
        sum += other.sum
        sumOuter = sumOuter + other.sumOuter
    }

    public var mean: SIMD3<Double> {
        count > 0 ? center + sum / count : center
    }

    /// Sample covariance `1/(n−1) · Σ(x−µ)(x−µ)ᵀ`, in row-major order.
    public var covariance: [Double] {
        guard count > 1 else { return [1, 0, 0, 0, 1, 0, 0, 0, 1] }
        let d = mean - center
        let n = count
        let k = n - 1
        let xx = (sumOuter.xx - n * d.x * d.x) / k
        let xy = (sumOuter.xy - n * d.x * d.y) / k
        let xz = (sumOuter.xz - n * d.x * d.z) / k
        let yy = (sumOuter.yy - n * d.y * d.y) / k
        let yz = (sumOuter.yz - n * d.y * d.z) / k
        let zz = (sumOuter.zz - n * d.z * d.z) / k
        return [xx, xy, xz,
                xy, yy, yz,
                xz, yz, zz]
    }
}
