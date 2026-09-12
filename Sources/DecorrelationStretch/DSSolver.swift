import simd

/// Builds the decorrelation-stretch transform from measured moments.
public enum DSSolver {

    /// Eigenvalues below `conditioningTolerance · λmax` are treated as carrying no
    /// signal and clamped, rather than driving `Λ^(-1/2)` to infinity.
    public static let defaultConditioningTolerance: Double = 1e-6

    /// `V = QΛQᵀ` for a symmetric 3×3 matrix, by cyclic Jacobi rotation.
    ///
    /// Returns eigenvalues in descending order with the matching eigenvectors as the
    /// columns of `vectors`. Jacobi is used rather than a closed-form cubic solve
    /// because it stays accurate when eigenvalues are clustered, which is the normal
    /// case here — highly correlated channels are exactly what this filter is for.
    public static func eigenSymmetric3x3(_ m: [Double]) -> (values: SIMD3<Double>, vectors: [Double]) {
        precondition(m.count == 9)
        var a = m
        var v: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        func idx(_ r: Int, _ c: Int) -> Int { r * 3 + c }

        for _ in 0..<24 {
            let off = a[idx(0, 1)] * a[idx(0, 1)]
                    + a[idx(0, 2)] * a[idx(0, 2)]
                    + a[idx(1, 2)] * a[idx(1, 2)]
            if off <= 1e-30 { break }

            for (p, q) in [(0, 1), (0, 2), (1, 2)] {
                let apq = a[idx(p, q)]
                if abs(apq) <= 1e-300 { continue }

                let theta = (a[idx(q, q)] - a[idx(p, p)]) / (2 * apq)
                let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                let c = 1 / (t * t + 1).squareRoot()
                let s = t * c

                let app = a[idx(p, p)], aqq = a[idx(q, q)]
                a[idx(p, p)] = c * c * app - 2 * s * c * apq + s * s * aqq
                a[idx(q, q)] = s * s * app + 2 * s * c * apq + c * c * aqq
                a[idx(p, q)] = 0
                a[idx(q, p)] = 0

                for k in 0..<3 where k != p && k != q {
                    let akp = a[idx(k, p)], akq = a[idx(k, q)]
                    a[idx(k, p)] = c * akp - s * akq
                    a[idx(p, k)] = a[idx(k, p)]
                    a[idx(k, q)] = s * akp + c * akq
                    a[idx(q, k)] = a[idx(k, q)]
                }
                for k in 0..<3 {
                    let vkp = v[idx(k, p)], vkq = v[idx(k, q)]
                    v[idx(k, p)] = c * vkp - s * vkq
                    v[idx(k, q)] = s * vkp + c * vkq
                }
            }
        }

        var order = [0, 1, 2]
        order.sort { a[idx($0, $0)] > a[idx($1, $1)] }

        let values = SIMD3<Double>(a[idx(order[0], order[0])],
                                   a[idx(order[1], order[1])],
                                   a[idx(order[2], order[2])])
        var vectors = [Double](repeating: 0, count: 9)
        for (newCol, oldCol) in order.enumerated() {
            for r in 0..<3 { vectors[idx(r, newCol)] = v[idx(r, oldCol)] }
        }
        return (values, vectors)
    }

    /// Assembles `M = Σ_T · Q · Λ^(-1/2) · Qᵀ` and the offset `µ_T − Mµ`.
    ///
    /// `Q Λ^(-1/2) Qᵀ` is the inverse matrix square root of the covariance, so this is
    /// ZCA whitening followed by a rescale to the target sigmas — the whole of
    /// decorrelation stretch collapses into one affine map.
    public static func makeTransform(
        moments: DSMoments,
        target: DSTargetVariance,
        nominalScale: SIMD3<Float>,
        conditioningTolerance: Double = defaultConditioningTolerance
    ) -> DSTransform {
        guard moments.count > 1 else { return .identity }

        let cov = moments.covariance
        let (lambda, q) = eigenSymmetric3x3(cov)

        let lambdaMax = max(lambda.x, 1e-300)
        let floorValue = lambdaMax * conditioningTolerance
        var regularized = false
        var invSqrt = SIMD3<Double>()
        for i in 0..<3 {
            var l = lambda[i]
            if !(l > floorValue) {
                l = floorValue
                regularized = true
            }
            invSqrt[i] = 1 / l.squareRoot()
        }

        let sigmaTarget: SIMD3<Double>
        switch target {
        case .preserveOriginal:
            sigmaTarget = SIMD3(max(cov[0], 0).squareRoot(),
                                max(cov[4], 0).squareRoot(),
                                max(cov[8], 0).squareRoot())
        case .uniform(let fraction):
            sigmaTarget = SIMD3(Double(fraction * nominalScale.x),
                                Double(fraction * nominalScale.y),
                                Double(fraction * nominalScale.z))
        }

        // M = Σ_T · (Q Λ^(-1/2) Qᵀ)
        var m = [Double](repeating: 0, count: 9)
        for r in 0..<3 {
            for c in 0..<3 {
                var acc = 0.0
                for k in 0..<3 {
                    acc += q[r * 3 + k] * invSqrt[k] * q[c * 3 + k]
                }
                m[r * 3 + c] = sigmaTarget[r] * acc
            }
        }

        let mu = moments.mean
        var offset = SIMD3<Double>()
        for r in 0..<3 {
            offset[r] = mu[r] - (m[r * 3 + 0] * mu.x + m[r * 3 + 1] * mu.y + m[r * 3 + 2] * mu.z)
        }

        // simd_float3x3 is column-major; m is row-major.
        let matrix = simd_float3x3(
            SIMD3<Float>(Float(m[0]), Float(m[3]), Float(m[6])),
            SIMD3<Float>(Float(m[1]), Float(m[4]), Float(m[7])),
            SIMD3<Float>(Float(m[2]), Float(m[5]), Float(m[8]))
        )

        return DSTransform(
            matrix: matrix,
            offset: SIMD3<Float>(Float(offset.x), Float(offset.y), Float(offset.z)),
            eigenvalues: SIMD3<Float>(Float(lambda.x), Float(lambda.y), Float(lambda.z)),
            wasRegularized: regularized,
            sampleCount: Int(moments.count)
        )
    }
}
