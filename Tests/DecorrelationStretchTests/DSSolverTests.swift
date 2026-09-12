import XCTest
import simd
@testable import DecorrelationStretch

/// Deterministic normal-ish generator so failures are reproducible.
private struct LCG {
    var state: UInt64
    mutating func next() -> Double {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53)
    }
    mutating func gaussian() -> Double {
        // Irwin-Hall: sum of 12 uniforms, mean 6, variance 1.
        var acc = 0.0
        for _ in 0..<12 { acc += next() }
        return acc - 6
    }
}

final class DSSolverTests: XCTestCase {

    func testEigenDecompositionReconstructsMatrix() {
        let v: [Double] = [4.0, 1.2, 0.7,
                           1.2, 2.5, -0.3,
                           0.7, -0.3, 1.1]
        let (lambda, q) = DSSolver.eigenSymmetric3x3(v)

        XCTAssertGreaterThanOrEqual(lambda.x, lambda.y)
        XCTAssertGreaterThanOrEqual(lambda.y, lambda.z)

        // Q must be orthonormal.
        for i in 0..<3 {
            for j in 0..<3 {
                var dot = 0.0
                for k in 0..<3 { dot += q[k * 3 + i] * q[k * 3 + j] }
                XCTAssertEqual(dot, i == j ? 1 : 0, accuracy: 1e-12)
            }
        }

        // QΛQᵀ must reproduce V.
        for r in 0..<3 {
            for c in 0..<3 {
                var acc = 0.0
                for k in 0..<3 { acc += q[r * 3 + k] * lambda[k] * q[c * 3 + k] }
                XCTAssertEqual(acc, v[r * 3 + c], accuracy: 1e-12)
            }
        }
    }

    /// The defining property: after the transform, channels are uncorrelated and each
    /// carries the requested variance.
    func testTransformDecorrelatesAndHitsTargetSigma() {
        var rng = LCG(state: 0xD57D0E7C4)
        var source: [SIMD3<Double>] = []
        for _ in 0..<20_000 {
            // Strongly correlated channels, as in a real photograph.
            let l = rng.gaussian() * 0.20 + 0.5
            let a = rng.gaussian() * 0.02
            let b = rng.gaussian() * 0.015
            let c0: Double = l + a
            let c1: Double = l + 0.3 * a + b
            let c2: Double = l - 0.2 * a - 0.5 * b
            source.append(SIMD3<Double>(c0, c1, c2))
        }

        var moments = DSMoments(center: SIMD3(0.5, 0.5, 0.5))
        for p in source { moments.add(p) }

        let targetSigma: Float = 0.12
        let t = DSSolver.makeTransform(
            moments: moments,
            target: .uniform(fraction: targetSigma),
            nominalScale: SIMD3(1, 1, 1)
        )
        XCTAssertFalse(t.wasRegularized)

        var out = DSMoments(center: SIMD3(0.5, 0.5, 0.5))
        for p in source {
            let f = SIMD3<Float>(Float(p.x), Float(p.y), Float(p.z))
            let y = t.matrix * f + t.offset
            out.add(SIMD3<Double>(Double(y.x), Double(y.y), Double(y.z)))
        }

        let cov = out.covariance
        let expectedVariance = Double(targetSigma * targetSigma)
        for i in 0..<3 {
            XCTAssertEqual(cov[i * 3 + i], expectedVariance, accuracy: expectedVariance * 0.02,
                           "axis \(i) variance")
        }
        // Off-diagonals must vanish: that is the "decorrelation" in the name.
        for (r, c) in [(0, 1), (0, 2), (1, 2)] {
            XCTAssertEqual(cov[r * 3 + c], 0, accuracy: expectedVariance * 0.02,
                           "covariance \(r),\(c)")
        }

        // Mean must be preserved.
        let inMean = moments.mean, outMean = out.mean
        for i in 0..<3 {
            XCTAssertEqual(outMean[i], inMean[i], accuracy: 1e-4)
        }
    }

    func testPreserveOriginalKeepsPerAxisSpread() {
        var rng = LCG(state: 99)
        var moments = DSMoments(center: SIMD3(0.5, 0.5, 0.5))
        var source: [SIMD3<Double>] = []
        for _ in 0..<20_000 {
            let l = rng.gaussian() * 0.2 + 0.5
            let d0: Double = l + rng.gaussian() * 0.03
            let d1: Double = l + rng.gaussian() * 0.02
            let p = SIMD3<Double>(d0, d1, l)
            source.append(p)
            moments.add(p)
        }

        let inCov = moments.covariance
        let t = DSSolver.makeTransform(moments: moments, target: .preserveOriginal,
                                       nominalScale: SIMD3(1, 1, 1))

        var out = DSMoments(center: SIMD3(0.5, 0.5, 0.5))
        for p in source {
            let f = SIMD3<Float>(Float(p.x), Float(p.y), Float(p.z))
            let y = t.matrix * f + t.offset
            out.add(SIMD3<Double>(Double(y.x), Double(y.y), Double(y.z)))
        }
        let outCov = out.covariance
        for i in 0..<3 {
            XCTAssertEqual(outCov[i * 3 + i], inCov[i * 3 + i],
                           accuracy: inCov[i * 3 + i] * 0.02, "axis \(i)")
        }
    }

    /// A linearly dependent channel makes the covariance singular. The transform must
    /// degrade gracefully instead of producing infinities.
    func testDegenerateChannelIsRegularizedNotExploded() {
        var rng = LCG(state: 7)
        var moments = DSMoments(center: .zero)
        for _ in 0..<5_000 {
            let x = rng.gaussian() * 0.1 + 0.5
            let y = rng.gaussian() * 0.1 + 0.5
            let dep: Double = 0.5 * x + 0.5 * y           // third plane is dependent
            moments.add(SIMD3<Double>(x, y, dep))
        }

        let t = DSSolver.makeTransform(moments: moments, target: .uniform(fraction: 0.1),
                                       nominalScale: SIMD3(1, 1, 1))
        XCTAssertTrue(t.wasRegularized, "singular covariance should be flagged")
        for c in 0..<3 {
            for r in 0..<3 {
                XCTAssertTrue(t.matrix[c][r].isFinite, "matrix[\(c)][\(r)] not finite")
            }
            XCTAssertTrue(t.offset[c].isFinite)
        }
    }

    /// Centered accumulation must agree with a direct two-pass covariance.
    func testCenteredMomentsMatchTwoPassCovariance() {
        var rng = LCG(state: 4242)
        var points: [SIMD3<Double>] = []
        for _ in 0..<10_000 {
            let q0: Double = 50 + rng.gaussian() * 2
            let q1: Double = 12 + rng.gaussian() * 1.5
            let q2: Double = -7 + rng.gaussian() * 0.8
            points.append(SIMD3<Double>(q0, q1, q2))
        }

        var moments = DSMoments(center: SIMD3(50, 12, -7))
        for p in points { moments.add(p) }

        let n = Double(points.count)
        var mean = SIMD3<Double>.zero
        for p in points { mean += p }
        mean /= n
        var reference = [Double](repeating: 0, count: 9)
        for p in points {
            let d = p - mean
            for r in 0..<3 { for c in 0..<3 { reference[r * 3 + c] += d[r] * d[c] } }
        }
        for i in 0..<9 { reference[i] /= (n - 1) }

        let cov = moments.covariance
        for i in 0..<9 {
            XCTAssertEqual(cov[i], reference[i], accuracy: 1e-9, "entry \(i)")
        }
    }
}
