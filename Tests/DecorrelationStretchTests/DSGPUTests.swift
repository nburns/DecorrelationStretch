import XCTest
import Metal
import simd
@testable import DecorrelationStretch

final class DSGPUTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "no Metal device available")
        queue = device.makeCommandQueue()
    }

    // MARK: - Helpers

    private func makeTexture(_ pixels: [SIMD4<Float>], width: Int, height: Int,
                             writable: Bool = false) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = writable ? [.shaderRead, .shaderWrite] : [.shaderRead]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        pixels.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: raw.baseAddress!,
                            bytesPerRow: width * MemoryLayout<SIMD4<Float>>.stride)
        }
        return texture
    }

    private func makeEmptyTexture(width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        return texture
    }

    private func readBack(_ texture: MTLTexture) -> [SIMD4<Float>] {
        var out = [SIMD4<Float>](repeating: .zero, count: texture.width * texture.height)
        out.withUnsafeMutableBytes { raw in
            texture.getBytes(raw.baseAddress!,
                             bytesPerRow: texture.width * MemoryLayout<SIMD4<Float>>.stride,
                             from: MTLRegionMake2D(0, 0, texture.width, texture.height),
                             mipmapLevel: 0)
        }
        return out
    }

    /// A synthetic rock-art panel: channels dominated by a shared luminance gradient,
    /// with a faint pigment field buried in them and independent per-channel sensor
    /// noise. The independent noise matters -- without it all three channels are
    /// `luma + k*pigment`, the pixel cloud collapses into a plane, and the covariance is
    /// genuinely singular rather than merely ill-conditioned.
    private struct Panel {
        var pixels: [SIMD4<Float>]
        var pigment: [Float]
        var width: Int
        var height: Int
    }

    private func hashLattice(_ ix: Int, _ iy: Int) -> Float {
        var h = UInt64(bitPattern: Int64(ix &* 73856093) ^ Int64(iy &* 19349663))
        h ^= h >> 33; h = h &* 0xff51afd7ed558ccd; h ^= h >> 33
        return Float(Double(h >> 11) / Double(1 << 53)) * 2 - 1
    }

    /// Smooth value noise on non-commensurate scales, so sub-sampling cannot alias it
    /// the way a pure sinusoid would.
    private func valueNoise(_ x: Float, _ y: Float) -> Float {
        let x0 = Int(x.rounded(.down)), y0 = Int(y.rounded(.down))
        let fx = x - Float(x0), fy = y - Float(y0)
        let sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy)
        let a = hashLattice(x0, y0), b = hashLattice(x0 + 1, y0)
        let c = hashLattice(x0, y0 + 1), d = hashLattice(x0 + 1, y0 + 1)
        return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy
    }

    private func makePanel(width: Int, height: Int) -> Panel {
        var pixels: [SIMD4<Float>] = []
        var pigmentField: [Float] = []
        pixels.reserveCapacity(width * height)
        pigmentField.reserveCapacity(width * height)

        var state: UInt64 = 0x9E3779B97F4A7C15
        func noise() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53)) - 0.5
        }

        for y in 0..<height {
            for x in 0..<width {
                let pigment = valueNoise(Float(x) / 13.0, Float(y) / 11.0)
                let luma = 0.45 + 0.18 * Float(x) / Float(width) + 0.05 * noise()
                let r = luma + 0.018 * pigment + 0.005 * noise()
                let g = luma - 0.005 * pigment + 0.005 * noise()
                let b = luma - 0.012 * pigment + 0.005 * noise()
                pixels.append(SIMD4(min(max(r, 0), 1), min(max(g, 0), 1), min(max(b, 0), 1), 1))
                pigmentField.append(pigment)
            }
        }
        return Panel(pixels: pixels, pigment: pigmentField, width: width, height: height)
    }

    private func syntheticPanel(width: Int, height: Int) -> [SIMD4<Float>] {
        makePanel(width: width, height: height).pixels
    }

    /// Mean chroma of the pixels where the pigment field is strongly positive minus
    /// those where it is strongly negative: how visible the pigment is.
    private func pigmentSeparation(_ buffer: [SIMD4<Float>], _ panel: Panel) -> Float {
        var high: Float = 0, low: Float = 0
        var highCount = 0, lowCount = 0
        for i in 0..<buffer.count {
            let p = buffer[i]
            let chroma = p.x - 0.5 * (p.y + p.z)
            if panel.pigment[i] > 0.5 { high += chroma; highCount += 1 }
            else if panel.pigment[i] < -0.5 { low += chroma; lowCount += 1 }
        }
        guard highCount > 0, lowCount > 0 else { return 0 }
        return abs(high / Float(highCount) - low / Float(lowCount))
    }

    // MARK: - Shader / CPU agreement

    /// The shader and the CPU conversions must agree, since the fitting tools use the
    /// CPU path to reason about images the GPU produced.
    func testShaderColorSpaceRoundTripMatchesIdentity() throws {
        let library = try DSLibrary.make(device: device)
        let renderer = try DSRenderer(device: device, library: library)

        var pixels: [SIMD4<Float>] = []
        for r in stride(from: Float(0.02), through: 0.98, by: 0.12) {
            for g in stride(from: Float(0.02), through: 0.98, by: 0.12) {
                for b in stride(from: Float(0.02), through: 0.98, by: 0.12) {
                    pixels.append(SIMD4(r, g, b, 1))
                }
            }
        }
        let width = pixels.count
        let source = try makeTexture(pixels, width: width, height: 1)

        for space in [DSColorSpace.rgb, .yuv, .lab, .chromaBoost, .redEmphasis] {
            let destination = try makeEmptyTexture(width: width, height: 1)
            let identity = DSTransform(matrix: matrix_identity_float3x3, offset: .zero,
                                       eigenvalues: .one, wasRegularized: false, sampleCount: 1)
            guard let cb = queue.makeCommandBuffer() else { return XCTFail("no command buffer") }
            try renderer.encode(in: cb, source: source, destination: destination,
                                colorSpace: space, transform: identity, amount: 1)
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let result = readBack(destination)
            for (index, original) in pixels.enumerated() {
                let out = result[index]
                for channel in 0..<3 {
                    XCTAssertEqual(out[channel], original[channel], accuracy: 2e-3,
                                   "space \(space.family) pixel \(index) channel \(channel)")
                }
            }
        }
    }

    func testShaderAndCPUConversionsAgree() throws {
        let library = try DSLibrary.make(device: device)
        let analyzer = try DSAnalyzer(device: device, library: library)

        let pixels = syntheticPanel(width: 64, height: 64)
        let source = try makeTexture(pixels, width: 64, height: 64)

        for space in [DSColorSpace.rgb, .yuv, .lab, .redEmphasis] {
            let center = space.nominalCenter
            guard let cb = queue.makeCommandBuffer() else { return XCTFail("no command buffer") }
            try analyzer.encode(in: cb, texture: source,
                                roi: MTLRegionMake2D(0, 0, 64, 64),
                                colorSpace: space, center: center, samplingStride: 1)
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let gpu = analyzer.resolve()
            XCTAssertEqual(gpu.count, 4096, "every pixel should be counted")

            var cpu = DSMoments(center: SIMD3<Double>(Double(center.x), Double(center.y), Double(center.z)))
            for p in pixels {
                let w = DSColorSpaceMath.toSpace(SIMD3(p.x, p.y, p.z), space)
                cpu.add(SIMD3<Double>(Double(w.x), Double(w.y), Double(w.z)))
            }

            let gpuCov = gpu.covariance, cpuCov = cpu.covariance
            let scale = max(abs(cpuCov[0]), 1e-9)
            for i in 0..<9 {
                XCTAssertEqual(gpuCov[i], cpuCov[i], accuracy: scale * 1e-3,
                               "space \(space.family) covariance entry \(i)")
            }
            for i in 0..<3 {
                XCTAssertEqual(gpu.mean[i], cpu.mean[i], accuracy: max(abs(cpu.mean[i]), 1) * 1e-4,
                               "space \(space.family) mean \(i)")
            }
        }
    }

    // MARK: - End to end

    /// The whole point, verified on real GPU output: channels come out uncorrelated
    /// with the requested spread.
    func testEndToEndProducesDecorrelatedOutput() throws {
        let engine = try DSEngine(device: device)
        var config = DSConfiguration()
        config.colorSpace = .rgb
        config.target = .uniform(fraction: 0.12)
        config.amount = 1
        engine.configuration = config

        let width = 128, height = 128
        let pixels = syntheticPanel(width: width, height: height)
        let source = try makeTexture(pixels, width: width, height: height)
        let destination = try makeEmptyTexture(width: width, height: height)

        try engine.processSynchronously(source: source, destination: destination, queue: queue)

        let result = readBack(destination)
        var out = DSMoments(center: SIMD3(0.5, 0.5, 0.5))
        var clipped = 0
        for p in result {
            if p.x <= 0 || p.x >= 1 || p.y <= 0 || p.y >= 1 || p.z <= 0 || p.z >= 1 { clipped += 1 }
            out.add(SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z)))
        }

        // Clipping is expected at high stretch but must not dominate, or the statistics
        // below are measuring the clamp rather than the transform.
        XCTAssertLessThan(Double(clipped) / Double(result.count), 0.15, "too much clipping to assess")

        let cov = out.covariance
        for (r, c) in [(0, 1), (0, 2), (1, 2)] {
            let correlation = cov[r * 3 + c] / (cov[r * 3 + r].squareRoot() * cov[c * 3 + c].squareRoot())
            XCTAssertEqual(correlation, 0, accuracy: 0.1, "channels \(r),\(c) still correlated")
        }

        // The input was far more correlated than the output: confirm we actually moved.
        var input = DSMoments(center: SIMD3(0.5, 0.5, 0.5))
        for p in pixels { input.add(SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z))) }
        let inCov = input.covariance
        let inCorrelation = inCov[1] / (inCov[0].squareRoot() * inCov[4].squareRoot())
        XCTAssertGreaterThan(abs(inCorrelation), 0.9, "test fixture should start highly correlated")
    }

    /// A faint signal invisible in the source must become a large colour difference.
    func testFaintSignalIsAmplified() throws {
        let engine = try DSEngine(device: device)
        var config = DSConfiguration()
        config.colorSpace = .chromaBoost
        config.target = .uniform(fraction: 0.06)
        engine.configuration = config

        let panel = makePanel(width: 128, height: 128)
        let source = try makeTexture(panel.pixels, width: panel.width, height: panel.height)
        let destination = try makeEmptyTexture(width: panel.width, height: panel.height)
        try engine.processSynchronously(source: source, destination: destination, queue: queue)
        let result = readBack(destination)

        let before = pigmentSeparation(panel.pixels, panel)
        let after = pigmentSeparation(result, panel)
        XCTAssertGreaterThan(before, 0, "fixture should contain some pigment signal")
        XCTAssertGreaterThan(after, before * 3,
                             "faint pigment separation should be substantially amplified " +
                             "(before \(before), after \(after))")
    }

    /// Sub-sampled analysis must behave like a full scan -- this is what makes live 4K
    /// affordable. Compared on rendered output rather than on matrix entries: several
    /// entries sit near zero, where an entry-wise relative tolerance is meaningless.
    func testSamplingStrideMatchesFullAnalysis() throws {
        let panel = makePanel(width: 256, height: 256)
        let source = try makeTexture(panel.pixels, width: panel.width, height: panel.height)

        func render(stride: Int) throws -> (image: [SIMD4<Float>], transform: DSTransform) {
            let engine = try DSEngine(device: device)
            var config = DSConfiguration()
            config.colorSpace = .yuv
            config.target = .uniform(fraction: 0.1)
            config.samplingStride = stride
            engine.configuration = config
            let destination = try makeEmptyTexture(width: panel.width, height: panel.height)
            try engine.processSynchronously(source: source, destination: destination, queue: queue)
            return (readBack(destination), engine.transform)
        }

        let full = try render(stride: 1)
        let sampled = try render(stride: 4)
        XCTAssertEqual(sampled.transform.sampleCount, 64 * 64)
        XCTAssertFalse(full.transform.wasRegularized)

        var maxDifference: Float = 0
        var totalDifference: Float = 0
        for i in 0..<full.image.count {
            for channel in 0..<3 {
                let d = abs(full.image[i][channel] - sampled.image[i][channel])
                maxDifference = max(maxDifference, d)
                totalDifference += d
            }
        }
        let meanDifference = totalDifference / Float(full.image.count * 3)
        XCTAssertLessThan(meanDifference, 0.01,
                          "1/16 of the pixels should reach the same rendering (mean \(meanDifference))")
        XCTAssertLessThan(maxDifference, 0.08, "no pixel should diverge badly (max \(maxDifference))")
    }

    /// The region of interest must change the matrix: statistics come only from inside
    /// it, while the stretch still applies everywhere.
    func testRegionOfInterestChangesTheMatrix() throws {
        let width = 128, height = 128
        var pixels = syntheticPanel(width: width, height: height)
        // Flood the right half with saturated blue, as an irrelevant sky would.
        for y in 0..<height {
            for x in (width / 2)..<width {
                pixels[y * width + x] = SIMD4(0.1, 0.25, 0.9, 1)
            }
        }
        let source = try makeTexture(pixels, width: width, height: height)

        func matrix(roi: MTLRegion?) throws -> simd_float3x3 {
            let engine = try DSEngine(device: device)
            var config = DSConfiguration()
            config.colorSpace = .rgb
            config.target = .uniform(fraction: 0.1)
            config.regionOfInterest = roi
            engine.configuration = config
            let destination = try makeEmptyTexture(width: width, height: height)
            try engine.processSynchronously(source: source, destination: destination, queue: queue)
            return engine.transform.matrix
        }

        let whole = try matrix(roi: nil)
        let leftHalf = try matrix(roi: MTLRegionMake2D(0, 0, width / 2, height))

        var maxDelta: Float = 0
        for c in 0..<3 { for r in 0..<3 {
            maxDelta = max(maxDelta, abs(whole[c][r] - leftHalf[c][r]))
        }}
        XCTAssertGreaterThan(maxDelta, 0.5, "excluding the blue region should change the matrix")
    }

    /// amount = 0 must be a true bypass.
    func testAmountZeroIsIdentity() throws {
        let engine = try DSEngine(device: device)
        var config = DSConfiguration()
        config.colorSpace = .lab
        config.amount = 0
        engine.configuration = config

        let pixels = syntheticPanel(width: 32, height: 32)
        let source = try makeTexture(pixels, width: 32, height: 32)
        let destination = try makeEmptyTexture(width: 32, height: 32)
        try engine.processSynchronously(source: source, destination: destination, queue: queue)

        for (index, out) in readBack(destination).enumerated() {
            for channel in 0..<3 {
                XCTAssertEqual(out[channel], pixels[index][channel], accuracy: 1e-5,
                               "pixel \(index) channel \(channel)")
            }
        }
    }
}
