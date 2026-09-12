import XCTest
import Metal
import CoreGraphics
import simd
@testable import DecorrelationStretch

final class DSIntegrationTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "no Metal device available")
        queue = device.makeCommandQueue()
    }

    private func makeTestImage(width: Int, height: Int) throws -> CGImage {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw DSError.textureCreationFailed }

        for y in 0..<height {
            for x in 0..<width {
                let luma = 0.4 + 0.2 * CGFloat(x) / CGFloat(width)
                let pigment = (x / 8 + y / 8) % 2 == 0 ? 0.02 : -0.02
                context.setFillColor(red: luma + pigment, green: luma, blue: luma - pigment, alpha: 1)
                context.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        guard let image = context.makeImage() else { throw DSError.textureCreationFailed }
        return image
    }

    func testCGImageRoundTripProducesStretchedImage() throws {
        let engine = try DSEngine(device: device)
        var config = DSConfiguration()
        config.colorSpace = .chromaBoost
        config.target = .uniform(fraction: 0.08)
        engine.configuration = config

        let input = try makeTestImage(width: 96, height: 96)
        let output = try engine.process(image: input, device: device, queue: queue)

        XCTAssertEqual(output.width, input.width)
        XCTAssertEqual(output.height, input.height)
        XCTAssertGreaterThan(engine.transform.sampleCount, 0)
        XCTAssertFalse(engine.transform.wasRegularized)

        // The stretch must have measurably widened the colour distribution.
        func spread(_ image: CGImage) throws -> Double {
            let texture = try DSImageIO.makeTexture(from: image, device: device)
            var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
            bytes.withUnsafeMutableBytes { raw in
                texture.getBytes(raw.baseAddress!, bytesPerRow: image.width * 4,
                                 from: MTLRegionMake2D(0, 0, image.width, image.height),
                                 mipmapLevel: 0)
            }
            var moments = DSMoments(center: SIMD3(128, 128, 128))
            for i in stride(from: 0, to: bytes.count, by: 4) {
                moments.add(SIMD3(Double(bytes[i]), Double(bytes[i + 1]), Double(bytes[i + 2])))
            }
            let cov = moments.covariance
            return cov[0] + cov[4] + cov[8]
        }

        XCTAssertGreaterThan(try spread(output), try spread(input) * 2,
                             "output should occupy far more of the colour volume")
    }

    /// An image whose channels are exactly linearly dependent makes the covariance
    /// singular. The filter must flag it and still return a usable image rather than
    /// producing NaNs that propagate into the display.
    func testDegenerateImageIsFlaggedAndStillRenders() throws {
        let width = 64, height = 64
        var pixels = [SIMD4<Float>]()
        var state: UInt64 = 12345
        func noise() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53))
        }
        for _ in 0..<(width * height) {
            let v = 0.2 + 0.5 * noise()
            pixels.append(SIMD4(v, v, v, 1))   // pure grey: rank 1
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let source = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        pixels.withUnsafeBytes { raw in
            source.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                           withBytes: raw.baseAddress!,
                           bytesPerRow: width * MemoryLayout<SIMD4<Float>>.stride)
        }

        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let destination = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }

        let engine = try DSEngine(device: device)
        var config = DSConfiguration()
        config.colorSpace = .rgb
        engine.configuration = config
        try engine.processSynchronously(source: source, destination: destination, queue: queue)

        XCTAssertTrue(engine.transform.wasRegularized,
                      "a rank-deficient image should be reported, not silently mangled")

        var out = [SIMD4<Float>](repeating: .zero, count: width * height)
        out.withUnsafeMutableBytes { raw in
            destination.getBytes(raw.baseAddress!,
                                 bytesPerRow: width * MemoryLayout<SIMD4<Float>>.stride,
                                 from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        for (index, p) in out.enumerated() {
            for channel in 0..<3 {
                XCTAssertTrue(p[channel].isFinite, "pixel \(index) channel \(channel) is not finite")
                XCTAssertTrue(p[channel] >= 0 && p[channel] <= 1, "pixel \(index) out of range")
            }
        }
    }

    /// The live path must not block and must converge to the same matrix the
    /// synchronous path finds.
    func testLiveEncodeConvergesToSynchronousResult() throws {
        let width = 128, height = 128
        var pixels = [SIMD4<Float>]()
        var state: UInt64 = 777
        func noise() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53)) - 0.5
        }
        for i in 0..<(width * height) {
            let luma = 0.45 + 0.1 * Float(i % width) / Float(width) + 0.04 * noise()
            pixels.append(SIMD4(luma + 0.02 * noise(), luma + 0.02 * noise(), luma + 0.02 * noise(), 1))
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let source = device.makeTexture(descriptor: descriptor),
              let destination = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        pixels.withUnsafeBytes { raw in
            source.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                           withBytes: raw.baseAddress!,
                           bytesPerRow: width * MemoryLayout<SIMD4<Float>>.stride)
        }

        var config = DSConfiguration()
        config.colorSpace = .yuv
        config.target = .uniform(fraction: 0.1)
        config.temporalSmoothing = 0          // snap, so convergence is immediate
        config.analysisInterval = 1

        let reference = try DSEngine(device: device)
        reference.configuration = config
        try reference.processSynchronously(source: source, destination: destination, queue: queue)

        let live = try DSEngine(device: device)
        live.configuration = config
        for _ in 0..<4 {
            guard let cb = queue.makeCommandBuffer() else { return XCTFail("no command buffer") }
            try live.encode(source: source, destination: destination, in: cb)
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }

        let a = reference.transform.matrix, b = live.transform.matrix
        for c in 0..<3 {
            for r in 0..<3 {
                XCTAssertEqual(b[c][r], a[c][r], accuracy: max(abs(a[c][r]) * 0.02, 1e-3),
                               "live matrix[\(c)][\(r)] did not converge")
            }
        }
    }

    /// Changing the colour space must invalidate the matrix immediately rather than
    /// waiting for the next scheduled analysis.
    func testColorSpaceChangeForcesReanalysis() throws {
        let engine = try DSEngine(device: device)
        var config = DSConfiguration()
        config.colorSpace = .rgb
        config.analysisInterval = 1000        // would otherwise never re-analyse
        engine.configuration = config

        let image = try makeTestImage(width: 64, height: 64)
        let source = try DSImageIO.makeTexture(from: image, device: device)
        let destination = try DSImageIO.makeDestination(like: source, device: device)

        func renderOnce() throws {
            guard let cb = queue.makeCommandBuffer() else { throw DSError.commandEncodingFailed }
            try engine.encode(source: source, destination: destination, in: cb)
            cb.commit()
            cb.waitUntilCompleted()
        }

        try renderOnce()
        try renderOnce()
        let rgbMatrix = engine.transform.matrix

        config.colorSpace = .redEmphasis
        engine.configuration = config
        try renderOnce()
        try renderOnce()
        let labMatrix = engine.transform.matrix

        var maxDelta: Float = 0
        for c in 0..<3 { for r in 0..<3 {
            maxDelta = max(maxDelta, abs(rgbMatrix[c][r] - labMatrix[c][r]))
        }}
        XCTAssertGreaterThan(maxDelta, 0.1, "colour space change should have rebuilt the matrix")
    }
}
