import XCTest
import Metal
import simd
@testable import DecorrelationStretch

/// Switching the base colour space is a discontinuity: a transform measured in one space
/// is meaningless in another, so it must not survive the switch or be blended across it.
final class DSColorSpaceSwitchTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "no Metal device available")
        queue = device.makeCommandQueue()
    }

    private func makeSource(width: Int, height: Int) throws -> MTLTexture {
        var pixels = [SIMD4<Float>]()
        var state: UInt64 = 0xBEEF
        func noise() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(1 << 53)) - 0.5
        }
        for i in 0..<(width * height) {
            let luma = 0.45 + 0.15 * Float(i % width) / Float(width) + 0.05 * noise()
            pixels.append(SIMD4(luma + 0.02 * noise(), luma + 0.02 * noise(), luma + 0.02 * noise(), 1))
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        pixels.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: raw.baseAddress!,
                            bytesPerRow: width * MemoryLayout<SIMD4<Float>>.stride)
        }
        return texture
    }

    private func makeDestination(width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        return texture
    }

    /// The reported symptom: after switching base space, the stale matrix from the old
    /// space must never be applied to values in the new one.
    func testSwitchingFamilyDiscardsTheStaleMatrix() throws {
        let engine = try DSEngine(device: device)
        var config = DSConfiguration()
        config.colorSpace = .rgb
        config.target = .uniform(fraction: 0.1)
        config.temporalSmoothing = 0.85          // the setting that made this bad
        engine.configuration = config

        let source = try makeSource(width: 96, height: 96)
        let destination = try makeDestination(width: 96, height: 96)
        try engine.processSynchronously(source: source, destination: destination, queue: queue)

        let rgbMatrix = engine.transform.matrix
        XCTAssertGreaterThan(engine.transform.sampleCount, 0)

        config.colorSpace = .lab
        engine.configuration = config

        // Before any new analysis has run, the RGB matrix must be gone. Applying it to
        // L* up to 100 and a*/b* up to +-128 would produce wild colour.
        let carried = engine.transform
        XCTAssertEqual(carried.sampleCount, 0,
                       "a matrix measured in RGB must not survive a switch to LAB")
        for c in 0..<3 {
            for r in 0..<3 {
                XCTAssertEqual(carried.matrix[c][r], c == r ? 1 : 0, accuracy: 1e-6,
                               "stale matrix entry [\(c)][\(r)] carried across the switch")
            }
        }
        XCTAssertFalse(Self.matricesMatch(carried.matrix, rgbMatrix))
    }

    /// Having discarded it, the very next analysis must snap straight to the correct
    /// matrix rather than easing toward it from the discarded one.
    func testFirstAnalysisAfterSwitchSnapsInsteadOfBlending() throws {
        let source = try makeSource(width: 96, height: 96)
        let destination = try makeDestination(width: 96, height: 96)

        var config = DSConfiguration()
        config.target = .uniform(fraction: 0.1)
        config.temporalSmoothing = 0.85
        config.analysisInterval = 1

        // What a fresh engine measures in LAB, with no history at all.
        let reference = try DSEngine(device: device)
        var referenceConfig = config
        referenceConfig.colorSpace = .lab
        reference.configuration = referenceConfig
        try reference.processSynchronously(source: source, destination: destination, queue: queue)
        let expected = reference.transform.matrix

        // The same thing reached by switching from RGB, driven through the live path.
        let switched = try DSEngine(device: device)
        var switchedConfig = config
        switchedConfig.colorSpace = .rgb
        switched.configuration = switchedConfig
        try switched.processSynchronously(source: source, destination: destination, queue: queue)

        switchedConfig.colorSpace = .lab
        switched.configuration = switchedConfig

        guard let cb = queue.makeCommandBuffer() else { return XCTFail("no command buffer") }
        try switched.encode(source: source, destination: destination, in: cb)
        cb.commit()
        cb.waitUntilCompleted()

        let actual = switched.transform.matrix
        for c in 0..<3 {
            for r in 0..<3 {
                XCTAssertEqual(actual[c][r], expected[c][r], accuracy: max(abs(expected[c][r]) * 0.01, 1e-4),
                               "one analysis after the switch should already be correct at [\(c)][\(r)]")
            }
        }
    }

    /// Adjusting weights inside the same family is continuous, so smoothing should still
    /// apply — resetting there would make every slider drag flicker through identity.
    func testWeightChangeWithinAFamilyKeepsSmoothing() throws {
        let engine = try DSEngine(device: device)
        var config = DSConfiguration()
        config.colorSpace = DSColorSpace(family: .yuv, weights: SIMD3(1, 1, 1))
        config.target = .uniform(fraction: 0.1)
        config.temporalSmoothing = 0.85
        engine.configuration = config

        let source = try makeSource(width: 96, height: 96)
        let destination = try makeDestination(width: 96, height: 96)
        try engine.processSynchronously(source: source, destination: destination, queue: queue)
        XCTAssertGreaterThan(engine.transform.sampleCount, 0)

        config.colorSpace = DSColorSpace(family: .yuv, weights: SIMD3(1, 1.2, 1))
        engine.configuration = config

        XCTAssertGreaterThan(engine.transform.sampleCount, 0,
                             "a weight tweak within one family should not discard the matrix")
    }

    private static func matricesMatch(_ a: simd_float3x3, _ b: simd_float3x3) -> Bool {
        for column in 0..<3 where a[column] != b[column] { return false }
        return true
    }
}
