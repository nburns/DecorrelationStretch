import XCTest
import Metal
import simd
@testable import DecorrelationStretch

/// Measures the two passes at 4K so the "free per frame" claim is backed by numbers on
/// whatever machine this runs on, rather than asserted.
final class DSPerformanceTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "no Metal device available")
        queue = device.makeCommandQueue()
    }

    private func makeUHDTexture(usage: MTLTextureUsage) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 3840, height: 2160, mipmapped: false)
        descriptor.usage = usage
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        return texture
    }

    private func medianGPUMilliseconds(iterations: Int,
                                       _ encode: (MTLCommandBuffer) throws -> Void) rethrows -> Double {
        var samples: [Double] = []
        for _ in 0..<iterations {
            guard let cb = queue.makeCommandBuffer() else { continue }
            try encode(cb)
            cb.commit()
            cb.waitUntilCompleted()
            samples.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
        }
        samples.sort()
        return samples.isEmpty ? .nan : samples[samples.count / 2]
    }

    func testFourKThroughput() throws {
        let library = try DSLibrary.make(device: device)
        let renderer = try DSRenderer(device: device, library: library)
        let analyzer = try DSAnalyzer(device: device, library: library)

        let source = try makeUHDTexture(usage: [.shaderRead, .shaderWrite])
        let destination = try makeUHDTexture(usage: [.shaderRead, .shaderWrite])
        let roi = MTLRegionMake2D(0, 0, 3840, 2160)
        let transform = DSTransform(matrix: matrix_identity_float3x3, offset: .zero,
                                    eigenvalues: .one, wasRegularized: false, sampleCount: 1)

        // Warm up: first submission includes pipeline and allocation costs.
        _ = try medianGPUMilliseconds(iterations: 3) { cb in
            try renderer.encode(in: cb, source: source, destination: destination,
                                colorSpace: .lab, transform: transform, amount: 1)
        }

        let applyRGB = try medianGPUMilliseconds(iterations: 30) { cb in
            try renderer.encode(in: cb, source: source, destination: destination,
                                colorSpace: .rgb, transform: transform, amount: 1)
        }
        let applyLAB = try medianGPUMilliseconds(iterations: 30) { cb in
            try renderer.encode(in: cb, source: source, destination: destination,
                                colorSpace: .lab, transform: transform, amount: 1)
        }
        let analyseFull = try medianGPUMilliseconds(iterations: 30) { cb in
            try analyzer.encode(in: cb, texture: source, roi: roi, colorSpace: .lab,
                                center: DSColorSpace.lab.nominalCenter, samplingStride: 1)
        }
        let analyseStride4 = try medianGPUMilliseconds(iterations: 30) { cb in
            try analyzer.encode(in: cb, texture: source, roi: roi, colorSpace: .lab,
                                center: DSColorSpace.lab.nominalCenter, samplingStride: 4)
        }

        print("""

        === decorrelation stretch @ 3840x2160, \(device.name) ===
          apply  (RGB space)      \(String(format: "%6.3f", applyRGB)) ms   \
        -> \(String(format: "%5.0f", 1000 / applyRGB)) fps ceiling
          apply  (LAB space)      \(String(format: "%6.3f", applyLAB)) ms   \
        -> \(String(format: "%5.0f", 1000 / applyLAB)) fps ceiling
          analyse (every pixel)   \(String(format: "%6.3f", analyseFull)) ms
          analyse (stride 4)      \(String(format: "%6.3f", analyseStride4)) ms
        Per-frame cost at the default 12-frame analysis interval:
          \(String(format: "%.3f", applyLAB + analyseStride4 / 12)) ms

        """)

        // A 60fps budget is 16.6ms; the apply pass must be a small fraction of it.
        XCTAssertLessThan(applyLAB, 8.0, "apply pass too slow to be called real-time")
        XCTAssertLessThan(analyseStride4, analyseFull * 0.75,
                          "sub-sampling should meaningfully reduce analysis cost")
    }
}
