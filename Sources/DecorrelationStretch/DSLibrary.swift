import Metal
import Foundation
import simd

public enum DSError: Error, CustomStringConvertible {
    case shaderSourceMissing
    case libraryCompilationFailed(String)
    case functionMissing(String)
    case commandEncodingFailed
    case textureCreationFailed

    public var description: String {
        switch self {
        case .shaderSourceMissing:
            return "DSShaders.metal was not found in the package bundle."
        case .libraryCompilationFailed(let message):
            return "Metal library compilation failed: \(message)"
        case .functionMissing(let name):
            return "Metal function '\(name)' not found in the compiled library."
        case .commandEncodingFailed:
            return "Could not create a Metal compute command encoder."
        case .textureCreationFailed:
            return "Could not allocate a Metal texture."
        }
    }
}

enum DSLibrary {
    /// Prefers a precompiled `default.metallib` (what Xcode produces when the shader is
    /// part of an app target) and falls back to compiling the bundled `.metal` source at
    /// runtime, which is what SwiftPM leaves us with. Runtime compilation costs roughly
    /// 20-50ms once at init and removes any build-time Metal toolchain requirement.
    static func make(device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: Bundle.module),
           library.makeFunction(name: "ds_apply") != nil {
            return library
        }
        guard let url = Bundle.module.url(forResource: "DSShaders", withExtension: "metal"),
              let source = try? String(contentsOf: url, encoding: .utf8) else {
            throw DSError.shaderSourceMissing
        }
        do {
            let options = MTLCompileOptions()
            if #available(macOS 15.0, iOS 18.0, *) {
                options.mathMode = .fast
            } else {
                options.fastMathEnabled = true
            }
            return try device.makeLibrary(source: source, options: options)
        } catch {
            throw DSError.libraryCompilationFailed(String(describing: error))
        }
    }

    static func pipeline(_ device: MTLDevice,
                         _ library: MTLLibrary,
                         _ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw DSError.functionMissing(name)
        }
        return try device.makeComputePipelineState(function: function)
    }
}

// MARK: - GPU-facing layouts

/// Mirrors `DSSpaceParams` in DSShaders.metal.
struct DSSpaceParamsGPU {
    var weights: SIMD3<Float>
    var family: Int32
}

/// Mirrors `DSApplyParams` in DSShaders.metal.
struct DSApplyParamsGPU {
    var matrix: simd_float3x3
    var offset: SIMD3<Float>
    var amount: Float
}

/// Mirrors `DSStatsParams` in DSShaders.metal.
struct DSStatsParamsGPU {
    var center: SIMD3<Float>
    var originX: UInt32
    var originY: UInt32
    var width: UInt32
    var height: UInt32
    var stride: UInt32
}

enum DSLayout {
    static let threadsPerGroup = 256
    static let threadgroupEdge = 16
    static let accumulatorCount = 10

    /// The Swift and MSL struct layouts must agree byte for byte; a mismatch would show
    /// up as a plausible-looking but wrong image rather than a crash.
    static func validate() {
        assert(MemoryLayout<DSSpaceParamsGPU>.stride == 32)
        assert(MemoryLayout<DSApplyParamsGPU>.stride == 80)
        assert(MemoryLayout<DSStatsParamsGPU>.stride == 48)
    }
}
