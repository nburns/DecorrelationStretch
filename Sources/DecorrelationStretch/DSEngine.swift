import Metal
import simd
import Foundation

public struct DSConfiguration {
    /// Base space and axis weights the covariance is measured in.
    public var colorSpace: DSColorSpace = .chromaBoost
    /// How much variance to put on each axis after decorrelating.
    public var target: DSTargetVariance = .uniform(fraction: 0.0588)
    /// Blend between the original and the stretched result, 0...1.
    public var amount: Float = 1

    /// Sample every Nth pixel during analysis. A covariance estimate converges long
    /// before every pixel is read; 2-4 is imperceptible on 4K and meaningfully cheaper.
    public var samplingStride: Int = 1
    /// Frames between re-analysis in live mode. The matrix describes scene-wide colour
    /// statistics, which change slowly, so there is no reason to recompute per frame.
    public var analysisInterval: Int = 12
    /// Exponential smoothing on the matrix, 0...1. Zero snaps to each new analysis;
    /// higher values damp the colour shifts that would otherwise pulse as a camera pans.
    public var temporalSmoothing: Float = 0.85

    /// Restrict the statistics to a sub-rectangle while still stretching the whole
    /// frame. This is DStretch's selection feature and it matters a lot in practice:
    /// excluding sky, foliage, or shadow stops them from consuming the variance budget.
    public var regionOfInterest: MTLRegion?

    public init() {}

    /// True when the new configuration measures in a different base space, which makes
    /// any existing matrix meaningless rather than merely stale: a map built for RGB
    /// values applied to L* up to 100 and a*/b* up to +-128 produces wild colour, and
    /// blending the two is not a meaningful operation.
    ///
    /// Weight changes within one family are deliberately excluded. They are continuous,
    /// so smoothing across them is fine, and discarding the matrix would make every
    /// slider drag flicker through an unfiltered frame.
    func workingSpaceIsIncompatible(with other: DSConfiguration) -> Bool {
        colorSpace.family != other.colorSpace.family
    }

    /// Fields that invalidate a previously computed matrix.
    func analysisDiffers(from other: DSConfiguration) -> Bool {
        if colorSpace != other.colorSpace { return true }
        if target != other.target { return true }
        if samplingStride != other.samplingStride { return true }
        switch (regionOfInterest, other.regionOfInterest) {
        case (nil, nil): return false
        case let (a?, b?):
            return a.origin.x != b.origin.x || a.origin.y != b.origin.y
                || a.size.width != b.size.width || a.size.height != b.size.height
        default: return true
        }
    }
}

/// Drives the two passes at their natural cadences: analysis occasionally, apply every
/// frame. Safe to drive from a render loop — `encode` never blocks on the GPU.
public final class DSEngine {
    private let device: MTLDevice
    private let analyzer: DSAnalyzer
    private let renderer: DSRenderer

    private let lock = NSLock()
    private var _configuration = DSConfiguration()
    private var _transform = DSTransform.identity
    private var center: SIMD3<Float>?
    private var frameIndex = 0
    private var analysisInFlight = false
    private var needsImmediateAnalysis = true

    public init(device: MTLDevice) throws {
        DSLayout.validate()
        self.device = device
        let library = try DSLibrary.make(device: device)
        self.analyzer = try DSAnalyzer(device: device, library: library)
        self.renderer = try DSRenderer(device: device, library: library)
    }

    public var configuration: DSConfiguration {
        get { lock.withLock { _configuration } }
        set {
            lock.withLock {
                if newValue.analysisDiffers(from: _configuration) {
                    needsImmediateAnalysis = true
                    center = nil
                }
                if newValue.workingSpaceIsIncompatible(with: _configuration) {
                    // Dropping to identity also resets sampleCount, which makes the next
                    // analysis snap rather than ease in from a matrix built for a
                    // different space.
                    _transform = .identity
                }
                _configuration = newValue
            }
        }
    }

    /// The matrix currently being applied. Updated asynchronously as analyses land.
    public var transform: DSTransform {
        lock.withLock { _transform }
    }

    /// Discards accumulated state so the next frame re-analyses from scratch.
    public func reset() {
        lock.withLock {
            _transform = .identity
            center = nil
            frameIndex = 0
            needsImmediateAnalysis = true
        }
    }

    // MARK: - Live path

    /// Encodes both passes into the caller's command buffer and returns immediately.
    ///
    /// The apply pass uses the most recent completed analysis, so the matrix trails the
    /// image by up to `analysisInterval` frames. That lag is invisible and slightly
    /// desirable: it stops the colour mapping from chattering frame to frame.
    public func encode(source: MTLTexture,
                       destination: MTLTexture,
                       in commandBuffer: MTLCommandBuffer) throws {
        let (config, currentTransform, shouldAnalyze, statsCenter) = lock.withLock {
            () -> (DSConfiguration, DSTransform, Bool, SIMD3<Float>) in
            let due = needsImmediateAnalysis || frameIndex % max(1, _configuration.analysisInterval) == 0
            let go = due && !analysisInFlight
            if go {
                analysisInFlight = true
                needsImmediateAnalysis = false
            }
            frameIndex &+= 1
            return (_configuration, _transform, go, center ?? _configuration.colorSpace.nominalCenter)
        }

        try renderer.encode(in: commandBuffer,
                            source: source,
                            destination: destination,
                            colorSpace: config.colorSpace,
                            transform: currentTransform,
                            amount: config.amount)

        guard shouldAnalyze else { return }

        let roi = config.regionOfInterest ?? MTLRegion(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: source.width, height: source.height, depth: 1)
        )

        do {
            try analyzer.encode(in: commandBuffer,
                                texture: source,
                                roi: roi,
                                colorSpace: config.colorSpace,
                                center: statsCenter,
                                samplingStride: config.samplingStride)
        } catch {
            lock.withLock { analysisInFlight = false }
            throw error
        }

        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self else { return }
            let moments = self.analyzer.resolve()
            self.ingest(moments: moments, colorSpace: config.colorSpace, target: config.target,
                        smoothing: config.temporalSmoothing)
        }
    }

    private func ingest(moments: DSMoments,
                        colorSpace: DSColorSpace,
                        target: DSTargetVariance,
                        smoothing: Float) {
        guard moments.count > 1 else {
            lock.withLock { analysisInFlight = false }
            return
        }
        let fresh = DSSolver.makeTransform(moments: moments,
                                           target: target,
                                           nominalScale: colorSpace.nominalScale)
        let mean = moments.mean
        lock.withLock {
            // Re-center the next accumulation on the measured mean: keeps the covariance
            // subtraction well conditioned and tracks the scene as it changes.
            center = SIMD3<Float>(Float(mean.x), Float(mean.y), Float(mean.z))
            let blend = max(0, min(1, 1 - smoothing))
            _transform = _transform.sampleCount == 0 ? fresh : _transform.mix(with: fresh, t: blend)
            analysisInFlight = false
        }
    }

    // MARK: - Still path

    /// Analyses and applies in one call, waiting for the GPU. Intended for stills and
    /// export, not for a render loop.
    public func processSynchronously(source: MTLTexture,
                                     destination: MTLTexture,
                                     queue: MTLCommandQueue) throws {
        let config = configuration
        let roi = config.regionOfInterest ?? MTLRegion(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: source.width, height: source.height, depth: 1)
        )

        guard let analysisBuffer = queue.makeCommandBuffer() else { throw DSError.commandEncodingFailed }
        try analyzer.encode(in: analysisBuffer,
                            texture: source,
                            roi: roi,
                            colorSpace: config.colorSpace,
                            center: config.colorSpace.nominalCenter,
                            samplingStride: config.samplingStride)
        analysisBuffer.commit()
        analysisBuffer.waitUntilCompleted()
        if let error = analysisBuffer.error { throw error }

        let moments = analyzer.resolve()
        let fresh = DSSolver.makeTransform(moments: moments,
                                           target: config.target,
                                           nominalScale: config.colorSpace.nominalScale)
        lock.withLock { _transform = fresh }

        guard let applyBuffer = queue.makeCommandBuffer() else { throw DSError.commandEncodingFailed }
        try renderer.encode(in: applyBuffer,
                            source: source,
                            destination: destination,
                            colorSpace: config.colorSpace,
                            transform: fresh,
                            amount: config.amount)
        applyBuffer.commit()
        applyBuffer.waitUntilCompleted()
        if let error = applyBuffer.error { throw error }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock(); defer { unlock() }
        return try body()
    }
}
