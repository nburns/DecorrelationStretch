import Metal
import simd

/// The analysis pass: measures mean and covariance of the pixel cloud on the GPU.
///
/// Kept separate from the apply pass because it runs at a different cadence — once per
/// image, or every N frames for live video, rather than once per frame.
public final class DSAnalyzer {
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private var partials: MTLBuffer?
    private var groupsWide = 0
    private var groupsHigh = 0
    private var center: SIMD3<Float> = .zero

    public init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        self.pipeline = try DSLibrary.pipeline(device, library, "ds_stats")
    }

    /// Encodes a statistics pass over `roi`. The moments become readable via `resolve()`
    /// once `commandBuffer` completes.
    public func encode(in commandBuffer: MTLCommandBuffer,
                       texture: MTLTexture,
                       roi: MTLRegion,
                       colorSpace: DSColorSpace,
                       center: SIMD3<Float>,
                       samplingStride: Int) throws {
        let stride = max(1, samplingStride)
        let sampledWidth = (roi.size.width + stride - 1) / stride
        let sampledHeight = (roi.size.height + stride - 1) / stride
        guard sampledWidth > 0, sampledHeight > 0 else { return }

        let edge = DSLayout.threadgroupEdge
        groupsWide = (sampledWidth + edge - 1) / edge
        groupsHigh = (sampledHeight + edge - 1) / edge
        self.center = center

        let needed = groupsWide * groupsHigh * DSLayout.accumulatorCount * MemoryLayout<Float>.stride
        if partials == nil || partials!.length < needed {
            // Shared memory is free to read on unified-memory devices; discrete GPUs need
            // a managed buffer plus an explicit synchronize before the CPU can see it.
            // Every iOS device has unified memory, and .storageModeManaged does not exist
            // there, so that branch is compiled out rather than merely unused.
            #if os(macOS)
            let mode: MTLResourceOptions = device.hasUnifiedMemory ? .storageModeShared : .storageModeManaged
            #else
            let mode: MTLResourceOptions = .storageModeShared
            #endif
            guard let buffer = device.makeBuffer(length: needed, options: mode) else {
                throw DSError.textureCreationFailed
            }
            partials = buffer
        }
        guard let partials else { throw DSError.textureCreationFailed }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DSError.commandEncodingFailed
        }
        encoder.label = "DS analyze"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(partials, offset: 0, index: 0)

        var space = DSSpaceParamsGPU(weights: colorSpace.weights, family: colorSpace.family.rawValue)
        encoder.setBytes(&space, length: MemoryLayout<DSSpaceParamsGPU>.stride, index: 1)

        var stats = DSStatsParamsGPU(
            center: center,
            originX: UInt32(roi.origin.x),
            originY: UInt32(roi.origin.y),
            width: UInt32(roi.size.width),
            height: UInt32(roi.size.height),
            stride: UInt32(stride)
        )
        encoder.setBytes(&stats, length: MemoryLayout<DSStatsParamsGPU>.stride, index: 2)

        encoder.dispatchThreadgroups(
            MTLSize(width: groupsWide, height: groupsHigh, depth: 1),
            threadsPerThreadgroup: MTLSize(width: edge, height: edge, depth: 1)
        )
        encoder.endEncoding()

        #if os(macOS)
        if !device.hasUnifiedMemory, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "DS partials sync"
            blit.synchronize(resource: partials)
            blit.endEncoding()
        }
        #endif
    }

    /// Sums the per-threadgroup partials. Only valid after the encoding command buffer
    /// has completed.
    public func resolve() -> DSMoments {
        var moments = DSMoments(center: SIMD3<Double>(Double(center.x), Double(center.y), Double(center.z)))
        guard let partials, groupsWide > 0, groupsHigh > 0 else { return moments }

        let count = groupsWide * groupsHigh
        let values = partials.contents().bindMemory(to: Float.self, capacity: count * DSLayout.accumulatorCount)

        // Partials arrive as float32 from the GPU; the cross-group sum is done in Double
        // so that a few million pixels do not erode the accumulation.
        var n = 0.0
        var sum = SIMD3<Double>.zero
        var outer = DSMoments.SIMD6()
        for group in 0..<count {
            let base = group * DSLayout.accumulatorCount
            n += Double(values[base + 0])
            sum += SIMD3(Double(values[base + 1]), Double(values[base + 2]), Double(values[base + 3]))
            outer = outer + DSMoments.SIMD6(
                xx: Double(values[base + 4]), xy: Double(values[base + 5]), xz: Double(values[base + 6]),
                yy: Double(values[base + 7]), yz: Double(values[base + 8]), zz: Double(values[base + 9])
            )
        }
        moments.count = n
        moments.sum = sum
        moments.sumOuter = outer
        return moments
    }
}
