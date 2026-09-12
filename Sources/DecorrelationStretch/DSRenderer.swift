import Metal
import simd

/// The apply pass: one affine multiply per pixel. Memory-bandwidth bound, so it costs
/// essentially the same as a copy.
public final class DSRenderer {
    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice, library: MTLLibrary) throws {
        self.pipeline = try DSLibrary.pipeline(device, library, "ds_apply")
    }

    public func encode(in commandBuffer: MTLCommandBuffer,
                       source: MTLTexture,
                       destination: MTLTexture,
                       colorSpace: DSColorSpace,
                       transform: DSTransform,
                       amount: Float) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DSError.commandEncodingFailed
        }
        encoder.label = "DS apply"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(destination, index: 1)

        var space = DSSpaceParamsGPU(weights: colorSpace.weights, family: colorSpace.family.rawValue)
        encoder.setBytes(&space, length: MemoryLayout<DSSpaceParamsGPU>.stride, index: 0)

        var apply = DSApplyParamsGPU(matrix: transform.matrix,
                                     offset: transform.offset,
                                     amount: max(0, min(1, amount)))
        encoder.setBytes(&apply, length: MemoryLayout<DSApplyParamsGPU>.stride, index: 1)

        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let grid = MTLSize(width: destination.width, height: destination.height, depth: 1)

        if pipeline.maxTotalThreadsPerThreadgroup >= w * h {
            encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerGroup)
        } else {
            let groups = MTLSize(width: (destination.width + w - 1) / w,
                                 height: (destination.height + h - 1) / h,
                                 depth: 1)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
        }
        encoder.endEncoding()
    }
}
