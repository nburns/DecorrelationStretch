import Metal
import CoreVideo

/// Zero-copy `CVPixelBuffer` to `MTLTexture` bridging for the live path.
///
/// Capture buffers are already in GPU-accessible memory, so wrapping them through
/// `CVMetalTextureCache` avoids the copy that `texture.replace` would cost every frame.
/// This is the difference between the filter being free and the filter being the most
/// expensive thing in the pipeline.
public final class DSTextureCache {
    private var cache: CVMetalTextureCache?
    /// Holds each frame's CVMetalTexture alive until the next frame; releasing it while
    /// the GPU still reads the derived MTLTexture would be a use-after-free.
    private var retained: [CVMetalTexture] = []

    public init(device: MTLDevice) throws {
        var created: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &created)
        guard status == kCVReturnSuccess, let created else {
            throw DSError.textureCreationFailed
        }
        cache = created
    }

    /// Wraps the buffer's base plane as a texture. The returned texture is valid until
    /// `endFrame()` is called, which must happen only after the GPU work completes.
    public func texture(from pixelBuffer: CVPixelBuffer,
                        pixelFormat: MTLPixelFormat = .bgra8Unorm,
                        plane: Int = 0) throws -> MTLTexture {
        guard let cache else { throw DSError.textureCreationFailed }
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, plane)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, plane)

        var wrapped: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil,
            pixelFormat, width, height, plane, &wrapped)
        guard status == kCVReturnSuccess,
              let wrapped,
              let texture = CVMetalTextureGetTexture(wrapped) else {
            throw DSError.textureCreationFailed
        }
        retained.append(wrapped)
        return texture
    }

    /// Call from the command buffer's completion handler, once the GPU is done with the
    /// frame's textures.
    public func endFrame() {
        retained.removeAll(keepingCapacity: true)
        if let cache { CVMetalTextureCacheFlush(cache, 0) }
    }
}
