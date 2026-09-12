import Metal
import CoreGraphics
import Foundation

/// Bridging between CoreGraphics images and Metal textures, for the stills path.
public enum DSImageIO {

    /// Draws `image` into an sRGB RGBA8 texture.
    ///
    /// The draw is forced through sRGB rather than trusting the image's own profile,
    /// because the shader's LAB conversion assumes sRGB primaries and transfer. Feeding
    /// it Display P3 or Adobe RGB data unconverted would silently skew the covariance.
    public static func makeTexture(from image: CGImage,
                                   device: MTLDevice,
                                   usage: MTLTextureUsage = [.shaderRead]) throws -> MTLTexture {
        let width = image.width, height = image.height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw DSError.textureCreationFailed
        }
        let bytesPerRow = width * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * height)

        let drew: Bool = bytes.withUnsafeMutableBytes { raw in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drew else { throw DSError.textureCreationFailed }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = usage
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        bytes.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: raw.baseAddress!,
                            bytesPerRow: bytesPerRow)
        }
        return texture
    }

    /// Allocates a destination texture matching `source`, suitable for the apply pass.
    public static func makeDestination(like source: MTLTexture,
                                       device: MTLDevice) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: source.pixelFormat,
            width: source.width, height: source.height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DSError.textureCreationFailed
        }
        return texture
    }

    public static func makeImage(from texture: MTLTexture) throws -> CGImage {
        guard texture.pixelFormat == .rgba8Unorm || texture.pixelFormat == .bgra8Unorm else {
            throw DSError.textureCreationFailed
        }
        let width = texture.width, height = texture.height
        let bytesPerRow = width * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * height)
        bytes.withUnsafeMutableBytes { raw in
            texture.getBytes(raw.baseAddress!,
                             bytesPerRow: bytesPerRow,
                             from: MTLRegionMake2D(0, 0, width, height),
                             mipmapLevel: 0)
        }

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(bytes) as CFData) else {
            throw DSError.textureCreationFailed
        }
        var bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        if texture.pixelFormat == .bgra8Unorm {
            bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
                .union(.byteOrder32Little)
        }
        guard let image = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo,
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        ) else { throw DSError.textureCreationFailed }
        return image
    }
}

public extension DSEngine {
    /// Convenience for the stills path: analyse and stretch a `CGImage` end to end.
    func process(image: CGImage, device: MTLDevice, queue: MTLCommandQueue) throws -> CGImage {
        let source = try DSImageIO.makeTexture(from: image, device: device)
        let destination = try DSImageIO.makeDestination(like: source, device: device)
        try processSynchronously(source: source, destination: destination, queue: queue)
        return try DSImageIO.makeImage(from: destination)
    }
}
