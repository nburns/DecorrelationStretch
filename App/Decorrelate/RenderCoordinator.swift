import MetalKit
import CoreVideo
import CoreGraphics
import simd
import DecorrelationStretch

/// Owns the Metal objects and drives both passes from the view's draw callback.
///
/// Deliberately not `@MainActor`: state arrives from three threads — SwiftUI on main,
/// the capture queue, and Metal completion handlers — so configuration is snapshotted
/// under a lock rather than relying on actor isolation inside a delegate callback.
final class RenderCoordinator: NSObject, MTKViewDelegate {

    struct Diagnostics {
        var eigenvalues: SIMD3<Float>
        var wasRegularized: Bool
        var sampleCount: Int
        var milliseconds: Double
    }

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let engine: DSEngine
    private let displayPipeline: MTLRenderPipelineState
    private let sampler: MTLSamplerState
    private let textureCache: DSTextureCache
    private let camera = CameraSource()

    private let stateLock = NSLock()
    private var configuration = DSConfiguration()
    private var mode: SourceMode = .image

    private var imageTexture: MTLTexture?
    private var sourceImage: CGImage?
    private var intermediate: MTLTexture?

    private let frameLock = NSLock()
    private var pendingPixelBuffer: CVPixelBuffer?

    private var lastDiagnosticsPush: CFAbsoluteTime = 0
    private weak var hostView: MTKView?

    var onDiagnostics: ((Diagnostics) -> Void)?
    var onStatus: ((String) -> Void)?
    var onSourceSize: ((CGSize) -> Void)?

    private struct DisplayUniforms {
        var scale: SIMD2<Float>
    }

    override init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("This Mac has no usable Metal device.")
        }
        self.device = device
        self.queue = queue

        do {
            self.engine = try DSEngine(device: device)
            self.textureCache = try DSTextureCache(device: device)
        } catch {
            fatalError("Could not create the decorrelation stretch engine: \(error)")
        }

        // Display.metal is compiled into the app target's default library by Xcode.
        guard let library = device.makeDefaultLibrary(),
              let vertexFunction = library.makeFunction(name: "display_vertex"),
              let fragmentFunction = library.makeFunction(name: "display_fragment") else {
            fatalError("Display shaders are missing from the app's Metal library.")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: descriptor) else {
            fatalError("Could not build the display pipeline.")
        }
        self.displayPipeline = pipeline

        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.sAddressMode = .clampToEdge
        samplerDescriptor.tAddressMode = .clampToEdge
        guard let sampler = device.makeSamplerState(descriptor: samplerDescriptor) else {
            fatalError("Could not build the display sampler.")
        }
        self.sampler = sampler

        super.init()

        camera.onFrame = { [weak self] buffer in
            guard let self else { return }
            self.frameLock.lock()
            self.pendingPixelBuffer = buffer
            self.frameLock.unlock()
        }
        camera.onError = { [weak self] message in
            self?.onStatus?(message)
        }
    }

    // MARK: - Input

    func update(configuration: DSConfiguration, mode: SourceMode) {
        stateLock.lock()
        self.configuration = configuration
        let modeChanged = self.mode != mode
        self.mode = mode
        stateLock.unlock()

        engine.configuration = configuration

        if modeChanged {
            if mode == .camera {
                camera.start()
            } else {
                camera.stop()
                frameLock.lock(); pendingPixelBuffer = nil; frameLock.unlock()
            }
            engine.reset()
        }
    }

    func loadImage(url: URL) {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }

        guard let provider = CGDataProvider(url: url as CFURL),
              let image = decodeImage(provider: provider, url: url) else {
            onStatus?("Could not decode \(url.lastPathComponent).")
            return
        }
        do {
            let texture = try DSImageIO.makeTexture(from: image, device: device)
            stateLock.lock()
            imageTexture = texture
            sourceImage = image
            intermediate = nil
            stateLock.unlock()
            engine.reset()
            onSourceSize?(CGSize(width: image.width, height: image.height))
            onStatus?("\(url.lastPathComponent) — \(image.width)x\(image.height)")
        } catch {
            onStatus?("Could not upload the image to the GPU: \(error)")
        }
    }

    private func decodeImage(provider: CGDataProvider, url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithDataProvider(provider, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// Full-resolution render of the current source, for export. Runs the synchronous
    /// path rather than reusing the preview's matrix so the result is analysed at full
    /// resolution regardless of the preview's sampling stride.
    func exportCurrentFrame() -> CGImage? {
        stateLock.lock()
        let mode = self.mode
        let image = self.sourceImage
        stateLock.unlock()

        if mode == .image, let image {
            return try? engine.process(image: image, device: device, queue: queue)
        }
        guard let source = currentCameraTexture() else { return nil }
        guard let destination = try? DSImageIO.makeDestination(like: source, device: device) else { return nil }
        // The camera texture is BGRA; render into a matching layout so the readback is
        // interpreted correctly.
        try? engine.processSynchronously(source: source, destination: destination, queue: queue)
        return try? DSImageIO.makeImage(from: destination)
    }

    private func currentCameraTexture() -> MTLTexture? {
        frameLock.lock()
        let buffer = pendingPixelBuffer
        frameLock.unlock()
        guard let buffer else { return nil }
        return try? textureCache.texture(from: buffer)
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        stateLock.lock()
        let mode = self.mode
        let imageTexture = self.imageTexture
        stateLock.unlock()

        let source: MTLTexture?
        switch mode {
        case .image:  source = imageTexture
        case .camera: source = currentCameraTexture()
        }

        hostView = view

        guard let source,
              let passDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let commandBuffer = queue.makeCommandBuffer() else { return }

        if mode == .camera {
            onSourceSize?(CGSize(width: source.width, height: source.height))
        }

        let target = ensureIntermediate(width: source.width, height: source.height)
        guard let target else { return }

        // Read back what encode actually applied. The analysis this frame schedules
        // resolves in a completion handler afterwards, so in on-demand mode nothing would
        // ever show the new matrix unless we ask for one more frame once it lands.
        do {
            try engine.encode(source: source, destination: target, in: commandBuffer)
        } catch {
            onStatus?("Render failed: \(error)")
            return
        }
        let renderedWith = engine.transform.matrix

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else { return }
        encoder.label = "Display"
        encoder.setRenderPipelineState(displayPipeline)
        var uniforms = DisplayUniforms(scale: aspectFitScale(
            imageSize: CGSize(width: source.width, height: source.height),
            viewSize: view.drawableSize))
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<DisplayUniforms>.stride, index: 0)
        encoder.setFragmentTexture(target, index: 0)
        encoder.setFragmentSamplerState(sampler, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()

        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            self.textureCache.endFrame()
            self.publishDiagnostics(milliseconds: (buffer.gpuEndTime - buffer.gpuStartTime) * 1000)
            if mode == .image, !Self.matricesMatch(renderedWith, self.engine.transform.matrix) {
                DispatchQueue.main.async { self.hostView?.requestRedraw() }
            }
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private static func matricesMatch(_ a: simd_float3x3, _ b: simd_float3x3) -> Bool {
        for column in 0..<3 where a[column] != b[column] { return false }
        return true
    }

    private func publishDiagnostics(milliseconds: Double) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastDiagnosticsPush > 0.2 else { return }
        lastDiagnosticsPush = now
        let transform = engine.transform
        let diagnostics = Diagnostics(eigenvalues: transform.eigenvalues,
                                      wasRegularized: transform.wasRegularized,
                                      sampleCount: transform.sampleCount,
                                      milliseconds: milliseconds)
        DispatchQueue.main.async { self.onDiagnostics?(diagnostics) }
    }

    private func ensureIntermediate(width: Int, height: Int) -> MTLTexture? {
        if let existing = intermediate, existing.width == width, existing.height == height {
            return existing
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        intermediate = device.makeTexture(descriptor: descriptor)
        return intermediate
    }

    /// Shrinks the quad along whichever axis would otherwise overflow, letterboxing the
    /// rest. `PreviewGeometry.fittedRect` performs the same fit in view coordinates so
    /// that region-of-interest dragging lines up with what is drawn.
    private func aspectFitScale(imageSize: CGSize, viewSize: CGSize) -> SIMD2<Float> {
        guard imageSize.width > 0, imageSize.height > 0,
              viewSize.width > 0, viewSize.height > 0 else { return SIMD2(1, 1) }
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height
        return imageAspect > viewAspect
            ? SIMD2(1, Float(viewAspect / imageAspect))
            : SIMD2(Float(imageAspect / viewAspect), 1)
    }
}

/// Shared aspect-fit geometry so the SwiftUI overlay and the Metal display pass agree.
enum PreviewGeometry {
    static func fittedRect(imageSize: CGSize, in viewSize: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0,
              viewSize.width > 0, viewSize.height > 0 else { return .zero }
        let scale = min(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(x: (viewSize.width - size.width) / 2,
                      y: (viewSize.height - size.height) / 2,
                      width: size.width, height: size.height)
    }
}
