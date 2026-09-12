import SwiftUI
import UniformTypeIdentifiers
import AppKit
import ImageIO
import CoreGraphics

@main
struct DecorrelateApp: App {
    init() { HeadlessMode.runIfRequested() }

    var body: some Scene {
        WindowGroup("Decorrelate") {
            ContentView()
                .frame(minWidth: 900, minHeight: 560)
        }
        .windowToolbarStyle(.unified)
    }
}

struct ContentView: View {
    @StateObject private var model = FilterModel()
    @StateObject private var host = CoordinatorHost()
    @State private var revision = 0

    var body: some View {
        HStack(spacing: 0) {
            ZStack {
                MetalPreview(coordinator: host.coordinator,
                             isLive: model.sourceMode == .camera,
                             revision: revision)
                if model.sourceSize != .zero {
                    RegionSelectionOverlay(imageSize: model.sourceSize,
                                           regionOfInterest: $model.regionOfInterest)
                }
                if model.sourceSize == .zero {
                    VStack(spacing: 8) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 40, weight: .thin))
                        Text("Open an image or switch to the camera")
                            .font(.callout)
                    }
                    .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            ControlsView(model: model, onOpenImage: openImage, onExport: exportImage)
        }
        .onAppear {
            wire()
            if let url = HeadlessMode.preloadURL { load(url) }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                DispatchQueue.main.async { load(url) }
            }
            return true
        }
        .onChange(of: model.configurationSignature) { _ in push() }
        .onChange(of: model.sourceMode) { _ in push() }
    }

    private func wire() {
        host.coordinator.onDiagnostics = { diagnostics in
            model.eigenvalues = diagnostics.eigenvalues
            model.wasRegularized = diagnostics.wasRegularized
            model.sampleCount = diagnostics.sampleCount
            model.frameMilliseconds = diagnostics.milliseconds
        }
        host.coordinator.onStatus = { model.status = $0 }
        host.coordinator.onSourceSize = { size in
            if model.sourceSize != size { model.sourceSize = size }
        }
        push()
    }

    private func push() {
        host.coordinator.update(configuration: model.configuration, mode: model.sourceMode)
        revision &+= 1
    }

    private func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url)
    }

    private func load(_ url: URL) {
        model.sourceMode = .image
        model.regionOfInterest = nil
        host.coordinator.loadImage(url: url)
        push()
    }

    private func exportImage() {
        guard let image = host.coordinator.exportCurrentFrame() else {
            model.status = "Nothing to export yet."
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "stretched.png"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            model.status = "Could not create \(url.lastPathComponent)."
            return
        }
        CGImageDestinationAddImage(destination, image, nil)
        if CGImageDestinationFinalize(destination) {
            model.status = "Saved \(url.lastPathComponent)."
        } else {
            model.status = "Could not write \(url.lastPathComponent)."
        }
    }
}

/// Keeps one RenderCoordinator alive for the window's lifetime.
final class CoordinatorHost: ObservableObject {
    let coordinator = RenderCoordinator()
}
