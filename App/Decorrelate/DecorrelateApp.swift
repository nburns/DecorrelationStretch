import SwiftUI
import UniformTypeIdentifiers
import CoreGraphics
import DecorrelationStretch

@main
struct DecorrelateApp: App {
    init() { HeadlessMode.runIfRequested() }

    var body: some Scene {
        WindowGroup("Decorrelate") {
            ContentView()
                #if os(macOS)
                .frame(minWidth: 900, minHeight: 560)
                #endif
        }
    }
}

/// A PNG wrapper so export can go through SwiftUI's `fileExporter`, which works on both
/// platforms — unlike `NSSavePanel`.
struct PNGDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.png] }
    var data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        guard let contents = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        data = contents
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

struct ContentView: View {
    @StateObject private var model = FilterModel()
    @StateObject private var host = CoordinatorHost()
    @State private var revision = 0
    @State private var importing = false
    @State private var exportDocument: PNGDocument?
    @State private var exporting = false
    @State private var showingControls = false

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var sizeClass
    #endif

    var body: some View {
        Group {
            #if os(iOS)
            // iPad regular width gets the same side-by-side layout as the Mac; only the
            // phone needs the controls moved into a sheet.
            if sizeClass == .compact { compactLayout } else { splitLayout }
            #else
            splitLayout
            #endif
        }
        .onAppear {
            wire()
            if let url = HeadlessMode.preloadURL { load(url) }
        }
        .onChange(of: model.configurationSignature) { _ in push() }
        .onChange(of: model.sourceMode) { _ in push() }
        .onChange(of: model.selectedCameraID) { _ in push() }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                DispatchQueue.main.async { load(url) }
            }
            return true
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.image]) { result in
            switch result {
            case .success(let url): load(url)
            case .failure(let error): model.status = "Could not open: \(error.localizedDescription)"
            }
        }
        .fileExporter(isPresented: $exporting,
                      document: exportDocument,
                      contentType: .png,
                      defaultFilename: "stretched") { result in
            switch result {
            case .success(let url): model.status = "Saved \(url.lastPathComponent)."
            case .failure(let error): model.status = "Could not save: \(error.localizedDescription)"
            }
            exportDocument = nil
        }
    }

    // MARK: - Layouts

    private var splitLayout: some View {
        HStack(spacing: 0) {
            preview
            Divider()
            ControlsView(model: model, onOpenImage: { importing = true }, onExport: exportImage)
        }
    }

    #if os(iOS)
    /// iPhone: the preview owns the screen and the controls live in a sheet that can be
    /// dragged down to a peek. Background interaction stays enabled so the preview is
    /// still visible — and still updating — while a slider is being dragged.
    private var compactLayout: some View {
        preview
            .ignoresSafeArea(edges: .bottom)
            .overlay(alignment: .bottom) {
                if !showingControls {
                    Button {
                        showingControls = true
                    } label: {
                        Label("Adjust", systemImage: "slider.horizontal.3")
                            .padding(.horizontal, 18)
                            .padding(.vertical, 11)
                            .background(.thinMaterial, in: Capsule())
                    }
                    .padding(.bottom, 28)
                }
            }
            .sheet(isPresented: $showingControls) {
                ControlsView(model: model,
                             onOpenImage: { importing = true },
                             onExport: exportImage)
                    .presentationDetents([.height(260), .large])
                    .presentationBackgroundInteraction(.enabled(upThrough: .large))
                    .presentationDragIndicator(.visible)
            }
    }
    #endif

    private var preview: some View {
        ZStack {
            MetalPreview(coordinator: host.coordinator,
                         isLive: model.sourceMode == .camera,
                         revision: revision)
            if model.sourceSize != .zero {
                RegionSelectionOverlay(imageSize: model.sourceSize,
                                       regionOfInterest: $model.regionOfInterest)
            } else {
                VStack(spacing: 10) {
                    Image(systemName: model.sourceMode == .camera
                          ? "camera.metering.unknown" : "photo.on.rectangle.angled")
                        .font(.system(size: 40, weight: .thin))
                    Text(model.sourceMode == .camera
                         ? "Waiting for the camera…"
                         : "Open an image, or drag one in")
                        .font(.callout)
                        .multilineTextAlignment(.center)
                }
                .foregroundStyle(.secondary)
                .padding()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Actions

    private func wire() {
        host.coordinator.onDiagnostics = { diagnostics in
            model.eigenvalues = diagnostics.eigenvalues
            model.wasRegularized = diagnostics.wasRegularized
            model.sampleCount = diagnostics.sampleCount
            model.frameMilliseconds = diagnostics.milliseconds
        }
        host.coordinator.onStatus = { model.status = $0 }
        host.coordinator.onSourceSize = { size in
            guard model.sourceSize != size else { return }
            // Rotating swaps the buffer's width and height, and the region of interest is
            // stored in source pixels - so a selection made in portrait would point at the
            // wrong part of a landscape frame. Drop it rather than silently mis-measure.
            if model.sourceSize != .zero, model.regionOfInterest != nil {
                model.regionOfInterest = nil
            }
            model.sourceSize = size
        }
        host.coordinator.onDevices = { devices in
            model.cameras = devices
            // Adopt the system default the first time, so the picker shows a real
            // selection rather than an empty one.
            if model.selectedCameraID == nil || !devices.contains(where: { $0.id == model.selectedCameraID }) {
                model.selectedCameraID = devices.first?.id
            }
        }
        host.coordinator.refreshCameraList()
        push()
    }

    private func push() {
        host.coordinator.update(configuration: model.configuration,
                                mode: model.sourceMode,
                                cameraID: model.selectedCameraID)
        revision &+= 1
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
        do {
            exportDocument = PNGDocument(data: try DSImageIO.pngData(from: image))
            exporting = true
        } catch {
            model.status = "Could not encode the image: \(error)"
        }
    }
}

/// Keeps one RenderCoordinator alive for the window's lifetime.
final class CoordinatorHost: ObservableObject {
    let coordinator = RenderCoordinator()
}
