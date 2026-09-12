import SwiftUI
import MetalKit

/// Hosts the MTKView and redraws on demand rather than continuously: a still image only
/// needs a new frame when a control moves, and re-rendering a 4K source at 60fps to show
/// an unchanged picture is wasted power.
struct MetalPreview: NSViewRepresentable {
    let coordinator: RenderCoordinator
    let isLive: Bool
    let revision: Int

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: coordinator.device)
        view.delegate = coordinator
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColorMake(0.09, 0.09, 0.10, 1)
        view.autoResizeDrawable = true
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        view.isPaused = !isLive
        view.enableSetNeedsDisplay = !isLive
        if isLive {
            view.preferredFramesPerSecond = 60
        } else {
            view.needsDisplay = true
        }
    }
}

/// Drag to restrict the statistics to part of the frame. Coordinates are converted into
/// source-image pixels, since that is what the analysis pass needs.
struct RegionSelectionOverlay: View {
    let imageSize: CGSize
    @Binding var regionOfInterest: CGRect?

    @State private var dragStart: CGPoint?
    @State private var dragCurrent: CGPoint?

    var body: some View {
        GeometryReader { geometry in
            let fitted = PreviewGeometry.fittedRect(imageSize: imageSize, in: geometry.size)
            ZStack(alignment: .topLeading) {
                Color.clear.contentShape(Rectangle())
                if let rect = liveRect(in: fitted) {
                    Rectangle()
                        .strokeBorder(Color.accentColor, lineWidth: 1.5)
                        .background(Color.accentColor.opacity(0.12))
                        .frame(width: rect.width, height: rect.height)
                        .offset(x: rect.minX, y: rect.minY)
                        .allowsHitTesting(false)
                }
            }
            .gesture(
                DragGesture(minimumDistance: 3)
                    .onChanged { value in
                        guard fitted.width > 0 else { return }
                        if dragStart == nil { dragStart = value.startLocation }
                        dragCurrent = value.location
                    }
                    .onEnded { _ in
                        defer { dragStart = nil; dragCurrent = nil }
                        guard let rect = liveRect(in: fitted), fitted.width > 0 else { return }
                        let scale = imageSize.width / fitted.width
                        let converted = CGRect(
                            x: (rect.minX - fitted.minX) * scale,
                            y: (rect.minY - fitted.minY) * scale,
                            width: rect.width * scale,
                            height: rect.height * scale
                        ).intersection(CGRect(origin: .zero, size: imageSize))
                        regionOfInterest = converted.width >= 8 && converted.height >= 8 ? converted : nil
                    }
            )
            .overlay(alignment: .topLeading) {
                if dragStart == nil, let roi = regionOfInterest, fitted.width > 0 {
                    let scale = fitted.width / imageSize.width
                    let rect = CGRect(x: fitted.minX + roi.minX * scale,
                                      y: fitted.minY + roi.minY * scale,
                                      width: roi.width * scale, height: roi.height * scale)
                    Rectangle()
                        .strokeBorder(Color.accentColor, lineWidth: 1.5)
                        .frame(width: rect.width, height: rect.height)
                        .offset(x: rect.minX, y: rect.minY)
                        .allowsHitTesting(false)
                }
            }
        }
    }

    private func liveRect(in fitted: CGRect) -> CGRect? {
        guard let start = dragStart, let current = dragCurrent else { return nil }
        let rect = CGRect(x: min(start.x, current.x), y: min(start.y, current.y),
                          width: abs(current.x - start.x), height: abs(current.y - start.y))
        return rect.intersection(fitted)
    }
}
