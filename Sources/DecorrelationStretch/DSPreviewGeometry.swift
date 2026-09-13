import CoreGraphics
import simd

/// Fitting a rendered frame into a view.
///
/// Lives here rather than in a view layer because the two halves have to agree exactly:
/// the GPU needs a scale for the quad, and the UI needs the matching rect in view
/// coordinates for hit-testing. Deriving them separately is how a region-of-interest
/// selection ends up pointing somewhere other than what is on screen.
public enum DSPreviewGeometry {

    public enum ContentMode: Sendable {
        /// Whole frame visible, letterboxed on the axis that does not fill.
        case fit
        /// Frame covers the view, cropped on the axis that overflows.
        case fill
    }

    /// Scale for a unit quad centred on the viewport.
    ///
    /// `fill` is the reciprocal arrangement of `fit`: the quad grows past the viewport on
    /// one axis and the rasteriser clips it, so the same vertex shader serves both.
    public static func quadScale(content: CGSize,
                                 viewport: CGSize,
                                 mode: ContentMode) -> SIMD2<Float> {
        guard content.width > 0, content.height > 0,
              viewport.width > 0, viewport.height > 0 else { return SIMD2(1, 1) }
        let contentAspect = content.width / content.height
        let viewportAspect = viewport.width / viewport.height
        let wider = contentAspect > viewportAspect
        let shrinkVertically = (mode == .fill) ? !wider : wider
        return shrinkVertically
            ? SIMD2(1, Float(viewportAspect / contentAspect))
            : SIMD2(Float(contentAspect / viewportAspect), 1)
    }

    /// Where the content lands in view coordinates. Under `.fill` the rect extends beyond
    /// the viewport, which is correct: that is the part being cropped away.
    public static func contentRect(content: CGSize,
                                   viewport: CGSize,
                                   mode: ContentMode) -> CGRect {
        guard content.width > 0, content.height > 0,
              viewport.width > 0, viewport.height > 0 else { return .zero }
        let byWidth = viewport.width / content.width
        let byHeight = viewport.height / content.height
        let scale = (mode == .fill) ? max(byWidth, byHeight) : min(byWidth, byHeight)
        let size = CGSize(width: content.width * scale, height: content.height * scale)
        return CGRect(x: (viewport.width - size.width) / 2,
                      y: (viewport.height - size.height) / 2,
                      width: size.width, height: size.height)
    }
}
