import XCTest
import CoreGraphics
import simd
@testable import DecorrelationStretch

/// The GPU scale and the view-coordinate rect are computed separately but must agree, or
/// a region-of-interest drag lands somewhere other than what is on screen.
final class DSPreviewGeometryTests: XCTestCase {

    private let viewport = CGSize(width: 400, height: 800)     // tall, like a phone
    private let landscape = CGSize(width: 1600, height: 900)
    private let portrait = CGSize(width: 900, height: 1600)

    // MARK: - Fit

    func testFitLetterboxesWideContentInATallViewport() {
        let rect = DSPreviewGeometry.contentRect(content: landscape, viewport: viewport, mode: .fit)
        XCTAssertEqual(rect.width, 400, accuracy: 1e-6, "should span the full width")
        XCTAssertEqual(rect.height, 225, accuracy: 1e-6)
        XCTAssertEqual(rect.minX, 0, accuracy: 1e-6)
        XCTAssertEqual(rect.minY, (800 - 225) / 2, accuracy: 1e-6, "should be vertically centred")
    }

    func testFitNeverOverflowsTheViewport() {
        for content in [landscape, portrait, CGSize(width: 100, height: 100)] {
            let rect = DSPreviewGeometry.contentRect(content: content, viewport: viewport, mode: .fit)
            XCTAssertLessThanOrEqual(rect.width, viewport.width + 1e-6)
            XCTAssertLessThanOrEqual(rect.height, viewport.height + 1e-6)
        }
    }

    // MARK: - Fill

    func testFillCoversTheViewport() {
        for content in [landscape, portrait, CGSize(width: 100, height: 100)] {
            let rect = DSPreviewGeometry.contentRect(content: content, viewport: viewport, mode: .fill)
            XCTAssertGreaterThanOrEqual(rect.width, viewport.width - 1e-6, "\(content) left a gap")
            XCTAssertGreaterThanOrEqual(rect.height, viewport.height - 1e-6, "\(content) left a gap")
        }
    }

    /// A 16:9 camera frame on a 1:2 screen is the case that looked broken: fit leaves
    /// heavy bars, fill crops the sides instead.
    func testFillCropsWideContentRatherThanLetterboxing() {
        let rect = DSPreviewGeometry.contentRect(content: landscape, viewport: viewport, mode: .fill)
        XCTAssertEqual(rect.height, 800, accuracy: 1e-6, "should span the full height")
        XCTAssertGreaterThan(rect.width, viewport.width, "sides should overflow and be cropped")
        XCTAssertLessThan(rect.minX, 0, "overflow should be centred, not offset to one side")
    }

    // MARK: - Aspect is preserved, and the two computations agree

    func testAspectRatioIsPreservedInBothModes() {
        for mode in [DSPreviewGeometry.ContentMode.fit, .fill] {
            for content in [landscape, portrait] {
                let rect = DSPreviewGeometry.contentRect(content: content, viewport: viewport, mode: mode)
                XCTAssertEqual(rect.width / rect.height,
                               content.width / content.height,
                               accuracy: 1e-6, "\(mode) distorted \(content)")
            }
        }
    }

    /// The quad scale and the content rect are derived independently; this is the test
    /// that stops them drifting apart.
    func testQuadScaleMatchesContentRect() {
        for mode in [DSPreviewGeometry.ContentMode.fit, .fill] {
            for content in [landscape, portrait, CGSize(width: 640, height: 640)] {
                let scale = DSPreviewGeometry.quadScale(content: content, viewport: viewport, mode: mode)
                let rect = DSPreviewGeometry.contentRect(content: content, viewport: viewport, mode: mode)
                XCTAssertEqual(Double(scale.x), rect.width / viewport.width,
                               accuracy: 1e-5, "\(mode) \(content) x")
                XCTAssertEqual(Double(scale.y), rect.height / viewport.height,
                               accuracy: 1e-5, "\(mode) \(content) y")
            }
        }
    }

    func testDegenerateSizesDoNotProduceNaN() {
        for content in [CGSize.zero, CGSize(width: 0, height: 100), CGSize(width: 100, height: 0)] {
            let scale = DSPreviewGeometry.quadScale(content: content, viewport: viewport, mode: .fill)
            XCTAssertTrue(scale.x.isFinite && scale.y.isFinite)
            let rect = DSPreviewGeometry.contentRect(content: content, viewport: .zero, mode: .fit)
            XCTAssertEqual(rect, .zero)
        }
    }
}
