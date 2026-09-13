import SwiftUI
import simd
import Metal
import DecorrelationStretch

enum SourceMode: String, CaseIterable, Identifiable {
    case image = "Image"
    case camera = "Camera"
    var id: String { rawValue }
}

struct Preset: Identifiable, Hashable {
    let name: String
    let space: DSColorSpace
    var id: String { name }

    static let all: [Preset] = [
        .init(name: "Chroma Boost", space: .chromaBoost),
        .init(name: "Red Emphasis", space: .redEmphasis),
        .init(name: "Yellow Emphasis", space: .yellowEmphasis),
        .init(name: "Tonal Emphasis", space: .tonalEmphasis),
        .init(name: "RGB", space: .rgb),
        .init(name: "YUV", space: .yuv),
        .init(name: "LAB", space: .lab),
    ]
}

@MainActor
final class FilterModel: ObservableObject {

    // Colour space
    @Published var family: DSColorSpaceFamily = .yuv
    @Published var weightX: Float = 0.5
    @Published var weightY: Float = 1.5
    @Published var weightZ: Float = 1.5

    // Stretch
    @Published var useUniformTarget = true
    @Published var stretchFraction: Float = 0.0588
    @Published var amount: Float = 1

    // Analysis
    @Published var samplingStride = 1
    @Published var analysisInterval = 12
    @Published var temporalSmoothing: Float = 0.85

    /// In source-image pixel coordinates, origin top-left. Nil means the whole frame.
    @Published var regionOfInterest: CGRect?

    // Source
    @Published var sourceMode: SourceMode = .camera
    @Published var cameras: [CameraDevice] = []
    @Published var selectedCameraID: String?
    @Published var imageURL: URL?
    @Published var sourceSize: CGSize = .zero

    // Diagnostics, pushed back from the render loop
    @Published var eigenvalues: SIMD3<Float> = .one
    @Published var wasRegularized = false
    @Published var sampleCount = 0
    @Published var frameMilliseconds: Double = 0
    @Published var status = "Open an image or switch to the camera."

    var colorSpace: DSColorSpace {
        DSColorSpace(family: family, weights: SIMD3(weightX, weightY, weightZ))
    }

    var axisLabels: (String, String, String) {
        switch family {
        case .rgb: return ("R", "G", "B")
        case .yuv: return ("Y", "U", "V")
        case .lab: return ("L*", "a*", "b*")
        }
    }

    var configuration: DSConfiguration {
        var config = DSConfiguration()
        config.colorSpace = colorSpace
        config.target = useUniformTarget ? .uniform(fraction: stretchFraction) : .preserveOriginal
        config.amount = amount
        config.samplingStride = samplingStride
        config.analysisInterval = analysisInterval
        config.temporalSmoothing = temporalSmoothing
        if let roi = regionOfInterest, roi.width >= 8, roi.height >= 8 {
            config.regionOfInterest = MTLRegion(
                origin: MTLOrigin(x: Int(roi.minX), y: Int(roi.minY), z: 0),
                size: MTLSize(width: Int(roi.width), height: Int(roi.height), depth: 1)
            )
        }
        return config
    }

    func apply(preset: Preset) {
        family = preset.space.family
        weightX = preset.space.weights.x
        weightY = preset.space.weights.y
        weightZ = preset.space.weights.z
    }

    /// Weights are in the units of the current base space, so switching family makes the
    /// old values meaningless — reset rather than carry them across.
    func resetWeights() {
        weightX = 1; weightY = 1; weightZ = 1
    }

    /// Changes whenever anything the render path cares about changes, so the view can
    /// push a new configuration without comparing every field by hand.
    var configurationSignature: Int {
        var hasher = Hasher()
        hasher.combine(family)
        hasher.combine(weightX); hasher.combine(weightY); hasher.combine(weightZ)
        hasher.combine(useUniformTarget); hasher.combine(stretchFraction)
        hasher.combine(amount)
        hasher.combine(samplingStride); hasher.combine(analysisInterval)
        hasher.combine(temporalSmoothing)
        if let roi = regionOfInterest {
            hasher.combine(roi.minX); hasher.combine(roi.minY)
            hasher.combine(roi.width); hasher.combine(roi.height)
        }
        return hasher.finalize()
    }

    /// Conditioning of the colour cloud: how flat the pixel distribution is. Very high
    /// numbers mean one axis carries almost no signal and the stretch is amplifying
    /// mostly noise along it.
    var conditionNumber: Float {
        let smallest = max(eigenvalues.z, .leastNormalMagnitude)
        return eigenvalues.x / smallest
    }
}
