import Foundation
import Metal
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import DecorrelationStretch

/// Headless entry points, so the filter can be driven from a terminal and checked
/// without clicking through the UI.
///
///     Decorrelate --process in.png out.png [--space redEmphasis] [--stretch 0.08]
///     Decorrelate --open panel.png
enum HeadlessMode {
    private(set) static var preloadURL: URL?

    static func runIfRequested() {
        let arguments = CommandLine.arguments

        if let index = arguments.firstIndex(of: "--open"), arguments.count > index + 1 {
            preloadURL = URL(fileURLWithPath: arguments[index + 1]).standardizedFileURL
        }

        guard let index = arguments.firstIndex(of: "--process"), arguments.count > index + 2 else {
            return
        }
        let input = URL(fileURLWithPath: arguments[index + 1])
        let output = URL(fileURLWithPath: arguments[index + 2])

        var space = DSColorSpace.chromaBoost
        if let i = arguments.firstIndex(of: "--space"), arguments.count > i + 1,
           let match = Preset.all.first(where: {
               $0.name.replacingOccurrences(of: " ", with: "").lowercased() == arguments[i + 1].lowercased()
           }) {
            space = match.space
        }
        var fraction: Float = 0.0588
        if let i = arguments.firstIndex(of: "--stretch"), arguments.count > i + 1,
           let value = Float(arguments[i + 1]) {
            fraction = value
        }

        exit(process(input: input, output: output, space: space, fraction: fraction) ? 0 : 1)
    }

    private static func process(input: URL, output: URL,
                                space: DSColorSpace, fraction: Float) -> Bool {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            FileHandle.standardError.write(Data("no Metal device\n".utf8))
            return false
        }
        guard let source = CGImageSourceCreateWithURL(input as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            FileHandle.standardError.write(Data("could not read \(input.path)\n".utf8))
            return false
        }

        do {
            let engine = try DSEngine(device: device)
            var config = DSConfiguration()
            config.colorSpace = space
            config.target = .uniform(fraction: fraction)
            engine.configuration = config

            let started = CFAbsoluteTimeGetCurrent()
            let result = try engine.process(image: image, device: device, queue: queue)
            let elapsed = (CFAbsoluteTimeGetCurrent() - started) * 1000

            guard let destination = CGImageDestinationCreateWithURL(
                output as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                FileHandle.standardError.write(Data("could not create \(output.path)\n".utf8))
                return false
            }
            CGImageDestinationAddImage(destination, result, nil)
            guard CGImageDestinationFinalize(destination) else {
                FileHandle.standardError.write(Data("could not write \(output.path)\n".utf8))
                return false
            }

            let transform = engine.transform
            let message = """
            wrote \(output.lastPathComponent)  \(result.width)x\(result.height)  \
            \(String(format: "%.1f", elapsed)) ms
              eigenvalues  \(String(format: "%.4g  %.4g  %.4g",
                                    transform.eigenvalues.x, transform.eigenvalues.y, transform.eigenvalues.z))
              sampled      \(transform.sampleCount) px\
            \(transform.wasRegularized ? "\n  WARNING: colour planes nearly dependent" : "")

            """
            FileHandle.standardOutput.write(Data(message.utf8))
            return true
        } catch {
            FileHandle.standardError.write(Data("failed: \(error)\n".utf8))
            return false
        }
    }
}
