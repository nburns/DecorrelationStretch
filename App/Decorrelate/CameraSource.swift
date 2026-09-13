import AVFoundation
import CoreVideo
#if os(iOS)
import UIKit
#endif

/// Minimal capture session feeding BGRA pixel buffers to a callback.
final class CameraSource: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "decorrelate.camera")
    private var configured = false
    /// Written on the main thread, read on `queue`. Cached rather than queried inline so
    /// the capture path never blocks on main - a `main.sync` here would deadlock the
    /// session the first time it ran from a main-thread caller.
    private let angleLock = NSLock()
    private var cachedAngle: CGFloat = 90

    var onFrame: ((CVPixelBuffer) -> Void)?
    var onError: ((String) -> Void)?

    var isRunning: Bool { session.isRunning }

    override init() {
        super.init()
        #if os(iOS)
        cachedAngle = Self.rotationAngle()
        NotificationCenter.default.addObserver(
            self, selector: #selector(orientationChanged),
            name: UIDevice.orientationDidChangeNotification, object: nil)
        #endif
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            guard granted else {
                DispatchQueue.main.async {
                    #if os(iOS)
                    self.onError?("Camera access was denied. Enable it in Settings > Privacy > Camera.")
                    #else
                    self.onError?("Camera access was denied in System Settings > Privacy.")
                    #endif
                }
                return
            }
            self.queue.async { self.configureAndRun() }
        }
    }

    private func configureAndRun() {
        if !configured {
            session.beginConfiguration()
            session.sessionPreset = .high

            guard let device = AVCaptureDevice.default(for: .video),
                  let input = try? AVCaptureDeviceInput(device: device),
                  session.canAddInput(input) else {
                session.commitConfiguration()
                DispatchQueue.main.async { self.onError?("No usable video capture device was found.") }
                return
            }
            session.addInput(input)

            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: queue)
            guard session.canAddOutput(output) else {
                session.commitConfiguration()
                DispatchQueue.main.async { self.onError?("Could not attach the video output.") }
                return
            }
            session.addOutput(output)
            session.commitConfiguration()
            configured = true
        }

        applyRotation()
        if !session.isRunning { session.startRunning() }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    // MARK: - Orientation

    /// Sensor buffers arrive in the camera's native landscape orientation, so on a phone
    /// they have to be rotated to match how the device is being held. Handled here rather
    /// than in the shader so the filter never has to know about device orientation.
    private func applyRotation() {
        #if os(iOS)
        guard let connection = output.connection(with: .video) else { return }
        angleLock.lock()
        let angle = cachedAngle
        angleLock.unlock()
        if connection.isVideoRotationAngleSupported(angle) {
            connection.videoRotationAngle = angle
        }
        #endif
    }

    #if os(iOS)
    @objc private func orientationChanged() {
        let angle = Self.rotationAngle()      // notification is delivered on main
        angleLock.lock()
        cachedAngle = angle
        angleLock.unlock()
        queue.async { [weak self] in self?.applyRotation() }
    }

    private static func rotationAngle() -> CGFloat {
        switch UIDevice.current.orientation {
        case .landscapeLeft: return 180
        case .landscapeRight: return 0
        case .portraitUpsideDown: return 270
        default: return 90
        }
    }
    #endif

    // MARK: - Delegate

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(buffer)
    }
}
