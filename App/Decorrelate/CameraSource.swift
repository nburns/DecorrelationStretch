import AVFoundation
import CoreVideo

/// Minimal capture session feeding BGRA pixel buffers to a callback.
final class CameraSource: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "studio.camera")
    private var configured = false

    var onFrame: ((CVPixelBuffer) -> Void)?
    var onError: ((String) -> Void)?

    var isRunning: Bool { session.isRunning }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            guard granted else {
                DispatchQueue.main.async { self.onError?("Camera access was denied in System Settings > Privacy.") }
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
        if !session.isRunning { session.startRunning() }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(buffer)
    }
}
