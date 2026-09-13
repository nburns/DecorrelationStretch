import AVFoundation
import CoreVideo
#if os(iOS)
import UIKit
#endif

/// One selectable capture device.
struct CameraDevice: Identifiable, Hashable {
    let id: String
    let name: String
    let isFront: Bool
}

/// Capture session feeding BGRA pixel buffers to a callback, with live device switching.
final class CameraSource: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "decorrelate.camera")
    private var currentInput: AVCaptureDeviceInput?
    private var outputAttached = false
    private var requestedDeviceID: String?

    /// Written on the main thread, read on `queue`. Cached rather than queried inline so
    /// the capture path never blocks on main - a `main.sync` here would deadlock the
    /// session the first time it ran from a main-thread caller.
    private let angleLock = NSLock()
    private var cachedAngle: CGFloat = 90

    var onFrame: ((CVPixelBuffer) -> Void)?
    /// Carries both failures and plain status, e.g. which device is now live.
    var onStatus: ((String) -> Void)?
    var onDevices: (([CameraDevice]) -> Void)?

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

    // MARK: - Discovery

    private static var deviceTypes: [AVCaptureDevice.DeviceType] {
        #if os(iOS)
        // Wide angle covers front and back; the others are back-only extras.
        return [.builtInWideAngleCamera, .builtInUltraWideCamera, .builtInTelephotoCamera]
        #else
        var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera]
        if #available(macOS 14.0, *) {
            types.append(.external)
            types.append(.continuityCamera)
        } else {
            types.append(.externalUnknown)
        }
        return types
        #endif
    }

    static func discover() -> [CameraDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes, mediaType: .video, position: .unspecified
        ).devices.map {
            CameraDevice(id: $0.uniqueID, name: $0.localizedName, isFront: $0.position == .front)
        }
    }

    /// Publishes the device list. Called again after access is granted, because names can
    /// be withheld until then.
    func refreshDevices() {
        let devices = Self.discover()
        DispatchQueue.main.async { self.onDevices?(devices) }
    }

    // MARK: - Lifecycle

    func start(deviceID: String? = nil) {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            guard granted else {
                DispatchQueue.main.async {
                    #if os(iOS)
                    self.onStatus?("Camera access was denied. Enable it in Settings > Privacy > Camera.")
                    #else
                    self.onStatus?("Camera access was denied in System Settings > Privacy.")
                    #endif
                }
                return
            }
            self.refreshDevices()
            self.queue.async {
                self.requestedDeviceID = deviceID ?? self.requestedDeviceID
                self.configureAndRun()
            }
        }
    }

    /// Switches device without tearing the session down, so the preview does not blink.
    func select(deviceID: String) {
        queue.async { [weak self] in
            guard let self else { return }
            guard self.requestedDeviceID != deviceID else { return }
            self.requestedDeviceID = deviceID
            self.configureAndRun()
        }
    }

    private func resolveDevice() -> AVCaptureDevice? {
        if let id = requestedDeviceID,
           let match = Self.discover().first(where: { $0.id == id }),
           let device = AVCaptureDevice(uniqueID: match.id) {
            return device
        }
        return AVCaptureDevice.default(for: .video)
    }

    private func configureAndRun() {
        guard let device = resolveDevice() else {
            DispatchQueue.main.async { self.onStatus?("No usable video capture device was found.") }
            return
        }
        if currentInput?.device.uniqueID == device.uniqueID, session.isRunning { return }

        session.beginConfiguration()
        session.sessionPreset = .high

        if let existing = currentInput {
            session.removeInput(existing)
            currentInput = nil
        }
        guard let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async {
                self.onStatus?("Could not open \(device.localizedName).")
            }
            return
        }
        session.addInput(input)
        currentInput = input

        if !outputAttached {
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: queue)
            guard session.canAddOutput(output) else {
                session.commitConfiguration()
                DispatchQueue.main.async { self.onStatus?("Could not attach the video output.") }
                return
            }
            session.addOutput(output)
            outputAttached = true
        }
        session.commitConfiguration()

        // The connection is rebuilt when inputs change, so the rotation has to be
        // re-applied after every reconfiguration, not just at first start.
        applyRotation()
        if !session.isRunning { session.startRunning() }

        let name = device.localizedName
        DispatchQueue.main.async { self.onStatus?("Camera: \(name)") }
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
