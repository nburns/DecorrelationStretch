import AVFoundation
import CoreVideo

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

    /// Written from the rotation observer, read on `queue`. Cached rather than queried
    /// inline so the capture path never blocks on main - a `main.sync` here would
    /// deadlock the session the first time it ran from a main-thread caller.
    private let angleLock = NSLock()
    private var cachedAngle: CGFloat = 90

    #if os(iOS)
    /// AVFoundation's own answer to "which way is up". Hand-rolling this from
    /// UIDevice.orientation does not work: that property reports .unknown unless
    /// beginGeneratingDeviceOrientationNotifications() has been called, and
    /// UIDeviceOrientation.landscapeLeft is the opposite of
    /// UIInterfaceOrientation.landscapeLeft, so the obvious mapping is inverted. The
    /// coordinator also accounts for the sensor's own mounting per device.
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?
    #endif

    var onFrame: ((CVPixelBuffer) -> Void)?
    /// Carries both failures and plain status, e.g. which device is now live.
    var onStatus: ((String) -> Void)?
    var onDevices: (([CameraDevice]) -> Void)?

    var isRunning: Bool { session.isRunning }

    override init() {
        super.init()
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
        #if os(iOS)
        // The coordinator is per-device, so it is rebuilt on every camera switch.
        observeRotation(for: device)
        #endif

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

    /// Sensor buffers arrive in the camera's native landscape orientation, so they have
    /// to be rotated to match how the device is being held. Applied to the connection
    /// rather than in the shader, so the filter never has to know about orientation.
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
    /// `videoRotationAngleForHorizonLevelPreview` is the angle that keeps a preview
    /// upright while the interface rotates with the device, which is exactly this case.
    /// Observed rather than polled, so rotation is followed without a notification dance.
    private func observeRotation(for device: AVCaptureDevice) {
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: nil)
        rotationCoordinator = coordinator
        rotationObservation = coordinator.observe(
            \.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]
        ) { [weak self] coordinator, _ in
            guard let self else { return }
            let angle = coordinator.videoRotationAngleForHorizonLevelPreview
            self.angleLock.lock()
            self.cachedAngle = angle
            self.angleLock.unlock()
            self.queue.async { self.applyRotation() }
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
