import AVFoundation
import CoreVideo
import UIKit

/// 相机捕获层:负责 30fps 视频帧采集,输出 CVPixelBuffer 给下游(分割/姿态/合成)。
/// 注意:AVFoundation 会话必须在专用串行队列操作,故本类不做 @MainActor 隔离;
/// @Published 状态更新统一 hop 到主线程。
final class CameraLayer: NSObject, ObservableObject {
    enum Error: Swift.Error {
        case cameraUnavailable
        case cannotAddInput
        case cannotAddOutput
    }

    @Published private(set) var isRunning = false
    @Published private(set) var isAuthorized = false

    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.wear.camera.session")
    private let videoOutput = AVCaptureVideoDataOutput()
    /// 帧数据(相机线程写入/读取;Swift 5.9 无 nonisolated(unsafe),用注释说明)
    private var currentFrame: CVPixelBuffer?

    /// 每帧回调(由相机捕获线程调用)
    var onFrame: ((CVPixelBuffer) -> Void)?

    /// 请求相机权限
    func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            await MainActor.run { self.isAuthorized = true }
            return true
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            await MainActor.run { self.isAuthorized = granted }
            return granted
        default:
            await MainActor.run { self.isAuthorized = false }
            return false
        }
    }

    /// 启动会话(需先完成授权)
    func start() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.configureSession()
            self.session.startRunning()
            DispatchQueue.main.async { self.isRunning = self.session.isRunning }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.stopRunning()
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    private func configureSession() {
        guard !session.isRunning else { return }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        // 输入:后置或前置摄像头
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(for: .video) else {
            session.commitConfiguration()
            return
        }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input) }
        } catch {
            session.commitConfiguration()
            return
        }

        // 输出:像素缓冲
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        // 前置摄像头镜像,保证预览方向正确
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoMirroringSupported { connection.isVideoMirrored = true }
            if connection.isVideoRotationSupported { connection.videoRotationAngle = 90 }
        }

        session.commitConfiguration()
    }

    /// 取最新一帧(供增强层取关键帧用)
    func captureCurrentFrame() -> CVPixelBuffer? {
        currentFrame
    }
}

extension CameraLayer: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        currentFrame = pixelBuffer
        onFrame?(pixelBuffer)
    }
}
