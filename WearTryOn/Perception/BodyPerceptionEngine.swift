import Foundation
import CoreVideo
import UIKit
import MediaPipeTasksVision

/// 人体感知结果:分割掩码 + 姿态关键点
struct BodyPerception {
    /// 分割掩码(0-255,单通道,与输入帧同尺寸),255=人体区域
    var segmentationMask: CGImage?
    /// 姿态关键点(33 个,归一化坐标)
    var landmarks: [NormalizedLandmark] = []
    /// 是否有有效人体
    var hasPerson: Bool { !landmarks.isEmpty }
}

/// MediaPipe 分割 + 姿态封装(实时预览层核心感知)
/// 两个 task 独立运行,帧率由调用方节流。
final class BodyPerceptionEngine {
    enum Error: Swift.Error {
        case modelNotFound(String)
        case segmentationInitFailed
        case poseInitFailed
    }

    private var segmenter: ImageSegmenter?
    private var poseLandmarker: PoseLandmarker?
    private let queue = DispatchQueue(label: "com.wear.perception", qos: .userInitiated)

    // MARK: - 初始化

    func load() throws {
        try queue.sync {
            try loadSegmenter()
            try loadPoseLandmarker()
        }
    }

    private func loadSegmenter() throws {
        guard let modelPath = Bundle.main.path(forResource: "selfie_segmenter", ofType: "tflite") else {
            throw Error.modelNotFound("selfie_segmenter.tflite")
        }
        let options = ImageSegmenterOptions(modelPath: modelPath)
        options.runningMode = .video
        options.outputCategoryMask = true
        options.outputConfidenceMasks = false
        segmenter = try ImageSegmenter(options: options)
    }

    private func loadPoseLandmarker() throws {
        guard let modelPath = Bundle.main.path(forResource: "pose_landmarker_lite", ofType: "tflite") else {
            throw Error.modelNotFound("pose_landmarker_lite.tflite")
        }
        let options = PoseLandmarkerOptions(modelPath: modelPath)
        options.runningMode = .video
        options.numPoses = 1
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        poseLandmarker = try PoseLandmarker(options: options)
    }

    // MARK: - 推理

    /// 对单帧做分割 + 姿态。timestamp 必须单调递增(MediaPipe video 模式要求)。
    func process(pixelBuffer: CVPixelBuffer, timestamp: Int) -> BodyPerception? {
        queue.sync {
            guard let segmenter, let poseLandmarker else { return nil }
            guard let image = MPImage(pixelBuffer: pixelBuffer) else { return nil }

            let segmentationResult = try? segmenter.segment(videoFrame: image, timestampInMilliseconds: timestamp)
            let maskImage: CGImage? = segmentationResult?.categoryMask?.image

            let poseResult = try? poseLandmarker.detect(videoFrame: image, timestampInMilliseconds: timestamp)
            let landmarks = poseResult?.landmarks.first ?? []

            return BodyPerception(segmentationMask: maskImage, landmarks: landmarks)
        }
    }

    deinit {
        segmenter = nil
        poseLandmarker = nil
    }
}
