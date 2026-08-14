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
///
/// 对齐官方 google-ai-edge/mediapipe-samples 的 API 用法:
///   - Options 用无参构造 + baseOptions.modelAssetPath
///   - 单帧同步推理使用 segment(image:) / detect(image:)
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
        let options = ImageSegmenterOptions()
        options.runningMode = .video
        options.shouldOutputCategoryMask = true
        options.shouldOutputConfidenceMasks = false
        options.baseOptions.modelAssetPath = modelPath
        do {
            segmenter = try ImageSegmenter(options: options)
        } catch {
            throw Error.segmentationInitFailed
        }
    }

    private func loadPoseLandmarker() throws {
        guard let modelPath = Bundle.main.path(forResource: "pose_landmarker_lite", ofType: "task") else {
            throw Error.modelNotFound("pose_landmarker_lite.task")
        }
        let options = PoseLandmarkerOptions()
        options.runningMode = .video
        options.numPoses = 1
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        options.baseOptions.modelAssetPath = modelPath
        do {
            poseLandmarker = try PoseLandmarker(options: options)
        } catch {
            throw Error.poseInitFailed
        }
    }

    // MARK: - 推理

    /// 对单帧做分割 + 姿态(内部串行队列,帧率由调用方节流)。
    func process(pixelBuffer: CVPixelBuffer) -> BodyPerception? {
        queue.sync {
            guard let segmenter, let poseLandmarker,
                  let image = try? MPImage(pixelBuffer: pixelBuffer) else {
                return nil
            }

            let maskImage: CGImage? = {
                guard let result = try? segmenter.segment(image: image) else { return nil }
                return result.categoryMask?.image
            }()

            let landmarks: [NormalizedLandmark] = {
                guard let result = try? poseLandmarker.detect(image: image) else { return [] }
                return result.landmarks.first ?? []
            }()

            return BodyPerception(segmentationMask: maskImage, landmarks: landmarks)
        }
    }

    deinit {
        segmenter = nil
        poseLandmarker = nil
    }
}
