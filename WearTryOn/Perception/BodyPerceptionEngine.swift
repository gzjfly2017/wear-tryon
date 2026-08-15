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
        // 优先 Bundle 根,其次 Models/ 子目录(XcodeGen folder 资源)
        let modelPath = Bundle.main.path(forResource: "selfie_segmenter", ofType: "tflite")
            ?? Bundle.main.path(forResource: "selfie_segmenter", ofType: "tflite", inDirectory: "Models")
        guard let modelPath else {
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
        let modelPath = Bundle.main.path(forResource: "pose_landmarker_lite", ofType: "task")
            ?? Bundle.main.path(forResource: "pose_landmarker_lite", ofType: "task", inDirectory: "Models")
        guard let modelPath else {
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
        queue.sync { () -> BodyPerception? in
            guard let segmenter, let poseLandmarker,
                  let image = try? MPImage(pixelBuffer: pixelBuffer) else {
                return nil
            }

            // 分割掩码:从 Mask 的原始像素数据构建 CGImage
            var maskImage: CGImage?
            if let result = try? segmenter.segment(image: image),
               let mask = result.categoryMask {
                maskImage = Self.maskToCGImage(mask)
            }

            // 姿态关键点
            var landmarks: [NormalizedLandmark] = []
            if let result = try? poseLandmarker.detect(image: image) {
                landmarks = result.landmarks.first ?? []
            }

            return BodyPerception(segmentationMask: maskImage, landmarks: landmarks)
        }
    }

    /// 把 MediaPipe Mask 的 uint8 像素数据转为 CGImage(白色=人体,黑色=背景)
    private static func maskToCGImage(_ mask: Mask) -> CGImage? {
        let w = mask.width
        let h = mask.height
        guard w > 0, h > 0 else { return nil }
        let data = mask.uint8Data  // 非可选 UnsafePointer<UInt8>

        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) {
            let v = data[i]  // 0-255,类别掩码
            pixels[i * 4] = v
            pixels[i * 4 + 1] = v
            pixels[i * 4 + 2] = v
            pixels[i * 4 + 3] = 255
        }
        guard let ctx = CGContext(data: &pixels, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let cg = ctx.makeImage() else {
            return nil
        }
        return cg
    }

    deinit {
        segmenter = nil
        poseLandmarker = nil
    }
}
