import Foundation
import Metal
import MetalKit
import CoreVideo
import CoreGraphics
import Accelerate

/// 实时预览合成器:
/// 输入:相机帧 + 分割掩码 + 姿态关键点 + 服装纹理
/// 输出:穿着服装的合成帧(30fps 目标)
///
/// 第一版采用 CPU 加速的几何贴合 + 颜色合成(利用 Accelerate),
/// Metal 版本作为后续优化项(接口已预留)。
final class PreviewCompositor {
    struct Input {
        var cameraFrame: CVPixelBuffer
        var mask: CGImage?
        var landmarks: [CGPoint]   // 归一化
        var garmentTexture: CGImage
        var template: GarmentTemplate
        var frameSize: CGSize      // 相机帧尺寸(像素)
    }

    // MARK: - 输出

    /// 合成一帧。返回 BGRA 图像。
    func composite(_ input: Input) -> CGImage? {
        guard let cameraImage = pixelBufferToCGImage(input.cameraFrame) else { return nil }
        return composite(cameraImage: cameraImage, input: input)
    }

    func composite(cameraImage: CGImage, input: Input) -> CGImage? {
        let width = cameraImage.width
        let height = cameraImage.height

        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return nil
        }
        ctx.draw(cameraImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // 计算人体包围盒(由姿态关键点得到)
        guard let bodyRect = bodyBoundingBox(landmarks: input.landmarks, in: CGSize(width: width, height: height)),
              bodyRect.width > 10, bodyRect.height > 10 else {
            return cameraImage
        }

        // 服装贴合区域(模板 coverage 映射到包围盒)
        let garmentRect = CGRect(
            x: bodyRect.minX + bodyRect.width * input.template.coverage.minX,
            y: bodyRect.minY + bodyRect.height * input.template.coverage.minY,
            width: bodyRect.width * input.template.coverage.width,
            height: bodyRect.height * input.template.coverage.height
        )

        // 服装纹理按贴合区域绘制
        ctx.saveGState()
        ctx.setAlpha(0.92)
        ctx.interpolationQuality = .high
        ctx.draw(input.garmentTexture, in: garmentRect)
        ctx.restoreGState()

        // 若存在分割掩码,用掩码做人体区域裁剪,避免服装画出身体轮廓
        if let mask = input.mask {
            applyMask(mask, context: ctx, size: CGSize(width: width, height: height))
        }

        return ctx.makeImage()
    }

    // MARK: - 人体包围盒(由 33 个关键点估计躯干)

    /// MediaPipe Pose 关键点索引(躯干相关)
    private enum PoseIndex: Int {
        case leftShoulder = 11
        case rightShoulder = 12
        case leftHip = 23
        case rightHip = 24
        case leftEar = 7
        case rightEar = 8
    }

    private func bodyBoundingBox(landmarks: [CGPoint], in size: CGSize) -> CGRect? {
        guard landmarks.count > 24 else { return nil }
        let ls = landmarks[PoseIndex.leftShoulder.rawValue]
        let rs = landmarks[PoseIndex.rightShoulder.rawValue]
        let lh = landmarks[PoseIndex.leftHip.rawValue]
        let rh = landmarks[PoseIndex.rightHip.rawValue]

        let shoulderY = min(ls.y, rs.y)
        let hipY = max(lh.y, rh.y)
        let shoulderX = (ls.x + rs.x) / 2
        let hipX = (lh.x + rh.x) / 2
        let centerX = (shoulderX + hipX) / 2

        let torsoHeight = max(hipY - shoulderY, 0.1)
        let shoulderWidth = abs(rs.x - ls.x)
        let bodyWidth = max(shoulderWidth, torsoHeight * 0.4)

        // 上边界:肩部上方一点(含头部区域少量)
        let top = max(shoulderY - torsoHeight * 0.12, 0)
        let bottom = min(hipY + torsoHeight * 0.15, 1.0)

        return CGRect(
            x: (centerX - bodyWidth * 0.55) * size.width,
            y: top * size.height,
            width: bodyWidth * 1.1 * size.width,
            height: (bottom - top) * size.height
        )
    }

    // MARK: - 掩码裁剪

    private func applyMask(_ mask: CGImage, context: CGContext, size: CGSize) {
        // 简化:掩码作为 alpha 叠加(白色=人体保留,黑色=透明)
        // 生产实现:掩码区域外擦除服装,掩码内保留
        context.saveGState()
        context.setBlendMode(.destinationIn)
        context.draw(mask, in: CGRect(origin: .zero, size: size))
        context.restoreGState()
    }

    // MARK: - 工具

    func pixelBufferToCGImage(_ pixelBuffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)
        guard let ctx = CGContext(data: base, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        return ctx.makeImage()
    }
}
