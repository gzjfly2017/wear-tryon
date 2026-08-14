import Foundation
import CoreGraphics

/// 关键帧选择器:在实时预览流中检测"姿态稳定"的时刻,
/// 触发端侧 VTON 增强(定格高清)。
final class FrameSelector {
    struct Sample {
        let timestamp: TimeInterval
        let poseJitter: CGFloat   // 关键点平均位移(归一化)
        let score: CGFloat        // 帧质量分(越高越好)
    }

    /// 触发回调(在主线程)
    var onStableFrame: ((FrameSelector.Sample) -> Void)?

    private var history: [Sample] = []
    private let maxHistory = 20
    private var lastTriggerTime: TimeInterval = 0
    private let cooldown: TimeInterval = 3.0      // 两次触发最小间隔(秒)
    private let stableThreshold: CGFloat = 0.012  // 归一化位移阈值
    private let stableWindow = 8                  // 需要连续稳定的帧数

    /// 喂入每帧的姿态关键点(归一化),内部计算抖动并检测稳定
    func update(landmarks: [CGPoint], timestamp: TimeInterval) {
        let jitter = computeJitter(landmarks: landmarks)

        history.append(Sample(timestamp: timestamp, poseJitter: jitter, score: score(for: jitter)))
        if history.count > maxHistory { history.removeFirst(history.count - maxHistory) }

        guard history.count >= stableWindow else { return }
        let window = history.suffix(stableWindow)

        // 连续稳定窗口
        let allStable = window.allSatisfy { $0.poseJitter < stableThreshold }
        guard allStable else { return }

        // 冷却期检查
        guard timestamp - lastTriggerTime >= cooldown else { return }
        lastTriggerTime = timestamp

        let best = window.max(by: { $0.score < $1.score }) ?? window[window.startIndex]
        onStableFrame?(best)
    }

    // MARK: - 内部

    private var prevLandmarks: [CGPoint] = []
    private func computeJitter(landmarks: [CGPoint]) -> CGFloat {
        guard !prevLandmarks.isEmpty, prevLandmarks.count == landmarks.count else {
            prevLandmarks = landmarks
            return 1.0
        }
        var total: CGFloat = 0
        for (a, b) in zip(prevLandmarks, landmarks) {
            total += hypot(a.x - b.x, a.y - b.y)
        }
        let avg = total / CGFloat(landmarks.count)
        prevLandmarks = landmarks
        return avg
    }

    private func score(for jitter: CGFloat) -> CGFloat {
        // 抖动越小分越高;满分 1.0
        max(0, 1 - jitter / stableThreshold)
    }

    /// 手动触发(用户点快门),跳过稳定检测
    func manualTrigger(timestamp: TimeInterval) {
        guard timestamp - lastTriggerTime >= cooldown else { return }
        lastTriggerTime = timestamp
        onStableFrame?(Sample(timestamp: timestamp, poseJitter: 0, score: 1.0))
    }

    func reset() {
        history.removeAll()
        prevLandmarks.removeAll()
    }
}
