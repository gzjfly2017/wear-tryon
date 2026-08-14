import Foundation
import UIKit
import CoreVideo
import Combine

/// 试穿协调器:串联相机 → 感知 → 合成预览 → 关键帧触发 → 端侧增强。
/// 这是 App 的"大脑",UI 层只负责展示与用户交互。
@MainActor
final class TryOnCoordinator: ObservableObject {

    // MARK: - 发布状态(UI 绑定)

    @Published private(set) var previewImage: UIImage?
    @Published private(set) var enhancedImage: UIImage?
    @Published private(set) var isEnhancing = false
    @Published private(set) var isReady = false
    @Published private(set) var statusMessage = "初始化中…"
    @Published var selectedGarment: TryOnSelection?

    // MARK: - 组件

    private let camera = CameraLayer()
    private let perception = BodyPerceptionEngine()
    private let compositor = PreviewCompositor()
    private let frameSelector = FrameSelector()

    /// 增强引擎:优先 Mobile-VTON,失败则降级
    private let vtonEngine: VTONEngineProtocol
    private let fallback = FallbackVTONEngine()

    private var isPerceptionLoaded = false
    private var timestampCounter = 0
    private var lastPerceptionTime: CFTimeInterval = 0
    /// 感知帧率控制:MediaPipe 同步推理较重,限制在 15fps 左右
    private let perceptionInterval: CFTimeInterval = 1.0 / 15.0

    // MARK: - 初始化

    init() {
        // 尝试加载 VTON 引擎;若模型缺失,使用降级引擎
        let mobileEngine = MobileVTONEngine()
        do {
            try mobileEngine.load()
            self.vtonEngine = mobileEngine
        } catch {
            self.vtonEngine = fallback
            statusMessage = "增强模型不可用,已启用预览模式: \(error.localizedDescription)"
        }

        frameSelector.onStableFrame = { [weak self] sample in
            Task { @MainActor in
                self?.triggerEnhancement()
            }
        }
    }

    // MARK: - 生命周期

    func start() async {
        // 1. 相机权限
        let granted = await camera.requestAccess()
        guard granted else {
            statusMessage = "需要相机权限才能进行试穿"
            return
        }

        // 2. 加载感知模型(分割 + 姿态)
        do {
            try perception.load()
            isPerceptionLoaded = true
        } catch {
            statusMessage = "感知模型加载失败: \(error.localizedDescription)"
        }

        // 3. 启动相机并挂接帧回调
        camera.onFrame = { [weak self] pixelBuffer in
            self?.handleFrame(pixelBuffer)
        }
        camera.start()
        isReady = true
        statusMessage = "就绪"
    }

    func stop() {
        camera.stop()
    }

    // MARK: - 帧处理(相机回调线程)

    private func handleFrame(_ pixelBuffer: CVPixelBuffer) {
        timestampCounter += 1

        // 感知节流:同步双模型推理较重,按 interval 采样
        let now = CACurrentMediaTime()
        guard now - lastPerceptionTime >= perceptionInterval else { return }
        lastPerceptionTime = now

        // 感知:分割 + 姿态(线程安全,内部串行)
        guard isPerceptionLoaded,
              let perception = perception.process(pixelBuffer: pixelBuffer) else {
            return
        }

        // 帧选择器:检测稳定姿态(仅当有服装选择时)
        if selectedGarment != nil {
            let landmarks = perception.landmarks.map {
                CGPoint(x: CGFloat($0.x), y: CGFloat($0.y))
            }
            let timestamp = Double(timestampCounter) / 30.0
            frameSelector.update(landmarks: landmarks, timestamp: timestamp)
        }

        // 合成预览(有服装时叠加;无服装时显示原帧)
        guard let selection = selectedGarment,
              let garmentCG = UIImage(data: selection.garmentImageData)?.cgImage else {
            // 无服装:把原帧转为 UIImage 供预览(降采样)
            if let cg = compositor.pixelBufferToCGImage(pixelBuffer) {
                let ui = UIImage(cgImage: cg)
                DispatchQueue.main.async { [weak self] in
                    self?.previewImage = ui
                }
            }
            return
        }

        let input = PreviewCompositor.Input(
            cameraFrame: pixelBuffer,
            mask: perception.segmentationMask,
            landmarks: perception.landmarks.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) },
            garmentTexture: garmentCG,
            template: GarmentTemplate.template(for: selection.category),
            frameSize: CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                              height: CVPixelBufferGetHeight(pixelBuffer))
        )

        if let composed = compositor.composite(input) {
            let ui = UIImage(cgImage: composed)
            DispatchQueue.main.async { [weak self] in
                self?.previewImage = ui
                self?.fallback.latestPreview = ui
            }
        }
    }

    // MARK: - 增强触发

    func triggerEnhancement() {
        guard !isEnhancing, let selection = selectedGarment,
              let preview = previewImage else { return }

        isEnhancing = true
        statusMessage = "正在生成高清试穿效果…"

        let personImage = preview
        let garmentImage = UIImage(data: selection.garmentImageData) ?? preview
        let prompt = "Replace the upper body with \(selection.category.rawValue) garment"

        // 推理放到后台队列(避免阻塞 UI)
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do {
                let result = try self.vtonEngine.tryOn(
                    personImage: personImage,
                    garmentImage: garmentImage,
                    prompt: prompt
                )
                await MainActor.run {
                    self.enhancedImage = result
                    self.isEnhancing = false
                    self.statusMessage = "完成"
                }
            } catch {
                await MainActor.run {
                    self.isEnhancing = false
                    self.statusMessage = "增强失败: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - 用户操作

    func setGarmentImage(_ image: UIImage, category: GarmentCategory) {
        guard let data = image.jpegData(compressionQuality: 0.9) else { return }
        let selection = TryOnSelection(garmentImageData: data, category: category)
        selectedGarment = selection
        frameSelector.reset()
        statusMessage = "已选择: \(category.rawValue),对准身体并保持静止"
    }

    func clearGarment() {
        selectedGarment = nil
        enhancedImage = nil
        frameSelector.reset()
    }

    /// 手动快门
    func capture() {
        frameSelector.manualTrigger(timestamp: Date().timeIntervalSince1970)
    }
}
