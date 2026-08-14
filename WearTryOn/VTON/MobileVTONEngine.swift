import Foundation
import CoreML
import CoreGraphics
import UIKit
import Accelerate

/// VTON 增强引擎协议:输入人物帧 + 服装图,输出高清试穿图。
/// 第一版实现为 Mobile-VTON 的 CoreML 管线(由 macOS CI 转换产出模型),
/// 若模型缺失则优雅降级为"实时预览帧"(保证 App 可用)。
protocol VTONEngineProtocol {
    /// 是否就绪(模型已加载)
    var isReady: Bool { get }
    /// 同步执行试穿(阻塞调用方线程;UI 层应放入后台队列)
    func tryOn(personImage: UIImage, garmentImage: UIImage, prompt: String) throws -> UIImage
}

/// Mobile-VTON CoreML 引擎。
///
/// Pipeline 组件(全部由 scripts/convert_mobile_vton.py 转换产出):
/// - text_encoder.mlmodelc / text_encoder_2.mlmodelc (CLIP)
/// - image_encoder.mlmodelc (DINOv2 服装编码)
/// - denoiser.mlmodelc (主去噪 UNet)
/// - denoiser_garment.mlmodelc (服装去噪 UNet)
/// - vae.mlmodelc (编码器)
/// - vae_decoder.mlmodelc (解码器)
///
/// 调度器(FlowMatchEulerDiscreteScheduler)在 Swift 端实现。
final class MobileVTONEngine: VTONEngineProtocol {

    enum EngineError: Swift.Error, LocalizedError {
        case modelNotFound(String)
        case modelLoadFailed(String)
        case inferenceFailed(String)
        case pipelineIncomplete(String)

        var errorDescription: String? {
            switch self {
            case .modelNotFound(let name): return "模型未找到: \(name)"
            case .modelLoadFailed(let name): return "模型加载失败: \(name)"
            case .inferenceFailed(let msg): return "推理失败: \(msg)"
            case .pipelineIncomplete(let msg): return "管线不完整: \(msg)"
            }
        }
    }

    private(set) var isReady = false

    // CoreML 模型(全部运行在 ANE/GPU,CPU 兜底)
    private var textEncoder: MLModel?
    private var textEncoder2: MLModel?
    private var imageEncoder: MLModel?
    private var denoiser: MLModel?
    private var denoiserGarment: MLModel?
    private var vae: MLModel?
    private var vaeDecoder: MLModel?

    // 配置
    private let resolution: Int = 512          // 生成分辨率(高端机可尝试 768)
    private let numSteps: Int = 8              // FlowMatch 采样步数(蒸馏后)
    private let guidanceScale: Float = 2.0     // CFG 引导系数
    private let schedulerShift: Float = 3.0    // FlowMatch shift
    private let latentChannels: Int = 16       // SD3.5 VAE 潜在通道
    private let latentScale: Float = 16.0      // VAE 压缩比(8x8 空间)

    // MARK: - 加载

    func load() throws {
        let bundle = Bundle.main
        func load(_ name: String) throws -> MLModel {
            guard let url = bundle.url(forResource: name, withExtension: "mlmodelc")
                    ?? bundle.url(forResource: name, withExtension: "mlmodel") else {
                throw EngineError.modelNotFound(name)
            }
            do {
                return try MLModel(contentsOf: url)
            } catch {
                throw EngineError.modelLoadFailed(name)
            }
        }

        textEncoder = try load("text_encoder")
        textEncoder2 = try load("text_encoder_2")
        imageEncoder = try load("image_encoder")
        denoiser = try load("denoiser")
        denoiserGarment = try load("denoiser_garment")
        vae = try load("vae")
        vaeDecoder = try load("vae_decoder")

        isReady = true
    }

    // MARK: - VTONEngineProtocol

    func tryOn(personImage: UIImage, garmentImage: UIImage, prompt: String) throws -> UIImage {
        guard isReady else { throw EngineError.pipelineIncomplete("模型未加载") }

        // 1. 预处理:缩放/归一化到 [-1, 1]
        let personTensor = try preprocess(personImage)
        let garmentTensor = try preprocess(garmentImage)

        // 2. 文本编码(双 CLIP)
        let promptEmbeds = try encodeText(prompt)

        // 3. 服装编码(DINOv2)
        let garmentEmbeds = try encodeGarment(garmentTensor)

        // 4. VAE 编码人物帧
        let latents = try encodeVAE(personTensor)

        // 5. FlowMatch 去噪循环
        let denoised = try denoiseLoop(
            latents: latents,
            promptEmbeds: promptEmbeds,
            garmentEmbeds: garmentEmbeds
        )

        // 6. VAE 解码 → 图像
        let output = try decodeVAE(denoised)
        return output
    }

    // MARK: - Pipeline 步骤

    private func preprocess(_ image: UIImage) throws -> MLMultiArray {
        guard let cg = image.cgImage?.cropping(to: centeredSquare(of: image.size)) ?? image.cgImage else {
            throw EngineError.inferenceFailed("无法读取图像")
        }
        let size = CGSize(width: resolution, height: resolution)
        guard let resized = resize(cg, to: size) else {
            throw EngineError.inferenceFailed("图像缩放失败")
        }
        return try rgbToTensor(resized)  // [1,3,H,W] float32 [-1,1]
    }

    private func centeredSquare(of size: CGSize) -> CGRect {
        let side = min(size.width, size.height)
        return CGRect(x: (size.width - side) / 2, y: (size.height - side) / 2,
                      width: side, height: side)
    }

    private func resize(_ image: CGImage, to size: CGSize) -> CGImage? {
        guard let ctx = CGContext(data: nil, width: Int(size.width), height: Int(size.height),
                                  bitsPerComponent: 8, bytesPerRow: Int(size.width) * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return nil
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(origin: .zero, size: size))
        return ctx.makeImage()
    }

    private func rgbToTensor(_ image: CGImage) throws -> MLMultiArray {
        let w = image.width, h = image.height
        guard let data = CFDataGetBytePtr(image.dataProvider?.data) else {
            throw EngineError.inferenceFailed("图像数据读取失败")
        }
        // 注意:CGImage 为 BGRA/RGBA 顺序不定,这里假设 RGBA;生产代码需按 bitmapInfo 处理
        let shape = [1, 3, NSNumber(value: h), NSNumber(value: w)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(arr.dataPointer))
        let bytesPerRow = image.bytesPerRow
        for y in 0..<h {
            for x in 0..<w {
                let offset = y * bytesPerRow + x * 4
                let r = Float(data[offset]) / 255.0 * 2.0 - 1.0
                let g = Float(data[offset + 1]) / 255.0 * 2.0 - 1.0
                let b = Float(data[offset + 2]) / 255.0 * 2.0 - 1.0
                ptr[y * w + x] = r
                ptr[w * h + y * w + x] = g
                ptr[2 * w * h + y * w + x] = b
            }
        }
        return arr
    }

    private func encodeText(_ prompt: String) throws -> [String: MLMultiArray] {
        // 简化:双 CLIP 编码。生产中 tokenizer 在 Swift 端实现(或使用 MLTextEncoder 包装)。
        // 第一版:CLIP tokenizer 由转换脚本固化词汇表,Swift 端实现分词。
        let tokens = CLIPTokenizerSwift.encode(prompt, maxLength: 77)
        guard let te = textEncoder, let te2 = textEncoder2 else {
            throw EngineError.pipelineIncomplete("text encoder 未加载")
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLMultiArray(tokens)  // [1,77] int32
        ])
        let out1 = try te.prediction(from: input)
        let out2 = try te2.prediction(from: input)
        guard let e1 = out1.featureValue(for: "text_embeds")?.multiArrayValue,
              let e2 = out2.featureValue(for: "text_embeds")?.multiArrayValue else {
            throw EngineError.inferenceFailed("文本编码输出缺失")
        }
        return ["text_encoder": e1, "text_encoder_2": e2]
    }

    private func encodeGarment(_ garmentTensor: MLMultiArray) throws -> MLMultiArray {
        guard let ie = imageEncoder else { throw EngineError.pipelineIncomplete("image encoder 未加载") }
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": garmentTensor])
        let out = try ie.prediction(from: input)
        guard let embeds = out.featureValue(for: "image_embeds")?.multiArrayValue ?? 
            out.featureValue(for: "embeds")?.multiArrayValue else {
            throw EngineError.inferenceFailed("服装编码输出缺失")
        }
        return embeds
    }

    private func encodeVAE(_ tensor: MLMultiArray) throws -> MLMultiArray {
        guard let vae else { throw EngineError.pipelineIncomplete("vae 未加载") }
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": tensor])
        let out = try vae.prediction(from: input)
        guard let latent = out.featureValue(for: "latent")?.multiArrayValue
            ?? out.featureValue(for: "posterior")?.multiArrayValue else {
            throw EngineError.inferenceFailed("VAE 编码输出缺失")
        }
        return latent
    }

    private func denoiseLoop(latents: MLMultiArray,
                             promptEmbeds: [String: MLMultiArray],
                             garmentEmbeds: MLMultiArray) throws -> MLMultiArray {
        guard let denoiser, let denoiserGarment else {
            throw EngineError.pipelineIncomplete("denoiser 未加载")
        }

        let scheduler = FlowMatchEulerScheduler(
            numSteps: numSteps,
            shift: schedulerShift,
            sigmaMin: 0.06,
            sigmaMax: 5.0
        )
        let sigmas = scheduler.timesteps

        // 噪声:第一版用固定种子,保证可复现
        var current = try addNoise(to: latents, sigma: sigmas[0])

        for step in 0..<numSteps {
            let sigma = sigmas[step]

            // CFG:有条件 + 无条件(空文本)两次前向
            let cond = try denoiseStep(
                latents: current, sigma: sigma,
                promptEmbeds: promptEmbeds, garmentEmbeds: garmentEmbeds,
                denoiser: denoiser, denoiserGarment: denoiserGarment,
                isConditional: true
            )
            let uncond = try denoiseStep(
                latents: current, sigma: sigma,
                promptEmbeds: [:], garmentEmbeds: garmentEmbeds,
                denoiser: denoiser, denoiserGarment: denoiserGarment,
                isConditional: false
            )

            // classifier-free guidance 组合
            let guided = try combineCFG(cond: cond, uncond: uncond, scale: guidanceScale)

            // Euler 步进
            let nextSigma = step + 1 < numSteps ? sigmas[step + 1] : 0.0
            current = try eulerStep(latents: current, velocity: guided,
                                    sigma: sigma, nextSigma: nextSigma)
        }
        return current
    }

    private func denoiseStep(latents: MLMultiArray,
                             sigma: Float,
                             promptEmbeds: [String: MLMultiArray],
                             garmentEmbeds: MLMultiArray,
                             denoiser: MLModel,
                             denoiserGarment: MLModel,
                             isConditional: Bool) throws -> MLMultiArray {
        // 1. 服装去噪网络(garment denoiser):提取服装特征
        let garmentInput = try MLDictionaryFeatureProvider(dictionary: [
            "latent": garmentEmbeds,
            "sigma": MLMultiArray(shape: [1], dataType: .float32, initialValue: isConditional ? sigma : 0.0),
            "text_embeds": promptEmbeds["text_encoder"] ?? emptyTextEmbeds(),
        ])
        let garmentOut = try denoiserGarment.prediction(from: garmentInput)
        guard let garmentFeat = garmentOut.featureValue(for: "output")?.multiArrayValue else {
            throw EngineError.inferenceFailed("garment denoiser 输出缺失")
        }

        // 2. 主去噪网络
        let mainInput = try MLDictionaryFeatureProvider(dictionary: [
            "latent": latents,
            "garment_feature": garmentFeat,
            "sigma": MLMultiArray(shape: [1], dataType: .float32, initialValue: sigma),
            "text_embeds": promptEmbeds["text_encoder"] ?? emptyTextEmbeds(),
            "text_embeds_2": promptEmbeds["text_encoder_2"] ?? emptyTextEmbeds(),
        ])
        let mainOut = try denoiser.prediction(from: mainInput)
        guard let velocity = mainOut.featureValue(for: "velocity")?.multiArrayValue
            ?? mainOut.featureValue(for: "output")?.multiArrayValue else {
            throw EngineError.inferenceFailed("denoiser 输出缺失")
        }
        return velocity
    }

    private func emptyTextEmbeds() -> MLMultiArray {
        // 空提示词嵌入(无条件分支);生产环境用真实 tokenizer 编码空串
        let shape = [1, NSNumber(value: 77), NSNumber(value: 2048)]
        return try! MLMultiArray(shape: shape, dataType: .float32, initialValue: 0.0)
    }

    private func combineCFG(cond: MLMultiArray, uncond: MLMultiArray, scale: Float) throws -> MLMultiArray {
        let n = cond.count
        let result = try MLMultiArray(shape: cond.shape, dataType: .float32)
        let cp = UnsafeMutablePointer<Float>(OpaquePointer(cond.dataPointer))
        let up = UnsafeMutablePointer<Float>(OpaquePointer(uncond.dataPointer))
        let rp = UnsafeMutablePointer<Float>(OpaquePointer(result.dataPointer))
        for i in 0..<n {
            rp[i] = up[i] + scale * (cp[i] - up[i])
        }
        return result
    }

    private func eulerStep(latents: MLMultiArray, velocity: MLMultiArray,
                           sigma: Float, nextSigma: Float) throws -> MLMultiArray {
        let dt = sigma - nextSigma
        let n = latents.count
        let result = try MLMultiArray(shape: latents.shape, dataType: .float32)
        let lp = UnsafeMutablePointer<Float>(OpaquePointer(latents.dataPointer))
        let vp = UnsafeMutablePointer<Float>(OpaquePointer(velocity.dataPointer))
        let rp = UnsafeMutablePointer<Float>(OpaquePointer(result.dataPointer))
        for i in 0..<n {
            rp[i] = lp[i] + dt * vp[i]
        }
        return result
    }

    private func addNoise(to latents: MLMultiArray, sigma: Float) throws -> MLMultiArray {
        let n = latents.count
        let result = try MLMultiArray(shape: latents.shape, dataType: .float32)
        let lp = UnsafeMutablePointer<Float>(OpaquePointer(latents.dataPointer))
        let rp = UnsafeMutablePointer<Float>(OpaquePointer(result.dataPointer))
        var seed: UInt64 = 42
        for i in 0..<n {
            let noise = Float(gaussianRandom(&seed))
            rp[i] = lp[i] + sigma * noise
        }
        return result
    }

    private func gaussianRandom(_ state: inout UInt64) -> Double {
        // Box-Muller(用 SplitMix64 种子)
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        let u1 = Double(z >> 11) / Double(1 << 53)
        state &+= 0x9E3779B97F4A7C15
        var z2 = state
        z2 = (z2 ^ (z2 >> 30)) &* 0xBF58476D1CE4E5B9
        z2 = (z2 ^ (z2 >> 27)) &* 0x94D049BB133111EB
        let u2 = Double(z2 >> 11) / Double(1 << 53)
        return sqrt(-2.0 * log(max(u1, 1e-12))) * cos(2.0 * Double.pi * u2)
    }

    private func decodeVAE(_ latents: MLMultiArray) throws -> UIImage {
        guard let vaeDecoder else { throw EngineError.pipelineIncomplete("vae_decoder 未加载") }
        let input = try MLDictionaryFeatureProvider(dictionary: ["latent": latents])
        let out = try vaeDecoder.prediction(from: input)
        guard let image = out.featureValue(for: "image")?.multiArrayValue
            ?? out.featureValue(for: "decoded")?.multiArrayValue else {
            throw EngineError.inferenceFailed("VAE 解码输出缺失")
        }
        return try tensorToUIImage(image)
    }

    private func tensorToUIImage(_ tensor: MLMultiArray) throws -> UIImage {
        // [1,3,H,W] [-1,1] → UIImage
        let shape = tensor.shape.map { $0.intValue }
        guard shape.count == 4, shape[0] == 1, shape[1] == 3 else {
            throw EngineError.inferenceFailed("解码张量形状异常: \(shape)")
        }
        let h = shape[2], w = shape[3]
        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(tensor.dataPointer))

        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let idx = y * w + x
                let r = clamp01(ptr[idx])
                let g = clamp01(ptr[w*h + idx])
                let b = clamp01(ptr[2*w*h + idx])
                let o = idx * 4
                pixels[o] = r
                pixels[o+1] = g
                pixels[o+2] = b
                pixels[o+3] = 255
            }
        }
        guard let ctx = CGContext(data: &pixels, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let cg = ctx.makeImage() else {
            throw EngineError.inferenceFailed("图像生成失败")
        }
        return UIImage(cgImage: cg)
    }

    private func clamp01(_ v: Float) -> UInt8 {
        UInt8(max(0, min(255, (v + 1) * 127.5)))
    }
}

/// FlowMatch Euler 调度器(Swift 实现,对应 diffusers FlowMatchEulerDiscreteScheduler)
struct FlowMatchEulerScheduler {
    let numSteps: Int
    let shift: Float
    let sigmaMin: Float
    let sigmaMax: Float

    /// 离散 sigma 时间表(与 diffusers 的 shift 逻辑对应)
    var timesteps: [Float] {
        var steps: [Float] = []
        for i in 0..<numSteps {
            let t = Float(i) / Float(numSteps - 1)
            steps.append(sigmaMin + (sigmaMax - sigmaMin) * pow(t, shift))
        }
        return steps.reversed()
    }
}
