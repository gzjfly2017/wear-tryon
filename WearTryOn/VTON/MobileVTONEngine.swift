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
/// - text_encoder.mlmodelc / text_encoder_2.mlmodelc (CLIP,输出 hidden_states[-2])
/// - image_encoder.mlmodelc (DINOv2 服装编码)
/// - denoiser.mlmodelc (合并去噪:garment UNet + try-on UNet)
/// - vae.mlmodelc (编码器) / vae_decoder.mlmodelc (解码器)
///
/// Swift 端负责:CLIP tokenizer、双编码器拼接、FlowMatch Euler 调度器、CFG、噪声。
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

    // CoreML 模型
    private var textEncoder: MLModel?
    private var textEncoder2: MLModel?
    private var imageEncoder: MLModel?
    private var denoiser: MLModel?
    private var vae: MLModel?
    private var vaeDecoder: MLModel?

    // 配置(与 pipeline / 转换脚本一致)
    private let resolution: Int = 512
    private let numSteps: Int = 8
    private let guidanceScale: Float = 2.0
    private let schedulerShift: Float = 3.0
    private let latentChannels: Int = 16
    private let textDim: Int = 4096          // 双 CLIP 拼接 + pad 后的特征维度
    private let t5Rows: Int = 256            // 零填充的 t5 行数(333 = 77 + 256)
    private let clipMaxLength = 77

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
        vae = try load("vae")
        vaeDecoder = try load("vae_decoder")

        isReady = true
    }

    // MARK: - VTONEngineProtocol

    func tryOn(personImage: UIImage, garmentImage: UIImage, prompt: String) throws -> UIImage {
        guard isReady else { throw EngineError.pipelineIncomplete("模型未加载") }

        // 1. 预处理(人物/服装图 -> 0-1 张量)
        let personTensor = try preprocess(personImage)
        let garmentTensor = try preprocess(garmentImage)

        // 2. 文本编码(双 CLIP -> [1,333,4096])
        let promptEmbeds = try encodePrompt(prompt)
        // 服装描述用同一文本(第一版简化;与 inference.py 使用不同描述对齐后可优化)
        let clothPromptEmbeds = promptEmbeds

        // 3. 服装编码(DINOv2)
        let garmentEmbeds = try encodeGarment(garmentTensor)

        // 4. VAE 编码(人物/服装 -> 潜在)
        let personLatent = try encodeVAE(personTensor)
        let clothLatent = try encodeVAE(garmentTensor)

        // 5. FlowMatch 去噪
        let denoised = try denoiseLoop(
            personLatent: personLatent,
            clothLatent: clothLatent,
            promptEmbeds: promptEmbeds,
            clothPromptEmbeds: clothPromptEmbeds,
            garmentEmbeds: garmentEmbeds
        )

        // 6. VAE 解码
        return try decodeVAE(denoised)
    }

    // MARK: - 预处理

    private func preprocess(_ image: UIImage) throws -> MLMultiArray {
        guard let cg = image.cgImage?.cropping(to: centeredSquare(of: image.size)) ?? image.cgImage else {
            throw EngineError.inferenceFailed("无法读取图像")
        }
        let size = CGSize(width: resolution, height: resolution)
        guard let resized = resize(cg, to: size) else {
            throw EngineError.inferenceFailed("图像缩放失败")
        }
        return try rgbToTensor01(resized)  // [1,3,H,W] float32 [0,1]
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

    private func rgbToTensor01(_ image: CGImage) throws -> MLMultiArray {
        let w = image.width, h = image.height
        guard let data = CFDataGetBytePtr(image.dataProvider?.data) else {
            throw EngineError.inferenceFailed("图像数据读取失败")
        }
        let shape = [1, 3, NSNumber(value: h), NSNumber(value: w)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(arr.dataPointer))
        let bytesPerRow = image.bytesPerRow
        for y in 0..<h {
            for x in 0..<w {
                let offset = y * bytesPerRow + x * 4
                ptr[y * w + x] = Float(data[offset]) / 255.0
                ptr[w * h + y * w + x] = Float(data[offset + 1]) / 255.0
                ptr[2 * w * h + y * w + x] = Float(data[offset + 2]) / 255.0
            }
        }
        return arr
    }

    // MARK: - 文本编码(双 CLIP 拼接,对齐 pipeline)

    private func encodePrompt(_ prompt: String) throws -> MLMultiArray {
        let tokens = CLIPTokenizerSwift.encode(prompt, maxLength: clipMaxLength)
        let ids = try intArrayToMLMultiArray(tokens, shape: [1, clipMaxLength])  // [1,77]

        guard let te1 = textEncoder, let te2 = textEncoder2 else {
            throw EngineError.pipelineIncomplete("text encoder 未加载")
        }
        let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": ids])
        let out1 = try te1.prediction(from: input)
        let out2 = try te2.prediction(from: input)
        guard let e1 = out1.featureValue(for: "hidden_states")?.multiArrayValue,
              let e2 = out2.featureValue(for: "hidden_states")?.multiArrayValue else {
            throw EngineError.inferenceFailed("文本编码输出缺失")
        }

        // clip_embeds = cat([e1, e2], dim=-1)  [1,77,1536]
        let clipEmbeds = try concat([e1, e2], axis: 2)
        // pad 到 4096
        let padded = try padWidth(clipEmbeds, width: textDim)
        // 追加 256 个零行 -> [1,333,4096]
        return try appendZeroRows(padded, rows: t5Rows)
    }

    // MARK: - 服装编码(DINOv2)

    private func encodeGarment(_ garmentTensor: MLMultiArray) throws -> MLMultiArray {
        guard let ie = imageEncoder else { throw EngineError.pipelineIncomplete("image encoder 未加载") }
        // DINOv2 输入:518x518 + ImageNet 归一化(转换脚本接受 0-1 输入;归一化若在模型内则直接传入)
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": garmentTensor])
        let out = try ie.prediction(from: input)
        guard let embeds = out.featureValue(for: "image_embeds")?.multiArrayValue else {
            throw EngineError.inferenceFailed("服装编码输出缺失")
        }
        return embeds
    }

    // MARK: - VAE

    private func encodeVAE(_ tensor: MLMultiArray) throws -> MLMultiArray {
        guard let vae else { throw EngineError.pipelineIncomplete("vae 未加载") }
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": tensor])
        let out = try vae.prediction(from: input)
        guard let latent = out.featureValue(for: "latent")?.multiArrayValue else {
            throw EngineError.inferenceFailed("VAE 编码输出缺失")
        }
        return latent
    }

    private func decodeVAE(_ latents: MLMultiArray) throws -> UIImage {
        guard let vaeDecoder else { throw EngineError.pipelineIncomplete("vae_decoder 未加载") }
        let input = try MLDictionaryFeatureProvider(dictionary: ["latent": latents])
        let out = try vaeDecoder.prediction(from: input)
        guard let image = out.featureValue(for: "image")?.multiArrayValue else {
            throw EngineError.inferenceFailed("VAE 解码输出缺失")
        }
        return try tensorToUIImage(image)
    }

    // MARK: - 去噪循环(FlowMatch Euler + CFG)

    private func denoiseLoop(personLatent: MLMultiArray,
                             clothLatent: MLMultiArray,
                             promptEmbeds: MLMultiArray,
                             clothPromptEmbeds: MLMultiArray,
                             garmentEmbeds: MLMultiArray) throws -> MLMultiArray {
        guard let denoiser else { throw EngineError.pipelineIncomplete("denoiser 未加载") }

        let scheduler = FlowMatchEulerScheduler(
            numSteps: numSteps, shift: schedulerShift, sigmaMin: 0.06, sigmaMax: 5.0
        )
        let sigmas = scheduler.timesteps

        var current = try addNoise(to: personLatent, sigma: sigmas[0])

        for step in 0..<numSteps {
            let sigma = sigmas[step]

            let cond = try denoiseStep(
                personLatent: current, clothLatent: clothLatent,
                sigma: sigma, promptEmbeds: promptEmbeds,
                clothPromptEmbeds: clothPromptEmbeds, garmentEmbeds: garmentEmbeds,
                denoiser: denoiser
            )
            let uncond = try denoiseStep(
                personLatent: current, clothLatent: clothLatent,
                sigma: sigma, promptEmbeds: emptyPromptEmbeds(),
                clothPromptEmbeds: clothPromptEmbeds, garmentEmbeds: garmentEmbeds,
                denoiser: denoiser
            )
            let guided = try combineCFG(cond: cond, uncond: uncond, scale: guidanceScale)
            let nextSigma = step + 1 < numSteps ? sigmas[step + 1] : 0.0
            current = try eulerStep(latents: current, velocity: guided,
                                    sigma: sigma, nextSigma: nextSigma)
        }
        return current
    }

    private func denoiseStep(personLatent: MLMultiArray,
                             clothLatent: MLMultiArray,
                             sigma: Float,
                             promptEmbeds: MLMultiArray,
                             clothPromptEmbeds: MLMultiArray,
                             garmentEmbeds: MLMultiArray,
                             denoiser: MLModel) throws -> MLMultiArray {
        let sigmaArr = try MLMultiArray(shape: [1], dataType: .float32)
        sigmaArr[0] = NSNumber(value: sigma)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "person_latent": personLatent,
            "cloth_latent": clothLatent,
            "sigma": sigmaArr,
            "text_embeds": promptEmbeds,
            "cloth_text_embeds": clothPromptEmbeds,
            "image_embeds": garmentEmbeds,
        ])
        let out = try denoiser.prediction(from: input)
        guard let velocity = out.featureValue(for: "velocity")?.multiArrayValue else {
            throw EngineError.inferenceFailed("denoiser 输出缺失")
        }
        return velocity
    }

    private func emptyPromptEmbeds() -> MLMultiArray {
        let shape = [1, NSNumber(value: 77 + t5Rows), NSNumber(value: textDim)]
        return try! MLMultiArray(shape: shape, dataType: .float32)
    }

    // MARK: - 张量工具

    /// 从 [Int32] 创建 MLMultiArray(形状 [1, n])
    private func intArrayToMLMultiArray(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let arr = try MLMultiArray(shape: nsShape, dataType: .int32)
        for (i, v) in values.enumerated() {
            arr[i] = NSNumber(value: v)
        }
        return arr
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

    private func concat(_ arrays: [MLMultiArray], axis: Int) throws -> MLMultiArray {
        // 仅支持最后一维拼接(axis == 2 for [1,77,d])
        guard axis == 2, arrays.count == 2,
              arrays[0].shape[0] == arrays[1].shape[0],
              arrays[0].shape[1] == arrays[1].shape[1] else {
            throw EngineError.inferenceFailed("concat 参数不支持")
        }
        let a = arrays[0], b = arrays[1]
        let da = a.shape[2].intValue, db = b.shape[2].intValue
        let rows = a.shape[1].intValue, batch = a.shape[0].intValue
        let result = try MLMultiArray(shape: [batch, NSNumber(value: rows), NSNumber(value: da + db)],
                                      dataType: .float32)
        let ap = UnsafeMutablePointer<Float>(OpaquePointer(a.dataPointer))
        let bp = UnsafeMutablePointer<Float>(OpaquePointer(b.dataPointer))
        let rp = UnsafeMutablePointer<Float>(OpaquePointer(result.dataPointer))
        for r in 0..<rows {
            for c in 0..<da { rp[r * (da + db) + c] = ap[r * da + c] }
            for c in 0..<db { rp[r * (da + db) + da + c] = bp[r * db + c] }
        }
        return result
    }

    private func padWidth(_ array: MLMultiArray, width: Int) throws -> MLMultiArray {
        let d = array.shape[2].intValue
        guard d <= width else { throw EngineError.inferenceFailed("pad 宽度不足") }
        let rows = array.shape[1].intValue, batch = array.shape[0].intValue
        let result = try MLMultiArray(shape: [batch, NSNumber(value: rows), NSNumber(value: width)],
                                      dataType: .float32)
        let ap = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let rp = UnsafeMutablePointer<Float>(OpaquePointer(result.dataPointer))
        for r in 0..<rows {
            for c in 0..<d { rp[r * width + c] = ap[r * d + c] }
        }
        return result
    }

    private func appendZeroRows(_ array: MLMultiArray, rows: Int) throws -> MLMultiArray {
        let batch = array.shape[0].intValue
        let curRows = array.shape[1].intValue
        let width = array.shape[2].intValue
        let result = try MLMultiArray(shape: [batch, NSNumber(value: curRows + rows), NSNumber(value: width)],
                                      dataType: .float32)
        let ap = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let rp = UnsafeMutablePointer<Float>(OpaquePointer(result.dataPointer))
        let total = curRows * width
        for i in 0..<total { rp[i] = ap[i] }
        return result
    }

    private func tensorToUIImage(_ tensor: MLMultiArray) throws -> UIImage {
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
                pixels[idx * 4] = clamp01(ptr[idx])
                pixels[idx * 4 + 1] = clamp01(ptr[w * h + idx])
                pixels[idx * 4 + 2] = clamp01(ptr[2 * w * h + idx])
                pixels[idx * 4 + 3] = 255
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
        UInt8(max(0, min(255, v * 255.0)))
    }
}

/// FlowMatch Euler 调度器(Swift 实现,对应 diffusers FlowMatchEulerDiscreteScheduler)
struct FlowMatchEulerScheduler {
    let numSteps: Int
    let shift: Float
    let sigmaMin: Float
    let sigmaMax: Float

    /// 离散 sigma 时间表
    var timesteps: [Float] {
        var steps: [Float] = []
        for i in 0..<numSteps {
            let t = Float(i) / Float(numSteps - 1)
            steps.append(sigmaMin + (sigmaMax - sigmaMin) * pow(t, shift))
        }
        return steps.reversed()
    }
}
