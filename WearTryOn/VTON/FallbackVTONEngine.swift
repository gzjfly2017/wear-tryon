import Foundation
import UIKit

/// 降级引擎:当 Mobile-VTON CoreML 模型不可用时,
/// 直接返回实时预览帧(由 PreviewCompositor 合成),保证 App 功能可用。
final class FallbackVTONEngine: VTONEngineProtocol {
    var isReady: Bool { true }

    /// 最近的预览合成帧(由协调器注入)
    var latestPreview: UIImage?

    func tryOn(personImage: UIImage, garmentImage: UIImage, prompt: String) throws -> UIImage {
        if let latestPreview { return latestPreview }
        return personImage
    }
}
