import Foundation
import CoreGraphics

/// 版型类别:第一版覆盖最常见的 5 类
enum GarmentCategory: String, CaseIterable, Codable, Identifiable {
    case tShirt = "T恤"
    case shirt = "衬衫"
    case dress = "连衣裙"
    case hoodie = "卫衣"
    case jacket = "外套"

    var id: String { rawValue }

    /// 建议的物理参数索引(对应 FabricPresets 中的材质)
    var defaultFabric: FabricPreset { .cotton }
}

/// 布料物理预设(来自 Marvelous Designer 官方预设的参数化近似)
/// 每个预设定义模拟所需的相对物理参数:
/// - stiffness: 弯曲刚度(挺括度)
/// - stretch: 拉伸刚度
/// - shear: 剪切刚度
/// - friction: 表面摩擦
/// - density: 面密度(轻重)
struct FabricPreset: Codable, Equatable {
    let name: String
    let bendingStiffness: Float
    let stretchStiffness: Float
    let shearStiffness: Float
    let friction: Float
    let density: Float

    static let cotton = FabricPreset(name: "棉", bendingStiffness: 0.6, stretchStiffness: 0.8, shearStiffness: 0.5, friction: 0.6, density: 1.0)
    static let linen = FabricPreset(name: "麻", bendingStiffness: 0.9, stretchStiffness: 0.85, shearStiffness: 0.55, friction: 0.5, density: 0.9)
    static let silk = FabricPreset(name: "丝绸", bendingStiffness: 0.2, stretchStiffness: 0.5, shearStiffness: 0.3, friction: 0.35, density: 0.6)
    static let denim = FabricPreset(name: "牛仔", bendingStiffness: 1.5, stretchStiffness: 1.0, shearStiffness: 0.9, friction: 0.7, density: 1.4)
    static let knit = FabricPreset(name: "针织", bendingStiffness: 0.35, stretchStiffness: 0.4, shearStiffness: 0.35, friction: 0.55, density: 0.8)
    static let wool = FabricPreset(name: "羊毛", bendingStiffness: 0.7, stretchStiffness: 0.75, shearStiffness: 0.5, friction: 0.6, density: 1.2)

    static let all: [FabricPreset] = [.cotton, .linen, .silk, .denim, .knit, .wool]
}

/// 版型模板:定义服装在人体上的 2D 贴合区域(归一化坐标,基于 MediaPipe 33 关键点)。
/// 第一版用"关键点驱动的四边形区域"近似贴合,后续版本可替换为 3D 网格。
struct GarmentTemplate: Codable, Identifiable {
    let id: String
    let category: GarmentCategory
    let name: String

    /// 服装覆盖区域(归一化 0-1 矩形,相对人体包围盒)
    let coverage: CGRect
    /// 肩部宽度比例(相对躯干)
    let shoulderRatio: CGFloat
    /// 衣长比例(相对躯干高度)
    let lengthRatio: CGFloat

    static let templates: [GarmentTemplate] = [
        GarmentTemplate(id: "tshirt", category: .tShirt, name: "基础T恤",
                        coverage: CGRect(x: 0.12, y: 0.10, width: 0.76, height: 0.55),
                        shoulderRatio: 0.55, lengthRatio: 0.45),
        GarmentTemplate(id: "shirt", category: .shirt, name: "衬衫",
                        coverage: CGRect(x: 0.10, y: 0.08, width: 0.80, height: 0.62),
                        shoulderRatio: 0.56, lengthRatio: 0.52),
        GarmentTemplate(id: "dress", category: .dress, name: "连衣裙",
                        coverage: CGRect(x: 0.12, y: 0.06, width: 0.76, height: 0.85),
                        shoulderRatio: 0.55, lengthRatio: 0.75),
        GarmentTemplate(id: "hoodie", category: .hoodie, name: "卫衣",
                        coverage: CGRect(x: 0.10, y: 0.08, width: 0.80, height: 0.58),
                        shoulderRatio: 0.57, lengthRatio: 0.48),
        GarmentTemplate(id: "jacket", category: .jacket, name: "夹克",
                        coverage: CGRect(x: 0.10, y: 0.08, width: 0.80, height: 0.52),
                        shoulderRatio: 0.58, lengthRatio: 0.42),
    ]

    static func template(for category: GarmentCategory) -> GarmentTemplate {
        templates.first { $0.category == category } ?? templates[0]
    }
}

/// 用户选定的"服装 + 版型 + 材质"组合,作为实时预览与增强层的输入
struct TryOnSelection: Codable, Identifiable {
    let id: UUID
    /// 服装图片(裁剪后的前景图)
    var garmentImageData: Data
    var category: GarmentCategory
    var fabric: FabricPreset

    init(garmentImageData: Data, category: GarmentCategory, fabric: FabricPreset = .cotton) {
        self.id = UUID()
        self.garmentImageData = garmentImageData
        self.category = category
        self.fabric = fabric
    }
}
