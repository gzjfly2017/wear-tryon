import Foundation
import CoreML

/// MLMultiArray 张量工具(独立文件以降低 MobileVTONEngine 单文件复杂度)
enum MLTensorUtils {

    /// 从 [Int32] 创建 MLMultiArray(形状 [1, n])
    static func intArrayToMLMultiArray(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let arr = try MLMultiArray(shape: nsShape, dataType: .int32)
        for (i, v) in values.enumerated() {
            arr[i] = NSNumber(value: v)
        }
        return arr
    }

    /// 沿最后一维拼接两个数组(axis == 2,形状 [1, rows, d])
    static func concat(_ arrays: [MLMultiArray], axis: Int) throws -> MLMultiArray {
        guard axis == 2, arrays.count == 2,
              arrays[0].shape[0] == arrays[1].shape[0],
              arrays[0].shape[1] == arrays[1].shape[1] else {
            throw MLTensorError.invalidArgument("concat 参数不支持")
        }
        let a = arrays[0], b = arrays[1]
        let da = a.shape[2].intValue, db = b.shape[2].intValue
        let rows = a.shape[1].intValue, batch = a.shape[0].intValue
        let result = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: rows), NSNumber(value: da + db)],
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

    /// 最后一维补零到 width
    static func padWidth(_ array: MLMultiArray, width: Int) throws -> MLMultiArray {
        let d = array.shape[2].intValue
        guard d <= width else { throw MLTensorError.invalidArgument("pad 宽度不足") }
        let rows = array.shape[1].intValue, batch = array.shape[0].intValue
        let result = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: rows), NSNumber(value: width)],
                                      dataType: .float32)
        let ap = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let rp = UnsafeMutablePointer<Float>(OpaquePointer(result.dataPointer))
        for r in 0..<rows {
            for c in 0..<d { rp[r * width + c] = ap[r * d + c] }
        }
        return result
    }

    /// 行维度追加 rows 个零行
    static func appendZeroRows(_ array: MLMultiArray, rows: Int) throws -> MLMultiArray {
        let batch = array.shape[0].intValue
        let curRows = array.shape[1].intValue
        let width = array.shape[2].intValue
        let result = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: curRows + rows), NSNumber(value: width)],
                                      dataType: .float32)
        let ap = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let rp = UnsafeMutablePointer<Float>(OpaquePointer(result.dataPointer))
        let total = curRows * width
        for i in 0..<total { rp[i] = ap[i] }
        return result
    }

    /// 创建空文本嵌入(全零 [1, rows, dim])
    static func zeroEmbedding(rows: Int, dim: Int) -> MLMultiArray {
        let shape: [NSNumber] = [NSNumber(value: 1), NSNumber(value: rows), NSNumber(value: dim)]
        return try! MLMultiArray(shape: shape, dataType: .float32)
    }
}

enum MLTensorError: Error {
    case invalidArgument(String)
}
