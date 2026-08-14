import XCTest
@testable import WearTryOn

final class WearTryOnTests: XCTestCase {

    func testGarmentTemplatesCoverAllCategories() {
        // 每个版型类别都应有模板
        for category in GarmentCategory.allCases {
            let template = GarmentTemplate.template(for: category)
            XCTAssertEqual(template.category, category)
        }
    }

    func testTemplateGeometryValid() {
        // 覆盖率必须在 (0,1) 范围内
        for t in GarmentTemplate.templates {
            XCTAssertGreaterThan(t.coverage.width, 0)
            XCTAssertGreaterThan(t.coverage.height, 0)
            XCTAssertLessThanOrEqual(t.coverage.maxX, 1.0)
            XCTAssertLessThanOrEqual(t.coverage.maxY, 1.0)
        }
    }

    func testFabricPresetsHaveDistinctPhysics() {
        // 丝绸 vs 牛仔:弯曲刚度应有明显差异(丝绸软,牛仔硬)
        XCTAssertLessThan(FabricPreset.silk.bendingStiffness, FabricPreset.denim.bendingStiffness)
        // 密度:牛仔 > 丝绸
        XCTAssertGreaterThan(FabricPreset.denim.density, FabricPreset.silk.density)
    }

    func testFrameSelectorDetectsStablePose() {
        let selector = FrameSelector()
        var triggered = 0
        selector.onStableFrame = { _ in triggered += 1 }

        // 静止姿态:同一组关键点
        let landmarks = (0..<33).map { i -> CGPoint in
            CGPoint(x: 0.1 + 0.02 * CGFloat(i % 5), y: 0.1 + 0.02 * CGFloat(i / 5))
        }

        var t: TimeInterval = 0
        // 前几帧(冷启动,抖动为 1.0)
        for _ in 0..<4 {
            selector.update(landmarks: landmarks, timestamp: t)
            t += 1.0 / 30.0
        }
        XCTAssertEqual(triggered, 0, "冷启动不应触发")

        // 稳定帧(同一组点,抖动 ≈ 0)
        for _ in 0..<10 {
            selector.update(landmarks: landmarks, timestamp: t)
            t += 1.0 / 30.0
        }
        XCTAssertGreaterThanOrEqual(triggered, 1, "稳定姿态应触发增强")
    }

    func testFrameSelectorRespectsCooldown() {
        let selector = FrameSelector()
        var triggerTimes: [TimeInterval] = []
        selector.onStableFrame = { triggerTimes.append($0.timestamp) }

        let landmarks = (0..<33).map { i -> CGPoint in
            CGPoint(x: 0.1 + 0.02 * CGFloat(i % 5), y: 0.1 + 0.02 * CGFloat(i / 5))
        }

        var t: TimeInterval = 0
        for _ in 0..<15 { selector.update(landmarks: landmarks, timestamp: t); t += 1.0 / 30.0 }
        XCTAssertEqual(triggerTimes.count, 1, "首次稳定触发一次")

        // 继续稳定 3 秒内不应再触发(冷却)
        for _ in 0..<90 { selector.update(landmarks: landmarks, timestamp: t); t += 1.0 / 30.0 }
        XCTAssertEqual(triggerTimes.count, 1, "冷却期内不应重复触发")

        // 超过冷却后再次触发
        for _ in 0..<120 { selector.update(landmarks: landmarks, timestamp: t); t += 1.0 / 30.0 }
        XCTAssertGreaterThanOrEqual(triggerTimes.count, 2, "冷却后应再次触发")
    }

    func testCLIPTokenizerProducesFixedLength() {
        let tokens = CLIPTokenizerSwift.encode("a red dress", maxLength: 77)
        XCTAssertEqual(tokens.count, 77)
        // 首位是 start token
        XCTAssertEqual(tokens[0], 49406)
        // 含 end token
        XCTAssertTrue(tokens.contains(49407))
    }

    func testFlowMatchSchedulerMonotonic() {
        let scheduler = FlowMatchEulerScheduler(numSteps: 8, shift: 3.0, sigmaMin: 0.06, sigmaMax: 5.0)
        let steps = scheduler.timesteps
        XCTAssertEqual(steps.count, 8)
        // 从大到小(噪声先大后小)
        for i in 0..<(steps.count - 1) {
            XCTAssertGreaterThanOrEqual(steps[i], steps[i + 1])
        }
    }
}
