# WearTryOn — iOS 实时视频试穿(第一版)

纯端侧 iOS 试穿 App:MediaPipe 实时预览(30fps)+ 端侧 Mobile-VTON 定格高清。

## 架构

```
相机(30fps) → MediaPipe 分割/姿态 → 模板贴合合成(实时预览)
                                    ↓ 关键帧触发(姿态稳定/快门)
                              端侧 VTON(CoreML, 1-3s)
                                    ↓
                              高清试穿图(保存/分享)
```

## 目录

```
WearTryOn/
  App/            App 入口
  Camera/         相机捕获(AVFoundation)
  Perception/     MediaPipe 分割+姿态
  Rendering/      实时合成(CPU+Accelerate,后续 Metal)
  Models/         版型模板/布料预设/服装选择
  Core/           协调器、关键帧选择器
  VTON/           Mobile-VTON CoreML 引擎 + CLIP tokenizer
  UI/             SwiftUI 界面
  Resources/Models/   tflite + mlmodelc(CI 生成)
scripts/          权重下载、模型转换、MediaPipe 模型下载
.github/workflows/  CI(macOS runner)
```

## 构建(CI)

推送到 GitHub 后自动:
1. 下载 Mobile-VTON 权重并转换为 CoreML(convert-models job)
2. 下载 MediaPipe 模型、生成工程、构建模拟器包(build-ios job)

产出:`dist/WearTryOn-simulator.app.zip`(模拟器运行)+ `coreml-models` 工件。

## 本地构建(Mac)

```bash
# 1. 下载 MediaPipe 模型
python3 scripts/download_mediapipe_models.py --out WearTryOn/Resources/Models

# 2. 转换 CoreML(需要 macOS)
python3 scripts/convert_mobile_vton.py \
  --checkpoint Models/mobile-vton/checkpoint \
  --repo Models/Mobile-VTON-repo \
  --out build/coreml
cp -R build/coreml/*.mlmodelc WearTryOn/Resources/Models/

# 3. 生成工程并构建
xcodegen generate
pod install
open WearTryOn.xcworkspace   # Xcode 中运行
```

## 版本规划

- v1.0:实时预览 + 端侧定格高清 + 5 类版型 + 6 种布料预设
- v1.1:Metal 渲染优化、姿态稳定度参数调优、更多版型
- v2.0:3D 布料模拟(XPBD)交互层
