# WearTryOn 第一版安装与发布指南

## 1. 产物说明

CI 每次推送到 `main` 后自动产出两个工件(Artifacts):

| 工件 | 内容 | 用途 |
|---|---|---|
| `wear-tryon-device-app` | `WearTryOn-device.app.zip` | 真机部署(需签名)/代码签名参考 |
| `coreml-models` | 各 `*.mlmodelc` + tokenizer | 端侧 VTON 模型(调试/二次打包用) |

> 注:CI 构建为无签名真机目标(`generic/platform=iOS`),不含模拟器 runtime。
> 本地模拟器运行见下方"本地构建"章节(需有 Mac 且安装对应模拟器运行时)。

## 2. 模拟器运行(本地 Mac 验证,无需开发者账号)

1. 在 Mac 上克隆仓库并本地构建(需要 Xcode + 模拟器运行时):
   ```bash
   git clone <repo-url> && cd wear-tryon
   brew install xcodegen
   xcodegen generate
   pod install
   open WearTryOn.xcworkspace   # 选模拟器,⌘R 运行
   ```
2. 注意:模拟器没有真实相机,摄像头预览会显示黑帧/测试图案——**核心逻辑(分割/姿态/模板贴合)可运行,但建议用真机体验完整效果**。

## 3. 真机运行(免费 Apple ID,7 天签名)

1. 在 Mac 上打开 `WearTryOn.xcworkspace`(需先本地生成工程)
2. Xcode → Signing & Capabilities → Team 选自己的 Apple ID(免费账号)
3. 修改 `PRODUCT_BUNDLE_IDENTIFIER` 为唯一值(如 `com.yourname.tryon`)
4. iPhone 连接 Mac → 选设备 → ⌘R 运行
5. 首次运行需在 iPhone 上信任开发者证书:设置 → 通用 → VPN与设备管理 → 信任

## 4. TestFlight / App Store 发布(后续步骤,需付费账号)

1. 注册 Apple Developer Program($99/年):developer.apple.com
2. 在 CI 工作流中添加签名配置(secrets: `APPLE_CERTIFICATE`、`APPLE_PROVISIONING_PROFILE`、`APPLE_ID`、`APPLE_APP_SPECIFIC_PASSWORD`)
3. 使用 fastlane match + pilot 自动上传 TestFlight
4. 说明:本仓库当前 CI 产出**无签名模拟器包**,上架流程需额外配置签名步骤(可后续添加)

## 5. 常见问题

| 问题 | 处理 |
|---|---|
| 增强按钮灰色 | 未选择服装,先点"选服装"导入图片 |
| 增强结果与预览差异大 | 属正常:预览是轻量贴合,增强是完整 VTON 模型 |
| 无增强模型(预览模式) | CI 转换失败时自动降级;检查 `coreml-models` 工件是否生成 |
| 模型下载慢 | 已配置 hf-mirror 国内镜像;可设 `HF_ENDPOINT` 环境变量 |
