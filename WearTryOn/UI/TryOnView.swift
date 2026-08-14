import SwiftUI
import PhotosUI

/// 主试穿界面
struct TryOnView: View {
    @StateObject private var coordinator = TryOnCoordinator()
    @State private var showGarmentPicker = false
    @State private var pickerItem: PhotosPickerItem?
    @State private var selectedCategory: GarmentCategory = .tShirt
    @State private var showShareSheet = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // 实时预览(或增强结果)
            if let enhanced = coordinator.enhancedImage {
                Image(uiImage: enhanced)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
            } else if let preview = coordinator.previewImage {
                Image(uiImage: preview)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
            } else {
                ProgressView("初始化相机…")
                    .tint(.white)
                    .foregroundStyle(.white)
            }

            VStack {
                // 顶部状态
                statusBar

                Spacer()

                // 底部控制栏
                controlBar
            }
        }
        .task { await coordinator.start() }
        .onDisappear { coordinator.stop() }
        .photosPicker(isPresented: $showGarmentPicker,
                      selection: $pickerItem,
                      matching: .images)
        .onChange(of: pickerItem) { _, newItem in
            guard let newItem else { return }
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    coordinator.setGarmentImage(image, category: selectedCategory)
                }
            }
        }
        .sheet(isPresented: $showShareSheet) {
            if let image = coordinator.enhancedImage {
                ShareSheet(items: [image])
            }
        }
    }

    // MARK: - 子视图

    private var statusBar: some View {
        HStack {
            Text(coordinator.statusMessage)
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.9))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.black.opacity(0.5), in: Capsule())
            Spacer()
            if coordinator.isEnhancing {
                ProgressView()
                    .tint(.white)
                    .padding(.trailing, 12)
            }
        }
        .padding(.top, 8)
        .padding(.horizontal, 12)
    }

    private var controlBar: some View {
        VStack(spacing: 12) {
            // 版型选择
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(GarmentCategory.allCases) { category in
                        Button {
                            selectedCategory = category
                        } label: {
                            Text(category.rawValue)
                                .font(.subheadline)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 8)
                                .background(selectedCategory == category ? Color.blue : Color.white.opacity(0.2),
                                            in: Capsule())
                                .foregroundStyle(.white)
                        }
                    }
                }
                .padding(.horizontal, 12)
            }

            HStack(spacing: 24) {
                // 选择服装
                Button {
                    showGarmentPicker = true
                } label: {
                    VStack(spacing: 4) {
                        Image(systemName: "tshirt")
                            .font(.title2)
                        Text("选服装")
                            .font(.caption2)
                    }
                    .frame(width: 56, height: 56)
                    .background(.white.opacity(0.2), in: Circle())
                    .foregroundStyle(.white)
                }

                // 快门
                Button {
                    coordinator.capture()
                } label: {
                    Circle()
                        .stroke(.white, lineWidth: 4)
                        .frame(width: 72, height: 72)
                        .overlay {
                            Circle()
                                .fill(.white)
                                .frame(width: 58, height: 58)
                        }
                }
                .disabled(coordinator.selectedGarment == nil)
                .opacity(coordinator.selectedGarment == nil ? 0.4 : 1.0)

                // 分享/清除
                Menu {
                    if coordinator.enhancedImage != nil {
                        Button {
                            showShareSheet = true
                        } label: {
                            Label("分享结果", systemImage: "square.and.arrow.up")
                        }
                    }
                    Button(role: .destructive) {
                        coordinator.clearGarment()
                    } label: {
                        Label("清除服装", systemImage: "xmark.circle")
                    }
                } label: {
                    VStack(spacing: 4) {
                        Image(systemName: "ellipsis")
                            .font(.title2)
                        Text("更多")
                            .font(.caption2)
                    }
                    .frame(width: 56, height: 56)
                    .background(.white.opacity(0.2), in: Circle())
                    .foregroundStyle(.white)
                }
            }
            .padding(.bottom, 20)
        }
    }
}

/// 系统分享面板
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

#Preview {
    TryOnView()
}
