import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var galleryViewModel: GalleryViewModel

    enum Tab: String, CaseIterable, Identifiable {
        case generate = "Generate"
        case gallery = "Gallery"
        var id: String { rawValue }
    }

    @State private var selectedTab: Tab = .generate

    var body: some View {
        NavigationSplitView {
            List(Tab.allCases, selection: $selectedTab) { tab in
                Label(tab.rawValue, systemImage: tab == .generate ? "sparkles" : "photo.on.rectangle")
                    .tag(tab)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 180)
        } detail: {
            switch selectedTab {
            case .generate:
                GenerateScreen()
            case .gallery:
                GalleryView()
                    .onAppear { galleryViewModel.refresh() }
            }
        }
    }
}

private struct GenerateScreen: View {
    @EnvironmentObject private var generationViewModel: GenerationViewModel

    var body: some View {
        HSplitView {
            PromptFormView()
                .frame(minWidth: 360, idealWidth: 420)
            GenerationProgressView()
                .frame(minWidth: 360)
        }
        .sheet(item: memoryWarningBinding) { assessment in
            MemoryWarningView(
                assessment: assessment,
                onCancel: { generationViewModel.dismissMemoryWarning() },
                onProceed: { generationViewModel.proceedDespiteMemoryWarning() }
            )
        }
    }

    private var memoryWarningBinding: Binding<MemoryGuard.Assessment?> {
        Binding(
            get: { generationViewModel.pendingMemoryWarning },
            set: { if $0 == nil { generationViewModel.dismissMemoryWarning() } }
        )
    }
}

extension MemoryGuard.Assessment: Identifiable {
    var id: String { "\(estimatedRequiredGB)-\(estimatedAvailableGB)" }
}
