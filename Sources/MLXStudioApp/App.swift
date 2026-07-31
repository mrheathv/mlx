import SwiftUI

@main
struct MLXStudioApp: App {
    @StateObject private var appEnvironment = AppEnvironment()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appEnvironment.modelManager)
                .environmentObject(appEnvironment.generationViewModel)
                .environmentObject(appEnvironment.galleryViewModel)
                .frame(minWidth: 900, minHeight: 600)
                .onAppear { appEnvironment.start() }
        }
        .windowResizability(.contentSize)
    }
}

/// Composition root: owns the long-lived services (backend process,
/// history store, model manager) and the view models that wrap them for
/// SwiftUI. Kept as one small object so `App.swift` doesn't have to know
/// construction order/dependencies.
@MainActor
final class AppEnvironment: ObservableObject {
    let client: InferenceBackendClient
    let historyStore: HistoryStore
    let modelManager: ModelManager
    let generationViewModel: GenerationViewModel
    let galleryViewModel: GalleryViewModel

    init() {
        let client = InferenceBackendClient()
        guard let historyStore = try? HistoryStore() else {
            fatalError("Could not create ~/Library/Application Support/MLXStudio — the app cannot persist generations without it.")
        }
        let modelsCacheDirectory = historyStore.rootDirectory.appendingPathComponent("models", isDirectory: true)
        let modelManager = ModelManager(client: client, modelsCacheDirectory: modelsCacheDirectory)

        self.client = client
        self.historyStore = historyStore
        self.modelManager = modelManager
        self.generationViewModel = GenerationViewModel(modelManager: modelManager, client: client, historyStore: historyStore)
        self.galleryViewModel = GalleryViewModel(historyStore: historyStore)
    }

    func start() {
        do {
            try modelManager.loadCatalog()
            generationViewModel.onModelSelected()
        } catch {
            // Bundled models.json failing to load means the app bundle
            // itself is broken (not a runtime/network condition), so this
            // surfaces loudly during development rather than being
            // swallowed into an empty, confusing model picker.
            assertionFailure("Failed to load bundled model catalog: \(error)")
        }
    }
}
