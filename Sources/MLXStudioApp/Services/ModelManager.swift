import Foundation

/// Tracks which models are downloaded, drives downloads through the
/// backend, and exposes the model catalog to the UI.
///
/// "Downloaded" is re-derived from the on-disk cache directory on launch
/// rather than trusted from a persisted flag — a flag can go stale (model
/// manually deleted, download interrupted mid-way) in a way a directory
/// listing can't.
@MainActor
final class ModelManager: ObservableObject {
    enum DownloadState: Equatable {
        case notDownloaded
        case downloading(fractionComplete: Double)
        case ready
        case failed(message: String)
    }

    @Published private(set) var catalog: [ModelDescriptor] = []
    @Published private(set) var downloadStates: [String: DownloadState] = [:]
    @Published var selectedModelId: String?

    private let client: InferenceBackendClient
    private let modelsCacheDirectory: URL

    init(client: InferenceBackendClient, modelsCacheDirectory: URL) {
        self.client = client
        self.modelsCacheDirectory = modelsCacheDirectory
    }

    var selectedModel: ModelDescriptor? {
        catalog.first { $0.id == selectedModelId }
    }

    func loadCatalog() throws {
        let loaded = try ModelCatalog.loadBundled().models
        catalog = loaded
        if selectedModelId == nil {
            selectedModelId = loaded.first(where: { $0.tier == .fast })?.id ?? loaded.first?.id
        }
        for model in loaded {
            downloadStates[model.id] = detectLocalState(for: model)
        }
    }

    /// The backend downloads into `<modelsCacheDirectory>/<model id>/` and
    /// writes this marker only once the download has fully completed
    /// (`DOWNLOAD_COMPLETE_MARKER` in backend/mlxstudio_backend/server.py),
    /// so an interrupted download shows as not downloaded and can be resumed.
    static let downloadCompleteMarker = ".mlxstudio-download-complete"

    private func detectLocalState(for model: ModelDescriptor) -> DownloadState {
        let marker = modelsCacheDirectory
            .appendingPathComponent(model.id, isDirectory: true)
            .appendingPathComponent(Self.downloadCompleteMarker)
        return FileManager.default.fileExists(atPath: marker.path) ? .ready : .notDownloaded
    }

    func ensureDownloaded(_ model: ModelDescriptor) async {
        if case .ready = downloadStates[model.id] { return }
        downloadStates[model.id] = .downloading(fractionComplete: 0)
        do {
            for try await update in client.download(modelId: model.id) {
                switch update {
                case let .progress(done, total):
                    let fraction = total > 0 ? Double(done) / Double(total) : 0
                    downloadStates[model.id] = .downloading(fractionComplete: fraction)
                case .completed:
                    downloadStates[model.id] = .ready
                case let .failed(_, message):
                    downloadStates[model.id] = .failed(message: message)
                }
            }
        } catch {
            downloadStates[model.id] = .failed(message: error.localizedDescription)
        }
    }
}
