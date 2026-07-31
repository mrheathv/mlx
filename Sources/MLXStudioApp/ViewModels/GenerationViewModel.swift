import Foundation

@MainActor
final class GenerationViewModel: ObservableObject {
    enum Phase: Equatable {
        case idle
        case loadingModel
        case generating(step: Int, totalSteps: Int, elapsedSeconds: Double)
        case finishing
        case failed(message: String)
    }

    @Published var prompt: String = ""
    @Published var negativePrompt: String = ""
    @Published var parameters: GenerationParameters
    @Published private(set) var phase: Phase = .idle
    @Published var pendingMemoryWarning: MemoryGuard.Assessment?
    @Published private(set) var lastCompletedEntries: [HistoryEntry] = []

    let modelManager: ModelManager
    private let client: InferenceBackendClient
    private let historyStore: HistoryStore
    private let memoryGuard = MemoryGuard()
    private var generationStartedAt: Date?
    private var activeTask: Task<Void, Never>?
    private var activeRequestId: UUID?

    init(modelManager: ModelManager, client: InferenceBackendClient, historyStore: HistoryStore) {
        self.modelManager = modelManager
        self.client = client
        self.historyStore = historyStore
        self.parameters = GenerationParameters(
            steps: 4, guidanceScale: 1.0, width: 1024, height: 1024,
            batchSize: 1, seed: UInt32.random(in: 0...UInt32.max), seedLocked: false
        )
    }

    /// Loads a past generation's settings back into the form (gallery "reuse").
    func reuse(_ entry: HistoryEntry) {
        modelManager.selectedModelId = entry.modelId
        prompt = entry.prompt
        negativePrompt = entry.negativePrompt
        parameters = entry.parameters
    }

    func onModelSelected() {
        guard let model = modelManager.selectedModel else { return }
        parameters = GenerationParameters.defaults(for: model)
    }

    var canSubmit: Bool {
        guard case .idle = phase else {
            if case .failed = phase { return true }
            return false
        }
        return !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && modelManager.selectedModel != nil
    }

    func submit(forceIgnoreMemoryWarning: Bool = false) {
        guard let model = modelManager.selectedModel else { return }

        if !forceIgnoreMemoryWarning {
            let assessment = memoryGuard.assess(model: model)
            if !assessment.isLikelyToFit {
                pendingMemoryWarning = assessment
                return
            }
        }

        let request = GenerationRequest(
            modelId: model.id,
            prompt: prompt,
            negativePrompt: negativePrompt,
            parameters: parameters
        )
        activeRequestId = request.id
        phase = .loadingModel
        generationStartedAt = Date()
        lastCompletedEntries = []

        activeTask = Task {
            let outputDirectory = historyStore.imagesDirectory
            do {
                for try await update in client.generate(request, outputDirectory: outputDirectory) {
                    switch update {
                    case .modelLoading:
                        phase = .loadingModel
                    case let .progress(step, totalSteps, elapsedSeconds):
                        phase = .generating(step: step, totalSteps: totalSteps, elapsedSeconds: elapsedSeconds)
                    case let .completed(images, generationSeconds):
                        phase = .finishing
                        persist(images: images, request: request, generationSeconds: generationSeconds)
                        phase = .idle
                    case let .failed(_, message):
                        phase = .failed(message: message)
                    }
                }
            } catch {
                phase = .failed(message: error.localizedDescription)
            }
        }
    }

    func cancel() {
        guard let activeRequestId else { return }
        activeTask?.cancel()
        Task { try? await client.cancel(requestId: activeRequestId) }
        phase = .idle
    }

    func dismissMemoryWarning() {
        pendingMemoryWarning = nil
    }

    func proceedDespiteMemoryWarning() {
        pendingMemoryWarning = nil
        submit(forceIgnoreMemoryWarning: true)
    }

    private func persist(images: [BackendEvent.ImageResult], request: GenerationRequest, generationSeconds: Double) {
        let batchId = UUID()
        var newEntries: [HistoryEntry] = []
        for image in images {
            let sourceURL = URL(fileURLWithPath: image.path)
            let entry = HistoryEntry(
                id: UUID(),
                batchId: batchId,
                createdAt: Date(),
                modelId: request.modelId,
                modelDisplayName: modelManager.selectedModel?.displayName ?? request.modelId,
                prompt: request.prompt,
                negativePrompt: request.negativePrompt,
                parameters: request.parameters,
                seedUsed: image.seed,
                imageFilename: sourceURL.lastPathComponent,
                generationSeconds: generationSeconds
            )
            if let saved = try? historyStore.record(entry, sourceImageURL: sourceURL) {
                newEntries.append(saved)
            }
        }
        lastCompletedEntries = newEntries
    }
}
