import Foundation

/// Request-scoped updates for a single generation, delivered as they
/// arrive from the backend. Terminal cases (`completed`, `failed`) end the
/// stream.
enum GenerationUpdate {
    case modelLoading
    case progress(step: Int, totalSteps: Int, elapsedSeconds: Double)
    case completed(images: [BackendEvent.ImageResult], generationSeconds: Double)
    case failed(code: String, message: String)
}

enum DownloadUpdate {
    case progress(bytesDone: Int64, bytesTotal: Int64)
    case completed
    case failed(code: String, message: String)
}

/// Translates `BackendProcessManager`'s raw per-request event streams into
/// the typed update streams the view models consume.
final class InferenceBackendClient {
    private let processManager: BackendProcessManager

    init(processManager: BackendProcessManager) {
        self.processManager = processManager
    }

    /// The image files this yields on `.completed` are written by the
    /// backend directly into `outputDirectory` and are the caller's
    /// responsibility to hand to `HistoryStore`.
    func generate(_ request: GenerationRequest, outputDirectory: URL) -> AsyncThrowingStream<GenerationUpdate, Error> {
        let command = BackendCommand.generate(
            id: request.id,
            modelId: request.modelId,
            request: request,
            outputDirectory: outputDirectory.path
        )
        return stream(command, requestId: request.id) { event in
            switch event {
            case .modelLoading:
                return .modelLoading
            case let .progress(_, step, total, elapsed):
                return .progress(step: step, totalSteps: total, elapsedSeconds: elapsed)
            case let .result(_, images, seconds):
                return .completed(images: images, generationSeconds: seconds)
            case let .error(_, code, message):
                return .failed(code: code, message: message)
            default:
                return nil
            }
        }
    }

    func download(modelId: String) -> AsyncThrowingStream<DownloadUpdate, Error> {
        let id = UUID()
        return stream(.download(id: id, modelId: modelId), requestId: id) { event in
            switch event {
            case let .downloadProgress(_, done, total):
                return .progress(bytesDone: done, bytesTotal: total)
            case .downloadComplete:
                return .completed
            case let .error(_, code, message):
                return .failed(code: code, message: message)
            default:
                return nil
            }
        }
    }

    func cancel(requestId: UUID) async throws {
        try await processManager.send(.cancel(id: requestId))
    }

    func shutdown() async {
        await processManager.stop()
    }

    /// The backend sends exactly one terminal event (`result`,
    /// `download_complete` or `error`) per request; the stream ends there.
    private func stream<Update>(
        _ command: BackendCommand,
        requestId: UUID,
        translate: @escaping (BackendEvent) -> Update?
    ) -> AsyncThrowingStream<Update, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let events = try await processManager.send(command, awaitingEventsFor: requestId)
                    for await event in events {
                        guard let update = translate(event) else { continue }
                        continuation.yield(update)
                        if event.isTerminal { break }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

private extension BackendEvent {
    var isTerminal: Bool {
        switch self {
        case .result, .downloadComplete, .error: return true
        default: return false
        }
    }
}
