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

/// Demultiplexes the backend's single shared event stream (keyed by
/// request `id`) into focused `AsyncStream`s per in-flight request, and
/// turns `BackendCommand`s into writes on the process's stdin.
final class InferenceBackendClient {
    private let processManager: BackendProcessManager
    private var dispatchTask: Task<Void, Never>?
    private var waiters: [UUID: AsyncStream<BackendEvent>.Continuation] = [:]
    private let waitersLock = NSLock()

    init(processManager: BackendProcessManager = BackendProcessManager()) {
        self.processManager = processManager
    }

    private func ensureDispatching() async throws {
        guard dispatchTask == nil else { return }
        let stream = try await processManager.ensureStarted()
        dispatchTask = Task { [weak self] in
            for await event in stream {
                self?.dispatch(event)
            }
        }
    }

    private func dispatch(_ event: BackendEvent) {
        let id: UUID?
        switch event {
        case .ready, .unknown: id = nil
        case let .progress(reqId, _, _, _): id = reqId
        case let .modelLoading(reqId): id = reqId
        case let .downloadProgress(reqId, _, _): id = reqId
        case let .downloadComplete(reqId): id = reqId
        case let .result(reqId, _, _): id = reqId
        case let .modelList(reqId, _): id = reqId
        case let .error(reqId, _, _): id = reqId
        }
        guard let id else { return }
        waitersLock.lock()
        let continuation = waiters[id]
        waitersLock.unlock()
        continuation?.yield(event)
    }

    private func registerWaiter(id: UUID) -> AsyncStream<BackendEvent> {
        AsyncStream { continuation in
            waitersLock.lock()
            waiters[id] = continuation
            waitersLock.unlock()
            continuation.onTermination = { [weak self] _ in
                self?.waitersLock.lock()
                self?.waiters.removeValue(forKey: id)
                self?.waitersLock.unlock()
            }
        }
    }

    /// Streams updates for one generation request. The image files this
    /// yields on `.completed` are written by the backend directly into
    /// `outputDirectory` and are the caller's responsibility to hand to
    /// `HistoryStore`.
    func generate(_ request: GenerationRequest, outputDirectory: URL) -> AsyncThrowingStream<GenerationUpdate, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await ensureDispatching()
                    let requestEvents = registerWaiter(id: request.id)
                    try await processManager.send(.generate(
                        id: request.id,
                        modelId: request.modelId,
                        request: request,
                        outputDirectory: outputDirectory.path
                    ))
                    for await event in requestEvents {
                        switch event {
                        case .modelLoading:
                            continuation.yield(.modelLoading)
                        case let .progress(_, step, total, elapsed):
                            continuation.yield(.progress(step: step, totalSteps: total, elapsedSeconds: elapsed))
                        case let .result(_, images, seconds):
                            continuation.yield(.completed(images: images, generationSeconds: seconds))
                            continuation.finish()
                            return
                        case let .error(_, code, message):
                            continuation.yield(.failed(code: code, message: message))
                            continuation.finish()
                            return
                        default:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func download(modelId: String) -> AsyncThrowingStream<DownloadUpdate, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await ensureDispatching()
                    let id = UUID()
                    let requestEvents = registerWaiter(id: id)
                    try await processManager.send(.download(id: id, modelId: modelId))
                    for await event in requestEvents {
                        switch event {
                        case let .downloadProgress(_, done, total):
                            continuation.yield(.progress(bytesDone: done, bytesTotal: total))
                        case .downloadComplete:
                            continuation.yield(.completed)
                            continuation.finish()
                            return
                        case let .error(_, code, message):
                            continuation.yield(.failed(code: code, message: message))
                            continuation.finish()
                            return
                        default:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func cancel(requestId: UUID) async throws {
        try await processManager.send(.cancel(id: requestId))
    }

    func shutdown() async {
        dispatchTask?.cancel()
        await processManager.stop()
    }
}
