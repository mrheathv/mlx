import Foundation

/// Supervises the bundled Python/MLX inference backend as a long-lived
/// child process, kept warm across generations so loaded model weights
/// stay resident in unified memory instead of being reloaded per request
/// (see ARCHITECTURE.md §1).
///
/// Talks newline-delimited JSON over the child's own stdin/stdout pipes —
/// deliberately not a socket, see ARCHITECTURE.md §1 for why.
actor BackendProcessManager {
    enum ManagerError: Error {
        case backendNotFound
        case processExited(status: Int32)
        case notRunning
    }

    /// Where to find the backend. In development this points at the
    /// `backend/` directory checked into this repo; a packaged build would
    /// point at a bundled venv + copy of `backend/` inside the .app.
    struct Configuration {
        var pythonExecutable: URL
        var backendModuleDirectory: URL
        var moduleName: String = "mlxstudio_backend"

        static func developmentDefault() -> Configuration {
            let backendDir = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent() // Services/
                .deletingLastPathComponent() // MLXStudioApp/
                .deletingLastPathComponent() // Sources/
                .deletingLastPathComponent() // repo root
                .appendingPathComponent("backend", isDirectory: true)
            return Configuration(
                pythonExecutable: URL(fileURLWithPath: "/usr/bin/env"),
                backendModuleDirectory: backendDir
            )
        }
    }

    private let configuration: Configuration
    private var process: Process?
    private var stdinHandle: FileHandle?
    private var lineReader: AsyncLineReader?

    /// Every line the backend emits, in arrival order. Consumers
    /// (`InferenceBackendClient`) demultiplex by request id; a single
    /// shared stream keeps process lifecycle in one place.
    private(set) var events: AsyncStream<BackendEvent>?
    private var eventsContinuation: AsyncStream<BackendEvent>.Continuation?

    init(configuration: Configuration = .developmentDefault()) {
        self.configuration = configuration
    }

    var isRunning: Bool {
        process?.isRunning ?? false
    }

    /// Starts the backend if it isn't already running. Safe to call
    /// repeatedly. Returns the shared event stream.
    func ensureStarted() throws -> AsyncStream<BackendEvent> {
        if let events, isRunning {
            return events
        }

        let process = Process()
        process.executableURL = configuration.pythonExecutable
        process.arguments = ["python3", "-m", configuration.moduleName]
        process.currentDirectoryURL = configuration.backendModuleDirectory

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        var continuation: AsyncStream<BackendEvent>.Continuation!
        let stream = AsyncStream<BackendEvent> { cont in continuation = cont }
        self.eventsContinuation = continuation
        self.events = stream

        process.terminationHandler = { [weak self] proc in
            Task { await self?.handleTermination(status: proc.terminationStatus) }
        }

        try process.run()

        self.process = process
        self.stdinHandle = stdinPipe.fileHandleForWriting

        let reader = AsyncLineReader(fileHandle: stdoutPipe.fileHandleForReading)
        self.lineReader = reader
        Task { [weak self] in
            for await line in reader.lines() {
                guard let data = line.data(using: .utf8) else { continue }
                if let event = try? JSONDecoder().decode(BackendEvent.self, from: data) {
                    await self?.yieldEvent(event)
                }
            }
        }

        // stderr is logged, not parsed — Python tracebacks / mlx logging
        // land here and are surfaced for debugging, not as protocol events.
        Task {
            for try await line in stderrPipe.fileHandleForReading.bytes.lines {
                FileHandle.standardError.write(("[backend] " + line + "\n").data(using: .utf8) ?? Data())
            }
        }

        return stream
    }

    func send(_ command: BackendCommand) throws {
        guard let stdinHandle, isRunning else { throw ManagerError.notRunning }
        let line = try command.encodedLine()
        try stdinHandle.write(contentsOf: line)
    }

    func stop() {
        process?.terminate()
        process = nil
        stdinHandle = nil
        eventsContinuation?.finish()
        eventsContinuation = nil
        events = nil
    }

    private func yieldEvent(_ event: BackendEvent) {
        eventsContinuation?.yield(event)
    }

    private func handleTermination(status: Int32) {
        eventsContinuation?.finish()
        eventsContinuation = nil
        events = nil
        process = nil
        stdinHandle = nil
    }
}

/// Buffers a `FileHandle`'s bytes into newline-delimited `String`s,
/// exposed as an `AsyncStream` so `BackendProcessManager` doesn't block a
/// thread reading the pipe.
final class AsyncLineReader {
    private let fileHandle: FileHandle

    init(fileHandle: FileHandle) {
        self.fileHandle = fileHandle
    }

    func lines() -> AsyncStream<String> {
        AsyncStream { continuation in
            let task = Task {
                var buffer = Data()
                do {
                    for try await byte in fileHandle.bytes {
                        if byte == 0x0A {
                            if let line = String(data: buffer, encoding: .utf8), !line.isEmpty {
                                continuation.yield(line)
                            }
                            buffer.removeAll(keepingCapacity: true)
                        } else {
                            buffer.append(byte)
                        }
                    }
                } catch {
                    // Pipe closed / process exited — fall through to finish().
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
