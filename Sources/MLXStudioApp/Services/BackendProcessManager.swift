import Foundation

/// Supervises the Python/MLX inference backend as a long-lived child
/// process, kept warm across generations so loaded model weights stay
/// resident in unified memory instead of being reloaded per request
/// (see ARCHITECTURE.md §1).
///
/// Talks newline-delimited JSON over the child's own stdin/stdout pipes —
/// deliberately not a socket, see ARCHITECTURE.md §1 for why. Events are
/// routed to per-request streams by `id`. If the process dies, every
/// in-flight request gets a `backend_exited` error instead of waiting
/// forever, and the next `send` starts a fresh process.
actor BackendProcessManager {
    enum ManagerError: Error {
        case notRunning
    }

    struct Configuration {
        var pythonExecutable: URL
        var extraArguments: [String]
        var backendModuleDirectory: URL
        var moduleName: String = "mlxstudio_backend"
        var environment: [String: String] = [:]
        var logFileURL: URL?

        /// `swift run` / Xcode: the repo's `backend/` directory, using
        /// `backend/.venv` if it exists (Xcode's Run button doesn't inherit
        /// a Terminal's PATH or activated venv) and otherwise `python3` on
        /// PATH. The packaged app uses `BackendEnvironment.backendConfiguration`.
        static func developmentDefault() -> Configuration {
            let backendDir = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent() // Services/
                .deletingLastPathComponent() // MLXStudioApp/
                .deletingLastPathComponent() // Sources/
                .deletingLastPathComponent() // repo root
                .appendingPathComponent("backend", isDirectory: true)

            let venvPython = backendDir.appendingPathComponent(".venv/bin/python3")
            let usesVenv = FileManager.default.isExecutableFile(atPath: venvPython.path)
            return Configuration(
                pythonExecutable: usesVenv ? venvPython : URL(fileURLWithPath: "/usr/bin/env"),
                extraArguments: usesVenv ? [] : ["python3"],
                backendModuleDirectory: backendDir,
                environment: ["PYTHONUNBUFFERED": "1"],
                logFileURL: LogFile.directory.appendingPathComponent("backend.log")
            )
        }
    }

    private let configuration: Configuration
    private let log: LogFile?
    private var process: Process?
    private var stdinHandle: FileHandle?
    /// Bumped per launch so a dead process's reader can't tear down its successor.
    private var generation = 0
    private var waiters: [UUID: AsyncStream<BackendEvent>.Continuation] = [:]
    private var lastStderrLine: String?

    init(configuration: Configuration) {
        self.configuration = configuration
        self.log = configuration.logFileURL.map(LogFile.init(url:))
    }

    var isRunning: Bool {
        process?.isRunning ?? false
    }

    /// Starts the backend if needed, registers for events tagged with
    /// `requestId`, then writes `command`. Registering before writing means
    /// a fast reply can't arrive before anyone is listening for it.
    func send(_ command: BackendCommand, awaitingEventsFor requestId: UUID) throws -> AsyncStream<BackendEvent> {
        try ensureStarted()
        let (stream, continuation) = AsyncStream<BackendEvent>.makeStream()
        waiters[requestId] = continuation
        continuation.onTermination = { [weak self] _ in
            Task { await self?.removeWaiter(requestId) }
        }
        do {
            try write(command)
        } catch {
            waiters.removeValue(forKey: requestId)
            continuation.finish()
            throw error
        }
        return stream
    }

    /// Fire-and-forget commands (cancel) that expect no reply of their own.
    func send(_ command: BackendCommand) throws {
        guard isRunning else { throw ManagerError.notRunning }
        try write(command)
    }

    func stop() {
        process?.terminate()
    }

    private func write(_ command: BackendCommand) throws {
        guard let stdinHandle, isRunning else { throw ManagerError.notRunning }
        try stdinHandle.write(contentsOf: try command.encodedLine())
    }

    private func ensureStarted() throws {
        if isRunning { return }

        let process = Process()
        process.executableURL = configuration.pythonExecutable
        process.arguments = configuration.extraArguments + ["-m", configuration.moduleName]
        process.currentDirectoryURL = configuration.backendModuleDirectory
        process.environment = ProcessInfo.processInfo.environment.merging(configuration.environment) { $1 }

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        log?.append("--- starting backend: \(configuration.pythonExecutable.path) \(process.arguments?.joined(separator: " ") ?? "")")
        try process.run()

        generation += 1
        let launchGeneration = generation
        self.process = process
        self.stdinHandle = stdinPipe.fileHandleForWriting
        self.lastStderrLine = nil

        Task { [weak self] in
            do {
                for try await line in stdoutPipe.fileHandleForReading.bytes.lines {
                    guard let event = try? JSONDecoder().decode(BackendEvent.self, from: Data(line.utf8)) else {
                        continue
                    }
                    await self?.route(event)
                }
            } catch {
                // Pipe closed — handled below the same way as a clean EOF.
            }
            await self?.backendOutputClosed(generation: launchGeneration)
        }

        // stderr carries Python tracebacks and mflux/tqdm output: logged
        // to a file (and the console, under `swift run`), not parsed.
        let log = self.log
        Task { [weak self] in
            for try await line in stderrPipe.fileHandleForReading.bytes.lines {
                log?.append(line)
                FileHandle.standardError.write(Data(("[backend] " + line + "\n").utf8))
                await self?.noteStderr(line)
            }
        }
    }

    private func route(_ event: BackendEvent) {
        guard let id = event.requestId else { return }
        waiters[id]?.yield(event)
    }

    private func noteStderr(_ line: String) {
        if !line.trimmingCharacters(in: .whitespaces).isEmpty {
            lastStderrLine = line
        }
    }

    private func removeWaiter(_ id: UUID) {
        waiters.removeValue(forKey: id)
    }

    private func backendOutputClosed(generation closedGeneration: Int) {
        guard closedGeneration == generation else { return }
        log?.append("--- backend exited")
        var message = "The image generation backend stopped unexpectedly."
        if let lastStderrLine {
            message += " Last output: \(lastStderrLine)"
        }
        message += " Full log: Help → Show Logs in Finder."
        for (id, continuation) in waiters {
            continuation.yield(.error(id: id, code: "backend_exited", message: message))
            continuation.finish()
        }
        waiters.removeAll()
        process = nil
        stdinHandle = nil
    }
}

private extension BackendEvent {
    var requestId: UUID? {
        switch self {
        case .ready, .unknown: return nil
        case let .progress(id, _, _, _): return id
        case let .modelLoading(id): return id
        case let .downloadProgress(id, _, _): return id
        case let .downloadComplete(id): return id
        case let .result(id, _, _): return id
        case let .modelList(id, _): return id
        case let .error(id, _, _): return id
        }
    }
}

/// Appends lines to a file under ~/Library/Logs/MLXStudio (visible in
/// Console.app). A packaged app has no terminal, so this is where backend
/// tracebacks and setup output end up.
final class LogFile: @unchecked Sendable {
    static let directory: URL = FileManager.default
        .urls(for: .libraryDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Logs/MLXStudio", isDirectory: true)

    private let url: URL
    private let queue = DispatchQueue(label: "com.mlxstudio.logfile")

    init(url: URL) {
        self.url = url
    }

    func append(_ line: String) {
        queue.async { [url] in
            let fileManager = FileManager.default
            try? fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !fileManager.fileExists(atPath: url.path) {
                fileManager.createFile(atPath: url.path, contents: nil)
            }
            guard let handle = try? FileHandle(forWritingTo: url) else { return }
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data((line + "\n").utf8))
        }
    }
}
