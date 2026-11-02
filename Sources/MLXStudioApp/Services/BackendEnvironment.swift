import Foundation

/// Provisions the Python runtime the backend needs when running as a
/// packaged .app, so a user never has to install Python, Homebrew or a
/// venv themselves (see ARCHITECTURE.md §5).
///
/// The .app bundles `uv` (Contents/MacOS/uv) and the backend's source
/// (Contents/Resources/backend). First launch runs:
///   uv venv  → downloads a uv-managed CPython 3.11 into Application Support
///   uv pip install -r backend/pyproject.toml → MLX, mflux and their deps
/// The environment counts as installed only if a copy of the exact
/// pyproject.toml it was built from matches the bundled one, so an app
/// update that changes dependencies re-runs setup automatically.
///
/// Under `swift run` there is no bundle layout to use, so `paths` is nil,
/// state is immediately `.ready`, and the backend runs from the repo
/// (`BackendProcessManager.Configuration.developmentDefault()`).
@MainActor
final class BackendEnvironment: ObservableObject {
    enum State: Equatable {
        case checking
        case needsSetup
        case installing(step: String)
        case ready
        case failed(message: String)
    }

    struct Paths {
        let backendDirectory: URL
        let uvExecutable: URL
        let environmentDirectory: URL
        let pythonInstallDirectory: URL
        let uvCacheDirectory: URL
        let pycacheDirectory: URL

        var pythonExecutable: URL { environmentDirectory.appendingPathComponent("bin/python3") }
        var requirementsFile: URL { backendDirectory.appendingPathComponent("pyproject.toml") }
        var installedRequirementsFile: URL { environmentDirectory.appendingPathComponent("installed-pyproject.toml") }

        /// nil unless running from a .app built by scripts/build-dmg.sh.
        static func forRunningApp(bundle: Bundle = .main) -> Paths? {
            guard bundle.bundleURL.pathExtension == "app",
                  let resources = bundle.resourceURL,
                  let executables = bundle.executableURL?.deletingLastPathComponent() else {
                return nil
            }
            let fileManager = FileManager.default
            let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("MLXStudio", isDirectory: true)
            let caches = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("MLXStudio", isDirectory: true)
            return Paths(
                backendDirectory: resources.appendingPathComponent("backend", isDirectory: true),
                uvExecutable: executables.appendingPathComponent("uv"),
                environmentDirectory: appSupport.appendingPathComponent("python-env", isDirectory: true),
                pythonInstallDirectory: appSupport.appendingPathComponent("python-installs", isDirectory: true),
                uvCacheDirectory: caches.appendingPathComponent("uv", isDirectory: true),
                pycacheDirectory: caches.appendingPathComponent("pycache", isDirectory: true)
            )
        }
    }

    enum SetupError: LocalizedError {
        case commandFailed(command: String, status: Int32, lastLine: String?)

        var errorDescription: String? {
            switch self {
            case let .commandFailed(command, status, lastLine):
                var message = "`\(command)` failed (exit status \(status))."
                if let lastLine { message += " \(lastLine)" }
                return message
            }
        }
    }

    @Published private(set) var state: State = .checking
    /// Tail of the setup commands' combined output, for the setup screen.
    @Published private(set) var recentOutput: [String] = []

    let paths: Paths?
    private let log = LogFile(url: LogFile.directory.appendingPathComponent("setup.log"))

    init(paths: Paths?) {
        self.paths = paths
    }

    nonisolated static func backendConfiguration(paths: Paths?) -> BackendProcessManager.Configuration {
        guard let paths else { return .developmentDefault() }
        return BackendProcessManager.Configuration(
            pythonExecutable: paths.pythonExecutable,
            extraArguments: [],
            backendModuleDirectory: paths.backendDirectory,
            environment: [
                "PYTHONUNBUFFERED": "1",
                // Keep Python from writing __pycache__ into the signed .app.
                "PYTHONPYCACHEPREFIX": paths.pycacheDirectory.path,
            ],
            logFileURL: LogFile.directory.appendingPathComponent("backend.log")
        )
    }

    func check() {
        guard let paths else {
            state = .ready
            return
        }
        state = Self.isInstalled(paths) ? .ready : .needsSetup
    }

    private static func isInstalled(_ paths: Paths) -> Bool {
        guard FileManager.default.isExecutableFile(atPath: paths.pythonExecutable.path),
              let installed = try? Data(contentsOf: paths.installedRequirementsFile),
              let required = try? Data(contentsOf: paths.requirementsFile) else {
            return false
        }
        return installed == required
    }

    func install() async {
        guard let paths else { return }
        if case .installing = state { return }
        recentOutput = []
        log.append("=== setup started \(Date())")
        do {
            try? FileManager.default.removeItem(at: paths.environmentDirectory)

            state = .installing(step: "Downloading Python 3.11…")
            try await run(paths, [
                "venv", paths.environmentDirectory.path,
                "--python", "3.11",
                "--managed-python",
            ])

            state = .installing(step: "Installing MLX and mflux (roughly 1 GB)…")
            try await run(paths, [
                "pip", "install",
                "--python", paths.pythonExecutable.path,
                "-r", paths.requirementsFile.path,
            ])

            try FileManager.default.copyItem(at: paths.requirementsFile, to: paths.installedRequirementsFile)
            log.append("=== setup finished")
            state = .ready
        } catch {
            log.append("=== setup failed: \(error.localizedDescription)")
            state = .failed(message: error.localizedDescription)
        }
    }

    private func appendOutput(_ line: String) {
        recentOutput.append(line)
        if recentOutput.count > 200 {
            recentOutput.removeFirst(recentOutput.count - 200)
        }
    }

    /// Runs `uv` with its stdout and stderr merged, streaming each line
    /// to the setup screen and setup.log.
    private func run(_ paths: Paths, _ arguments: [String]) async throws {
        let process = Process()
        process.executableURL = paths.uvExecutable
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        environment["UV_PYTHON_INSTALL_DIR"] = paths.pythonInstallDirectory.path
        environment["UV_CACHE_DIR"] = paths.uvCacheDirectory.path
        environment["NO_COLOR"] = "1"
        process.environment = environment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        let commandDescription = "uv " + arguments.prefix(2).joined(separator: " ")
        log.append("$ uv \(arguments.joined(separator: " "))")

        let log = self.log
        let reader = Task { @MainActor [weak self] in
            for try await line in pipe.fileHandleForReading.bytes.lines {
                log.append(line)
                self?.appendOutput(line)
            }
        }

        let status: Int32 = try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { continuation.resume(returning: $0.terminationStatus) }
            do {
                try process.run()
            } catch {
                process.terminationHandler = nil
                continuation.resume(throwing: error)
            }
        }
        _ = try? await reader.value

        guard status == 0 else {
            throw SetupError.commandFailed(command: commandDescription, status: status, lastLine: recentOutput.last)
        }
    }
}
