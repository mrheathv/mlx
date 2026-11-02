import AppKit
import SwiftUI

/// Shown instead of the main UI until the packaged app's Python
/// environment exists (see `BackendEnvironment`). One-time per install,
/// and again only if an app update changes the backend's dependencies.
struct BackendSetupView: View {
    @EnvironmentObject private var backendEnvironment: BackendEnvironment

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("One-time setup", systemImage: "shippingbox")
                .font(.title2.bold())

            Text(
                "MLX Studio runs image models through a private Python environment. " +
                "Setting it up downloads Python 3.11 and the MLX libraries (roughly 1 GB). " +
                "Nothing is installed outside the app's own folders, and models are downloaded separately when you first pick one."
            )
            .fixedSize(horizontal: false, vertical: true)
            .foregroundStyle(.secondary)

            status

            if !backendEnvironment.recentOutput.isEmpty {
                outputLog
            }

            HStack {
                Button("Show Logs in Finder") { LogFile.revealInFinder() }
                Spacer()
                actionButton
            }
        }
        .padding(28)
        .frame(minWidth: 560, minHeight: 360)
    }

    @ViewBuilder
    private var status: some View {
        switch backendEnvironment.state {
        case .checking, .needsSetup, .ready:
            EmptyView()
        case let .installing(step):
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text(step)
            }
        case let .failed(message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
                .textSelection(.enabled)
        }
    }

    private var outputLog: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(backendEnvironment.recentOutput.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .padding(8)
            }
            .frame(maxHeight: 180)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            .onChange(of: backendEnvironment.recentOutput.count) { _, count in
                proxy.scrollTo(count - 1, anchor: .bottom)
            }
        }
    }

    @ViewBuilder
    private var actionButton: some View {
        switch backendEnvironment.state {
        case .installing:
            Button("Setting Up…") {}.disabled(true)
        case .failed:
            Button("Try Again") { Task { await backendEnvironment.install() } }
                .buttonStyle(.borderedProminent)
        default:
            Button("Set Up") { Task { await backendEnvironment.install() } }
                .buttonStyle(.borderedProminent)
        }
    }
}

extension LogFile {
    @MainActor
    static func revealInFinder() {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([directory])
    }
}
