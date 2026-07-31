import SwiftUI

/// Shows per-step progress when the backend reports a step count, an
/// indeterminate spinner + elapsed time otherwise (e.g. while a model is
/// still loading). Never estimates a total duration — see
/// ARCHITECTURE.md §4 on why: a fanless machine's per-step time isn't
/// stable across a session.
struct GenerationProgressView: View {
    @EnvironmentObject private var generationViewModel: GenerationViewModel

    var body: some View {
        VStack(spacing: 16) {
            switch generationViewModel.phase {
            case .idle:
                if generationViewModel.lastCompletedEntries.isEmpty {
                    emptyState
                } else {
                    resultsGrid(generationViewModel.lastCompletedEntries)
                }
            case .loadingModel:
                indeterminate(label: "Loading model…")
            case let .generating(step, totalSteps, elapsedSeconds):
                if totalSteps > 0 {
                    VStack(spacing: 8) {
                        ProgressView(value: Double(step), total: Double(totalSteps))
                            .frame(maxWidth: 280)
                        Text("Step \(step) of \(totalSteps) · \(elapsedFormatted(elapsedSeconds))")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    indeterminate(label: "Generating… \(elapsedFormatted(elapsedSeconds))")
                }
            case .finishing:
                indeterminate(label: "Saving…")
            case .failed:
                emptyState
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "photo")
                .font(.system(size: 40))
                .foregroundStyle(.tertiary)
            Text("Generated images will appear here")
                .foregroundStyle(.secondary)
        }
    }

    private func indeterminate(label: String) -> some View {
        VStack(spacing: 8) {
            ProgressView()
                .controlSize(.large)
            Text(label)
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private func resultsGrid(_ entries: [HistoryEntry]) -> some View {
        let columns = [GridItem(.adaptive(minimum: 220), spacing: 12)]
        return ScrollView {
            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(entries) { entry in
                    HistoryThumbnail(entry: entry)
                }
            }
            .padding()
        }
    }

    private func elapsedFormatted(_ seconds: Double) -> String {
        let s = Int(seconds)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

struct HistoryThumbnail: View {
    let entry: HistoryEntry
    @EnvironmentObject private var galleryViewModel: GalleryViewModel

    var body: some View {
        AsyncImageFromDisk(url: galleryViewModel.imageURL(for: entry))
            .aspectRatio(contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

/// `AsyncImage` only loads http(s) URLs; our images are local files, so
/// this loads off the main thread and renders with a plain `NSImage`.
struct AsyncImageFromDisk: View {
    let url: URL
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
            } else {
                Rectangle()
                    .fill(.quaternary)
                    .overlay(ProgressView())
            }
        }
        .task(id: url) {
            image = await Task.detached(priority: .userInitiated) {
                NSImage(contentsOf: url)
            }.value
        }
    }
}
