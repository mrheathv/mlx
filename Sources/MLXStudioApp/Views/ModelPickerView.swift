import SwiftUI

struct ModelPickerView: View {
    @EnvironmentObject private var modelManager: ModelManager
    @EnvironmentObject private var generationViewModel: GenerationViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("Model", selection: selectionBinding) {
                ForEach(modelManager.catalog) { model in
                    Text(model.displayName).tag(Optional(model.id))
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if let model = modelManager.selectedModel {
                statusRow(for: model)
            }
        }
    }

    private var selectionBinding: Binding<String?> {
        Binding(
            get: { modelManager.selectedModelId },
            set: { newValue in
                modelManager.selectedModelId = newValue
                generationViewModel.onModelSelected()
            }
        )
    }

    @ViewBuilder
    private func statusRow(for model: ModelDescriptor) -> some View {
        switch modelManager.downloadStates[model.id] ?? .notDownloaded {
        case .notDownloaded:
            HStack {
                Label("Not downloaded (~\(String(format: "%.1f", model.approxDownloadGB)) GB)", systemImage: "arrow.down.circle")
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Download") {
                    Task { await modelManager.ensureDownloaded(model) }
                }
            }
            .font(.callout)
        case let .downloading(fraction):
            VStack(alignment: .leading, spacing: 2) {
                ProgressView(value: fraction)
                Text("Downloading… \(Int(fraction * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .ready:
            Label("Ready", systemImage: "checkmark.circle.fill")
                .font(.callout)
                .foregroundStyle(.green)
        case let .failed(message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.red)
                .lineLimit(2)
        }
    }
}
