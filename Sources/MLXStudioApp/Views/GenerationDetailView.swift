import SwiftUI

struct GenerationDetailView: View {
    let entry: HistoryEntry

    @EnvironmentObject private var galleryViewModel: GalleryViewModel
    @EnvironmentObject private var generationViewModel: GenerationViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        HSplitView {
            AsyncImageFromDisk(url: galleryViewModel.imageURL(for: entry))
                .aspectRatio(contentMode: .fit)
                .frame(minWidth: 320)

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    metadataRow("Model", entry.modelDisplayName)
                    Divider()
                    metadataRow("Prompt", entry.prompt, multiline: true)
                    if !entry.negativePrompt.isEmpty {
                        metadataRow("Negative prompt", entry.negativePrompt, multiline: true)
                    }
                    Divider()
                    metadataRow("Steps", "\(entry.parameters.steps)")
                    metadataRow("Guidance", String(format: "%.1f", entry.parameters.guidanceScale))
                    metadataRow("Size", "\(entry.parameters.width) × \(entry.parameters.height)")
                    metadataRow("Seed", "\(entry.seedUsed)")
                    if let seconds = entry.generationSeconds {
                        metadataRow("Generation time", String(format: "%.1fs", seconds))
                    }
                    metadataRow("Created", entry.createdAt.formatted(date: .abbreviated, time: .shortened))

                    Spacer(minLength: 16)

                    HStack {
                        Button("Reuse Settings") {
                            generationViewModel.reuse(entry)
                            dismiss()
                        }
                        .buttonStyle(.borderedProminent)

                        Button("Delete", role: .destructive) {
                            galleryViewModel.delete(entry)
                            dismiss()
                        }
                    }
                }
                .padding()
            }
            .frame(minWidth: 280)
        }
        .frame(minWidth: 700, minHeight: 480)
    }

    private func metadataRow(_ label: String, _ value: String, multiline: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .lineLimit(multiline ? nil : 1)
                .textSelection(.enabled)
        }
    }
}
