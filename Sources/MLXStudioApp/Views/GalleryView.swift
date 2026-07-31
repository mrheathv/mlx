import SwiftUI

struct GalleryView: View {
    @EnvironmentObject private var galleryViewModel: GalleryViewModel

    private let columns = [GridItem(.adaptive(minimum: 200), spacing: 12)]

    var body: some View {
        Group {
            if galleryViewModel.entries.isEmpty {
                ContentUnavailableView(
                    "No generations yet",
                    systemImage: "photo.on.rectangle.angled",
                    description: Text("Images you generate will show up here.")
                )
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(galleryViewModel.entries) { entry in
                            Button {
                                galleryViewModel.selectedEntry = entry
                            } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    AsyncImageFromDisk(url: galleryViewModel.imageURL(for: entry))
                                        .aspectRatio(1, contentMode: .fit)
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                    Text(entry.prompt)
                                        .font(.caption)
                                        .lineLimit(2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding()
                }
            }
        }
        .navigationTitle("Gallery")
        .sheet(item: $galleryViewModel.selectedEntry) { entry in
            GenerationDetailView(entry: entry)
        }
    }
}
