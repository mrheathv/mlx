import Foundation

@MainActor
final class GalleryViewModel: ObservableObject {
    @Published private(set) var entries: [HistoryEntry] = []
    @Published var selectedEntry: HistoryEntry?

    private let historyStore: HistoryStore

    init(historyStore: HistoryStore) {
        self.historyStore = historyStore
        refresh()
    }

    func refresh() {
        entries = historyStore.all()
    }

    func imageURL(for entry: HistoryEntry) -> URL {
        historyStore.imageURL(for: entry)
    }

    func delete(_ entry: HistoryEntry) {
        try? historyStore.delete(id: entry.id)
        if selectedEntry?.id == entry.id {
            selectedEntry = nil
        }
        refresh()
    }
}
