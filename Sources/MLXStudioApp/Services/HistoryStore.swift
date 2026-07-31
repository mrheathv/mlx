import Foundation

/// Local, file-backed generation history: images live as plain files on
/// disk, metadata lives in a single JSON index file alongside them.
///
/// A JSON index (rather than SQLite) was chosen because v1's access pattern
/// is "load the whole list, filter/sort in memory, append occasionally" —
/// a few thousand entries at most for a local single-user history. That's
/// well within what a flat JSON file handles fine, and it avoids pulling in
/// a SQLite dependency (or hand-rolling `NSPersistentContainer`/Core Data
/// setup) for a problem this small. Revisit if history grows large enough
/// that whole-file rewrites become a real cost (see ARCHITECTURE.md future work).
///
/// Deliberately has no dependency on SwiftUI/Combine so it can be unit
/// tested in isolation; `GalleryViewModel` wraps it for `@Published` state.
final class HistoryStore {
    enum StoreError: Error {
        case indexCorrupt(underlying: Error)
    }

    let rootDirectory: URL
    let imagesDirectory: URL
    private let indexFileURL: URL
    private let fileManager: FileManager
    private let queue = DispatchQueue(label: "com.mlxstudio.historystore", qos: .userInitiated)

    private var cachedEntries: [HistoryEntry]

    /// - Parameter rootDirectory: defaults to
    ///   `~/Library/Application Support/MLXStudio`; overridable for tests.
    init(rootDirectory: URL? = nil, fileManager: FileManager = .default) throws {
        self.fileManager = fileManager
        let root: URL
        if let rootDirectory {
            root = rootDirectory
        } else {
            let appSupport = try fileManager.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            root = appSupport.appendingPathComponent("MLXStudio", isDirectory: true)
        }
        self.rootDirectory = root
        self.imagesDirectory = root.appendingPathComponent("Generations", isDirectory: true)
        self.indexFileURL = root.appendingPathComponent("history.json")

        try fileManager.createDirectory(at: imagesDirectory, withIntermediateDirectories: true)

        if fileManager.fileExists(atPath: indexFileURL.path) {
            let data = try Data(contentsOf: indexFileURL)
            do {
                cachedEntries = try JSONDecoder.mlxStudio.decode([HistoryEntry].self, from: data)
            } catch {
                throw StoreError.indexCorrupt(underlying: error)
            }
        } else {
            cachedEntries = []
        }
    }

    /// All entries, newest first.
    func all() -> [HistoryEntry] {
        queue.sync { cachedEntries.sorted { $0.createdAt > $1.createdAt } }
    }

    func entry(id: UUID) -> HistoryEntry? {
        queue.sync { cachedEntries.first { $0.id == id } }
    }

    func imageURL(for entry: HistoryEntry) -> URL {
        imagesDirectory.appendingPathComponent(entry.imageFilename)
    }

    /// Takes ownership of the file at `sourceImageURL` (moving it into the
    /// managed images directory if it isn't already there) and records the
    /// metadata. Returns the persisted entry, with `imageFilename`
    /// pointing at its final on-disk name.
    @discardableResult
    func record(_ entry: HistoryEntry, sourceImageURL: URL) throws -> HistoryEntry {
        try queue.sync {
            var finalEntry = entry
            let destinationURL = imagesDirectory.appendingPathComponent(entry.imageFilename)
            if sourceImageURL.standardizedFileURL != destinationURL.standardizedFileURL {
                if fileManager.fileExists(atPath: destinationURL.path) {
                    try fileManager.removeItem(at: destinationURL)
                }
                try fileManager.moveItem(at: sourceImageURL, to: destinationURL)
            }
            finalEntry.imageFilename = destinationURL.lastPathComponent
            cachedEntries.append(finalEntry)
            try persist()
            return finalEntry
        }
    }

    func delete(id: UUID) throws {
        try queue.sync {
            guard let index = cachedEntries.firstIndex(where: { $0.id == id }) else { return }
            let entry = cachedEntries[index]
            let imageURL = imagesDirectory.appendingPathComponent(entry.imageFilename)
            if fileManager.fileExists(atPath: imageURL.path) {
                try fileManager.removeItem(at: imageURL)
            }
            cachedEntries.remove(at: index)
            try persist()
        }
    }

    private func persist() throws {
        let data = try JSONEncoder.mlxStudio.encode(cachedEntries)
        try data.write(to: indexFileURL, options: .atomic)
    }
}

extension JSONEncoder {
    static let mlxStudio: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()
}

extension JSONDecoder {
    static let mlxStudio: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}
