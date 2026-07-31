import XCTest
@testable import MLXStudioApp

/// NOTE: written against documented Foundation/XCTest APIs but not run —
/// this sandbox has no Swift toolchain (see ARCHITECTURE.md §6). Run with
/// `swift test` on macOS before relying on this passing.
final class HistoryStoreTests: XCTestCase {
    private var tempRoot: URL!
    private var store: HistoryStore!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxstudio-tests-\(UUID().uuidString)", isDirectory: true)
        store = try HistoryStore(rootDirectory: tempRoot)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempRoot)
    }

    private func makeSourceImage(named name: String = "\(UUID().uuidString).png") throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        try Data("fake-png".utf8).write(to: url)
        return url
    }

    private func makeEntry(imageFilename: String) -> HistoryEntry {
        HistoryEntry(
            id: UUID(),
            batchId: UUID(),
            createdAt: Date(),
            modelId: "flux1-schnell-4bit",
            modelDisplayName: "FLUX.1 [schnell] — Fast",
            prompt: "a cat wearing sunglasses",
            negativePrompt: "",
            parameters: GenerationParameters(
                steps: 4, guidanceScale: 1.0, width: 1024, height: 1024,
                batchSize: 1, seed: 42, seedLocked: true
            ),
            seedUsed: 42,
            imageFilename: imageFilename,
            generationSeconds: 3.2
        )
    }

    func testRecordMovesImageAndPersistsMetadata() throws {
        let source = try makeSourceImage()
        let entry = makeEntry(imageFilename: source.lastPathComponent)

        let saved = try store.record(entry, sourceImageURL: source)

        XCTAssertFalse(FileManager.default.fileExists(atPath: source.path), "source should be moved, not copied")
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.imageURL(for: saved).path))
        XCTAssertEqual(store.all().count, 1)
        XCTAssertEqual(store.all().first?.prompt, "a cat wearing sunglasses")
    }

    func testAllReturnsNewestFirst() throws {
        let older = makeEntry(imageFilename: "older.png")
        let newer = makeEntry(imageFilename: "newer.png")
        var olderEntry = older
        olderEntry.createdAt = Date().addingTimeInterval(-3600)
        var newerEntry = newer
        newerEntry.createdAt = Date()

        _ = try store.record(olderEntry, sourceImageURL: makeSourceImage(named: "older.png"))
        _ = try store.record(newerEntry, sourceImageURL: makeSourceImage(named: "newer.png"))

        let all = store.all()
        XCTAssertEqual(all.count, 2)
        XCTAssertEqual(all.first?.imageFilename, "newer.png")
    }

    func testDeleteRemovesImageAndIndexEntry() throws {
        let source = try makeSourceImage()
        let entry = makeEntry(imageFilename: source.lastPathComponent)
        let saved = try store.record(entry, sourceImageURL: source)

        try store.delete(id: saved.id)

        XCTAssertTrue(store.all().isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.imageURL(for: saved).path))
    }

    func testIndexSurvivesReload() throws {
        let source = try makeSourceImage()
        let entry = makeEntry(imageFilename: source.lastPathComponent)
        _ = try store.record(entry, sourceImageURL: source)

        let reloaded = try HistoryStore(rootDirectory: tempRoot)
        XCTAssertEqual(reloaded.all().count, 1)
        XCTAssertEqual(reloaded.all().first?.modelId, "flux1-schnell-4bit")
    }

    func testAsReusableRequestLocksSeedToTheOneActuallyUsed() throws {
        var entry = makeEntry(imageFilename: "x.png")
        entry.parameters.seed = 1
        entry.parameters.seedLocked = false
        entry.seedUsed = 999 // distinct from parameters.seed, as a batch member would have

        let request = entry.asReusableRequest()

        XCTAssertEqual(request.parameters.seed, 999)
        XCTAssertTrue(request.parameters.seedLocked)
        XCTAssertEqual(request.modelId, entry.modelId)
        XCTAssertEqual(request.prompt, entry.prompt)
    }
}
