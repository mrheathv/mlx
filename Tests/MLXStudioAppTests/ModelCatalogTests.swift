import XCTest
@testable import MLXStudioApp

/// Unverified in this sandbox — see note in HistoryStoreTests.swift. This
/// specifically guards against `Resources/models.json` and the Python
/// registry (`backend/mlxstudio_backend/models.py`) drifting apart, which
/// ARCHITECTURE.md §6 flags as a manual-sync risk.
final class ModelCatalogTests: XCTestCase {
    func testBundledCatalogLoadsAndHasThreeModels() throws {
        let catalog = try ModelCatalog.loadBundled()
        XCTAssertEqual(catalog.models.count, 3)
    }

    func testBundledCatalogIdsMatchPythonRegistry() throws {
        let catalog = try ModelCatalog.loadBundled()
        let ids = Set(catalog.models.map(\.id))
        XCTAssertEqual(ids, ["flux1-schnell-4bit", "flux1-dev-4bit", "z-image-turbo-4bit"])
    }

    func testExactlyOneModelPerTier() throws {
        let catalog = try ModelCatalog.loadBundled()
        let tiers = catalog.models.map(\.tier)
        XCTAssertEqual(Set(tiers).count, 3, "expected one fast / one quality / one alternate model")
    }

    func testSchnellDoesNotSupportGuidanceScale() throws {
        let catalog = try ModelCatalog.loadBundled()
        let schnell = try XCTUnwrap(catalog.models.first { $0.id == "flux1-schnell-4bit" })
        XCTAssertFalse(schnell.supportsGuidanceScale)
    }
}
