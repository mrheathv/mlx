import XCTest
@testable import MLXStudioApp

/// Unverified in this sandbox — see note in HistoryStoreTests.swift.
final class GenerationParametersTests: XCTestCase {
    func testRerolledChangesSeedWhenUnlocked() {
        let params = GenerationParameters(
            steps: 4, guidanceScale: 1.0, width: 1024, height: 1024,
            batchSize: 1, seed: 1, seedLocked: false
        )
        let rerolled = params.rerolled()
        // Astronomically unlikely to collide; this isn't testing
        // randomness quality, just that a reroll happens when unlocked.
        XCTAssertNotEqual(params.seed, rerolled.seed)
    }

    func testRerolledKeepsSeedWhenLocked() {
        let params = GenerationParameters(
            steps: 4, guidanceScale: 1.0, width: 1024, height: 1024,
            batchSize: 1, seed: 12345, seedLocked: true
        )
        let rerolled = params.rerolled()
        XCTAssertEqual(rerolled.seed, 12345)
    }

    func testDefaultsForModelUsesModelDescriptorDefaults() {
        let model = ModelDescriptor(
            id: "flux2-klein-4b-4bit", displayName: "FLUX.2 [klein] 4B", tier: .fast,
            engine: .mflux, family: "flux2_klein", hfRepo: "x/y",
            approxDownloadGB: 4.8, approxRuntimeRAMGB: 8, supportsGuidanceScale: false,
            supportsNegativePrompt: false,
            defaultSteps: 4, maxRecommendedSteps: 8, defaultGuidance: 1.0,
            defaultWidth: 1024, defaultHeight: 1024
        )
        let params = GenerationParameters.defaults(for: model)
        XCTAssertEqual(params.steps, 4)
        XCTAssertEqual(params.guidanceScale, 1.0)
        XCTAssertEqual(params.width, 1024)
        XCTAssertEqual(params.height, 1024)
        XCTAssertEqual(params.batchSize, 1)
        XCTAssertFalse(params.seedLocked)
    }

    func testAspectRatioDimensionsAreMultiplesOf16() {
        for preset in AspectRatioPreset.allCases {
            let dims = preset.dimensions()
            XCTAssertEqual(dims.width % 16, 0, "\(preset) width not aligned")
            XCTAssertEqual(dims.height % 16, 0, "\(preset) height not aligned")
        }
    }
}
