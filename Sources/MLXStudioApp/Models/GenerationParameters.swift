import Foundation

/// User-tunable generation settings, independent of which model they're
/// applied to (defaults for a given model come from `ModelDescriptor`).
struct GenerationParameters: Codable, Hashable {
    var steps: Int
    var guidanceScale: Double
    var width: Int
    var height: Int
    var batchSize: Int
    var seed: UInt32
    var seedLocked: Bool

    static func defaults(for model: ModelDescriptor) -> GenerationParameters {
        GenerationParameters(
            steps: model.defaultSteps,
            guidanceScale: model.defaultGuidance,
            width: model.defaultWidth,
            height: model.defaultHeight,
            batchSize: 1,
            seed: UInt32.random(in: 0...UInt32.max),
            seedLocked: false
        )
    }

    /// Returns a copy with a freshly randomized seed, unless the seed is locked.
    func rerolled() -> GenerationParameters {
        guard !seedLocked else { return self }
        var copy = self
        copy.seed = UInt32.random(in: 0...UInt32.max)
        return copy
    }
}

enum AspectRatioPreset: String, CaseIterable, Identifiable {
    case square = "1:1"
    case portrait = "3:4"
    case landscape = "4:3"
    case widescreen = "16:9"

    var id: String { rawValue }

    /// (width, height) at a ~1024px-long-edge budget, rounded to multiples
    /// of 16 as most MLX diffusion VAEs expect.
    func dimensions() -> (width: Int, height: Int) {
        switch self {
        case .square: return (1024, 1024)
        case .portrait: return (896, 1152)
        case .landscape: return (1152, 896)
        case .widescreen: return (1280, 720)
        }
    }
}
