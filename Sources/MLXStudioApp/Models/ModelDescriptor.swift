import Foundation

/// A switchable generation model, as listed in `Resources/models.json`.
///
/// This is a Swift-side mirror of `backend/mlxstudio_backend/models.py`'s
/// registry (the Python side is authoritative — see ARCHITECTURE.md §2/§6).
struct ModelDescriptor: Codable, Identifiable, Hashable {
    enum Tier: String, Codable {
        case fast
        case quality
        case alternate
    }

    enum Engine: String, Codable {
        case mflux
        case diffusionkit
    }

    let id: String
    let displayName: String
    let tier: Tier
    let engine: Engine
    let family: String
    let hfRepo: String
    let fallbackHfRepo: String?
    let approxDownloadGB: Double
    let approxRuntimeRAMGB: Double
    let supportsGuidanceScale: Bool
    let defaultSteps: Int
    let maxRecommendedSteps: Int
    let defaultGuidance: Double
    let defaultWidth: Int
    let defaultHeight: Int
}

struct ModelCatalog: Codable {
    let schemaVersion: Int
    let models: [ModelDescriptor]

    static func loadBundled() throws -> ModelCatalog {
        guard let url = Bundle.module.url(forResource: "models", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ModelCatalog.self, from: data)
    }
}
