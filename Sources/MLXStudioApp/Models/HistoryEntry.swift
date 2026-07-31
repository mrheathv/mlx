import Foundation

/// A single persisted generation result: one image plus the exact request
/// that produced it. Batches of N images become N `HistoryEntry` records
/// sharing a `batchId` but each with the seed actually used for that frame.
struct HistoryEntry: Codable, Identifiable, Hashable {
    var id: UUID
    var batchId: UUID
    var createdAt: Date
    var modelId: String
    var modelDisplayName: String
    var prompt: String
    var negativePrompt: String
    var parameters: GenerationParameters
    /// Seed actually used for this specific frame (may differ from
    /// `parameters.seed` for batch members after the first, since each
    /// frame in a batch gets a distinct seed even when the request's base
    /// seed is locked).
    var seedUsed: UInt32
    /// Filename only (not a full path) — resolved against the store's
    /// images directory, so the on-disk location can move between app
    /// versions without invalidating the index.
    var imageFilename: String
    var generationSeconds: Double?

    func asReusableRequest() -> GenerationRequest {
        var params = parameters
        params.seed = seedUsed
        params.seedLocked = true
        return GenerationRequest(
            modelId: modelId,
            prompt: prompt,
            negativePrompt: negativePrompt,
            parameters: params
        )
    }
}
