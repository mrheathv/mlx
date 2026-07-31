import Foundation

/// Everything needed to reproduce a generation: what gets sent to the
/// backend, and what gets persisted alongside the resulting image(s).
struct GenerationRequest: Codable, Hashable, Identifiable {
    var id: UUID
    var modelId: String
    var prompt: String
    var negativePrompt: String
    var parameters: GenerationParameters

    init(
        id: UUID = UUID(),
        modelId: String,
        prompt: String,
        negativePrompt: String = "",
        parameters: GenerationParameters
    ) {
        self.id = id
        self.modelId = modelId
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.parameters = parameters
    }
}
