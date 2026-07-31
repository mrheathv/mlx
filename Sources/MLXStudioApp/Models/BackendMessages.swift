import Foundation

// MARK: - Outgoing (Swift -> Python backend), one JSON object per line.

/// Everything sent to the backend is `{"type": ..., "id": ..., ...}`.
/// Encoding directly to a flat struct per command (rather than one big
/// enum with a custom encoder) keeps this side simple since Swift is only
/// ever producing these, never switching on them.
enum BackendCommand {
    case generate(id: UUID, modelId: String, request: GenerationRequest, outputDirectory: String)
    case download(id: UUID, modelId: String)
    case cancel(id: UUID)
    case listModels(id: UUID)

    /// One line of newline-delimited JSON, including the trailing `\n`.
    func encodedLine() throws -> Data {
        var payload: [String: JSONValue] = [:]
        switch self {
        case let .generate(id, modelId, request, outputDirectory):
            payload["type"] = .string("generate")
            payload["id"] = .string(id.uuidString)
            payload["model"] = .string(modelId)
            payload["prompt"] = .string(request.prompt)
            payload["negative_prompt"] = .string(request.negativePrompt)
            payload["steps"] = .int(request.parameters.steps)
            payload["guidance"] = .double(request.parameters.guidanceScale)
            payload["seed"] = .int(Int(request.parameters.seed))
            payload["width"] = .int(request.parameters.width)
            payload["height"] = .int(request.parameters.height)
            payload["batch_size"] = .int(request.parameters.batchSize)
            payload["output_dir"] = .string(outputDirectory)
        case let .download(id, modelId):
            payload["type"] = .string("download")
            payload["id"] = .string(id.uuidString)
            payload["model"] = .string(modelId)
        case let .cancel(id):
            payload["type"] = .string("cancel")
            payload["id"] = .string(id.uuidString)
        case let .listModels(id):
            payload["type"] = .string("list_models")
            payload["id"] = .string(id.uuidString)
        }
        var data = try JSONEncoder().encode(payload)
        data.append(0x0A) // '\n'
        return data
    }
}

/// Minimal untyped JSON value so `BackendCommand` can build a flat
/// `[String: JSONValue]` payload without a hand-written `Encodable` per case.
enum JSONValue: Encodable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(v): try container.encode(v)
        case let .int(v): try container.encode(v)
        case let .double(v): try container.encode(v)
        case let .bool(v): try container.encode(v)
        }
    }
}

// MARK: - Incoming (Python backend -> Swift), one JSON object per line.

/// Every incoming line is tagged with a `type` field; `BackendEvent`
/// decodes that tag first and then the matching payload.
enum BackendEvent {
    struct ImageResult: Decodable, Hashable {
        let path: String
        let seed: UInt32
    }

    case ready
    case progress(id: UUID, step: Int, totalSteps: Int, elapsedSeconds: Double)
    case modelLoading(id: UUID)
    case downloadProgress(id: UUID, bytesDone: Int64, bytesTotal: Int64)
    case downloadComplete(id: UUID)
    case result(id: UUID, images: [ImageResult], generationSeconds: Double)
    case modelList(id: UUID, modelIds: [String])
    case error(id: UUID?, code: String, message: String)
    case unknown(type: String)
}

extension BackendEvent: Decodable {
    private enum CodingKeys: String, CodingKey {
        case type, id, step, totalSteps = "total_steps", elapsedSeconds = "elapsed_s"
        case bytesDone = "bytes_done", bytesTotal = "bytes_total"
        case images, generationSeconds = "generation_seconds"
        case modelIds = "model_ids"
        case code, message
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        func requireId() throws -> UUID {
            let raw = try container.decode(String.self, forKey: .id)
            guard let uuid = UUID(uuidString: raw) else {
                throw DecodingError.dataCorruptedError(forKey: .id, in: container, debugDescription: "not a UUID")
            }
            return uuid
        }

        switch type {
        case "ready":
            self = .ready
        case "progress":
            self = .progress(
                id: try requireId(),
                step: try container.decode(Int.self, forKey: .step),
                totalSteps: try container.decode(Int.self, forKey: .totalSteps),
                elapsedSeconds: try container.decode(Double.self, forKey: .elapsedSeconds)
            )
        case "model_loading":
            self = .modelLoading(id: try requireId())
        case "download_progress":
            self = .downloadProgress(
                id: try requireId(),
                bytesDone: try container.decode(Int64.self, forKey: .bytesDone),
                bytesTotal: try container.decode(Int64.self, forKey: .bytesTotal)
            )
        case "download_complete":
            self = .downloadComplete(id: try requireId())
        case "result":
            self = .result(
                id: try requireId(),
                images: try container.decode([ImageResult].self, forKey: .images),
                generationSeconds: try container.decode(Double.self, forKey: .generationSeconds)
            )
        case "model_list":
            self = .modelList(id: try requireId(), modelIds: try container.decode([String].self, forKey: .modelIds))
        case "error":
            self = .error(
                id: try container.decodeIfPresent(String.self, forKey: .id).flatMap(UUID.init(uuidString:)),
                code: try container.decode(String.self, forKey: .code),
                message: try container.decode(String.self, forKey: .message)
            )
        default:
            self = .unknown(type: type)
        }
    }
}
