import XCTest
@testable import MLXStudioApp

/// Unverified in this sandbox — see note in HistoryStoreTests.swift. Field
/// names/shapes here must stay in sync with
/// `backend/mlxstudio_backend/protocol.py`, exercised independently by
/// `backend/tests/test_protocol.py`.
final class BackendMessagesTests: XCTestCase {
    func testGenerateCommandEncodesExpectedFields() throws {
        let request = GenerationRequest(
            modelId: "flux1-schnell-4bit",
            prompt: "a cat",
            negativePrompt: "blurry",
            parameters: GenerationParameters(
                steps: 4, guidanceScale: 1.0, width: 1024, height: 1024,
                batchSize: 2, seed: 42, seedLocked: true
            )
        )
        let command = BackendCommand.generate(id: request.id, modelId: request.modelId, request: request, outputDirectory: "/tmp/out")
        let data = try command.encodedLine()
        XCTAssertEqual(data.last, 0x0A)

        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["type"] as? String, "generate")
        XCTAssertEqual(json["model"] as? String, "flux1-schnell-4bit")
        XCTAssertEqual(json["prompt"] as? String, "a cat")
        XCTAssertEqual(json["negative_prompt"] as? String, "blurry")
        XCTAssertEqual(json["steps"] as? Int, 4)
        XCTAssertEqual(json["batch_size"] as? Int, 2)
        XCTAssertEqual(json["seed"] as? Int, 42)
        XCTAssertEqual(json["output_dir"] as? String, "/tmp/out")
    }

    func testCancelCommandEncodesIdOnly() throws {
        let id = UUID()
        let data = try BackendCommand.cancel(id: id).encodedLine()
        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["type"] as? String, "cancel")
        XCTAssertEqual(json["id"] as? String, id.uuidString)
    }

    func testDecodesProgressEvent() throws {
        let id = UUID()
        let line = #"{"type":"progress","id":"\#(id.uuidString)","step":3,"total_steps":10,"elapsed_s":1.5}"#
        let event = try JSONDecoder().decode(BackendEvent.self, from: Data(line.utf8))
        guard case let .progress(eventId, step, total, elapsed) = event else {
            return XCTFail("expected .progress, got \(event)")
        }
        XCTAssertEqual(eventId, id)
        XCTAssertEqual(step, 3)
        XCTAssertEqual(total, 10)
        XCTAssertEqual(elapsed, 1.5)
    }

    func testDecodesResultEventWithImages() throws {
        let id = UUID()
        let line = #"""
        {"type":"result","id":"\#(id.uuidString)","images":[{"path":"/a.png","seed":7}],"generation_seconds":12.3}
        """#
        let event = try JSONDecoder().decode(BackendEvent.self, from: Data(line.utf8))
        guard case let .result(eventId, images, seconds) = event else {
            return XCTFail("expected .result, got \(event)")
        }
        XCTAssertEqual(eventId, id)
        XCTAssertEqual(images, [BackendEvent.ImageResult(path: "/a.png", seed: 7)])
        XCTAssertEqual(seconds, 12.3, accuracy: 0.0001)
    }

    func testDecodesErrorEventWithoutId() throws {
        let line = #"{"type":"error","code":"bad_command","message":"nope"}"#
        let event = try JSONDecoder().decode(BackendEvent.self, from: Data(line.utf8))
        guard case let .error(id, code, message) = event else {
            return XCTFail("expected .error, got \(event)")
        }
        XCTAssertNil(id)
        XCTAssertEqual(code, "bad_command")
        XCTAssertEqual(message, "nope")
    }

    func testDecodesReadyEvent() throws {
        let event = try JSONDecoder().decode(BackendEvent.self, from: Data(#"{"type":"ready"}"#.utf8))
        guard case .ready = event else { return XCTFail("expected .ready, got \(event)") }
    }

    func testUnknownTypeDecodesToUnknownCaseRatherThanThrowing() throws {
        let event = try JSONDecoder().decode(BackendEvent.self, from: Data(#"{"type":"future_thing"}"#.utf8))
        guard case let .unknown(type) = event else { return XCTFail("expected .unknown, got \(event)") }
        XCTAssertEqual(type, "future_thing")
    }
}
