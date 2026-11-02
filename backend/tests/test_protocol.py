import json

import pytest

from mlxstudio_backend import protocol


def test_parse_generate_command_round_trips_all_fields():
    line = json.dumps({
        "type": "generate", "id": "abc-123", "model": "flux2-klein-4b-4bit",
        "prompt": "a cat", "negative_prompt": "", "steps": 4, "guidance": 1.0,
        "seed": 42, "width": 1024, "height": 1024, "batch_size": 1,
        "output_dir": "/tmp/out",
    })
    command = protocol.parse_command(line)
    assert isinstance(command, protocol.GenerateCommand)
    assert command.id == "abc-123"
    assert command.model == "flux2-klein-4b-4bit"
    assert command.steps == 4
    assert command.seed == 42


def test_parse_download_command():
    line = json.dumps({"type": "download", "id": "x", "model": "z-image-turbo-4bit"})
    command = protocol.parse_command(line)
    assert command == protocol.DownloadCommand(id="x", model="z-image-turbo-4bit")


def test_parse_cancel_command():
    command = protocol.parse_command(json.dumps({"type": "cancel", "id": "req-1"}))
    assert command == protocol.CancelCommand(id="req-1")


def test_parse_list_models_command():
    command = protocol.parse_command(json.dumps({"type": "list_models", "id": "req-2"}))
    assert command == protocol.ListModelsCommand(id="req-2")


def test_parse_rejects_empty_line():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_command("   ")


def test_parse_rejects_invalid_json():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_command("{not json")


def test_parse_rejects_unknown_type():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_command(json.dumps({"type": "nonsense"}))


def test_parse_generate_missing_field_raises():
    payload = {
        "type": "generate", "id": "abc", "model": "m", "prompt": "p",
        "negative_prompt": "", "steps": 1, "guidance": 1.0, "seed": 1,
        "width": 512, "height": 512,
        # batch_size and output_dir intentionally missing
    }
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_command(json.dumps(payload))


def test_progress_event_shape():
    line = protocol.progress_event("id-1", 3, 10, 1.5)
    payload = json.loads(line)
    assert payload == {"type": "progress", "id": "id-1", "step": 3, "total_steps": 10, "elapsed_s": 1.5}


def test_result_event_shape():
    line = protocol.result_event("id-1", [{"path": "/a.png", "seed": 7}], 12.3)
    payload = json.loads(line)
    assert payload["type"] == "result"
    assert payload["images"] == [{"path": "/a.png", "seed": 7}]
    assert payload["generation_seconds"] == 12.3


def test_error_event_omits_id_when_absent():
    payload = json.loads(protocol.error_event("bad_command", "nope"))
    assert "id" not in payload
    assert payload["code"] == "bad_command"


def test_error_event_includes_id_when_present():
    payload = json.loads(protocol.error_event("generation_failed", "boom", "req-9"))
    assert payload["id"] == "req-9"


def test_events_are_single_line_json():
    for line in (
        protocol.ready_event(),
        protocol.progress_event("i", 1, 2, 0.1),
        protocol.model_loading_event("i"),
        protocol.download_progress_event("i", 1, 2),
        protocol.download_complete_event("i"),
        protocol.result_event("i", [], 0.0),
        protocol.model_list_event("i", ["a", "b"]),
        protocol.error_event("code", "message"),
    ):
        assert "\n" not in line
        json.loads(line)  # doesn't raise
