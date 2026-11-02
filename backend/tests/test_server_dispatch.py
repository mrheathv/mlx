"""Exercises `Server` end to end with fake engines/downloads standing in
for `mflux`/network access — none of which are available in
this sandbox (see ARCHITECTURE.md §6). This verifies the dispatch,
event-emission, and cancellation plumbing that *is* testable without
Apple Silicon; it does not verify real image generation.
"""
from pathlib import Path

import pytest

from mlxstudio_backend import downloader, protocol, server as server_module
from mlxstudio_backend.engine_base import GeneratedImage, GenerationParams, InferenceEngine


class FakeEngine(InferenceEngine):
    """Records calls; `generate` reports two steps then returns one image."""

    def __init__(self):
        self.loaded_with = None
        self.unloaded = False

    def load(self, spec, weights_dir):
        self.loaded_with = (spec.id, weights_dir)

    def generate(self, params: GenerationParams, output_dir: Path, on_step):
        on_step(1, 2)
        on_step(2, 2)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "fake.png"
        image_path.write_bytes(b"fake-png-bytes")
        return [GeneratedImage(path=image_path, seed=params.seed)]

    def unload(self):
        self.unloaded = True


class HangingEngine(InferenceEngine):
    """Loops calling on_step, yielding the GIL each time, so cancellation
    can be exercised without either racing ahead of the polling test
    thread or spinning long enough to make the test slow."""

    def load(self, spec, weights_dir):
        pass

    def generate(self, params, output_dir, on_step):
        import time
        for i in range(1, 10_000):
            on_step(i, 10_000)
            time.sleep(0.001)
        return []  # pragma: no cover - should be interrupted before this

    def unload(self):
        pass


@pytest.fixture
def collected_events():
    events = []

    def emit(line: str) -> None:
        events.append(protocol_decode(line))

    return events, emit


def protocol_decode(line: str) -> dict:
    import json
    return json.loads(line)


def make_server(tmp_path, monkeypatch, engine):
    srv = server_module.Server(weights_cache_root=tmp_path / "models")
    weights_dir = srv.weights_dir("flux1-schnell-4bit")
    weights_dir.mkdir(parents=True)
    (weights_dir / "config.json").write_text("{}")
    (weights_dir / server_module.DOWNLOAD_COMPLETE_MARKER).touch()
    monkeypatch.setattr(server_module, "_engine_for", lambda name: engine)
    return srv


def test_generate_emits_loading_progress_and_result(tmp_path, monkeypatch, collected_events):
    events, emit = collected_events
    engine = FakeEngine()
    srv = make_server(tmp_path, monkeypatch, engine)

    cmd = protocol.GenerateCommand(
        id="req-1", model="flux1-schnell-4bit", prompt="a cat", negative_prompt="",
        steps=2, guidance=1.0, seed=7, width=64, height=64, batch_size=1,
        output_dir=str(tmp_path / "out"),
    )
    srv.handle(cmd, emit)

    types = [e["type"] for e in events]
    assert types == ["model_loading", "progress", "progress", "result"]
    assert events[-1]["images"] == [{"path": str(tmp_path / "out" / "fake.png"), "seed": 7}]
    assert engine.loaded_with[0] == "flux1-schnell-4bit"


def test_second_generate_with_same_model_skips_reload(tmp_path, monkeypatch, collected_events):
    events, emit = collected_events
    engine = FakeEngine()
    srv = make_server(tmp_path, monkeypatch, engine)
    cmd = protocol.GenerateCommand(
        id="req-1", model="flux1-schnell-4bit", prompt="a cat", negative_prompt="",
        steps=2, guidance=1.0, seed=7, width=64, height=64, batch_size=1,
        output_dir=str(tmp_path / "out"),
    )
    srv.handle(cmd, emit)
    events.clear()

    cmd2 = protocol.GenerateCommand(
        id="req-2", model="flux1-schnell-4bit", prompt="another cat", negative_prompt="",
        steps=2, guidance=1.0, seed=8, width=64, height=64, batch_size=1,
        output_dir=str(tmp_path / "out"),
    )
    srv.handle(cmd2, emit)

    types = [e["type"] for e in events]
    assert "model_loading" not in types  # already-active model, no reload
    assert types[-1] == "result"


def test_generate_unknown_model_emits_error(tmp_path, monkeypatch, collected_events):
    events, emit = collected_events
    srv = server_module.Server(weights_cache_root=tmp_path / "models")
    cmd = protocol.GenerateCommand(
        id="req-1", model="does-not-exist", prompt="x", negative_prompt="",
        steps=1, guidance=1.0, seed=1, width=64, height=64, batch_size=1,
        output_dir=str(tmp_path / "out"),
    )
    srv.handle(cmd, emit)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "unknown_model"
    assert events[0]["id"] == "req-1"


def test_generate_not_downloaded_emits_error(tmp_path, monkeypatch, collected_events):
    events, emit = collected_events
    srv = server_module.Server(weights_cache_root=tmp_path / "models")  # weights dir never created
    cmd = protocol.GenerateCommand(
        id="req-1", model="flux1-schnell-4bit", prompt="x", negative_prompt="",
        steps=1, guidance=1.0, seed=1, width=64, height=64, batch_size=1,
        output_dir=str(tmp_path / "out"),
    )
    srv.handle(cmd, emit)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "model_not_downloaded"


def test_cancel_stops_generation_before_completion(tmp_path, monkeypatch, collected_events):
    events, emit = collected_events
    engine = HangingEngine()
    srv = make_server(tmp_path, monkeypatch, engine)
    cmd = protocol.GenerateCommand(
        id="req-1", model="flux1-schnell-4bit", prompt="x", negative_prompt="",
        steps=1, guidance=1.0, seed=1, width=64, height=64, batch_size=1,
        output_dir=str(tmp_path / "out"),
    )

    import threading
    gen_thread = threading.Thread(target=srv.handle, args=(cmd, emit))
    gen_thread.start()

    # Wait for at least one progress event, then cancel.
    import time
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if any(e["type"] == "progress" for e in events):
            break
        time.sleep(0.001)
    srv._handle_cancel(protocol.CancelCommand(id="req-1"))
    gen_thread.join(timeout=5)

    assert not gen_thread.is_alive()
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "cancelled"


def test_download_success_writes_completion_marker(tmp_path, monkeypatch, collected_events):
    events, emit = collected_events
    srv = server_module.Server(weights_cache_root=tmp_path / "models")

    calls = []

    def fake_download(repo_id, destination, on_progress, cancel_event):
        calls.append(repo_id)
        destination.mkdir(parents=True, exist_ok=True)
        on_progress(50, 100)
        on_progress(100, 100)
        return downloader.DownloadResult(local_path=destination)

    monkeypatch.setattr(downloader, "download", fake_download)

    srv.handle(protocol.DownloadCommand(id="dl-1", model="flux1-schnell-4bit"), emit)

    assert calls == ["dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit"]
    assert [e["type"] for e in events] == ["download_progress", "download_progress", "download_complete"]
    assert (srv.weights_dir("flux1-schnell-4bit") / server_module.DOWNLOAD_COMPLETE_MARKER).exists()


def test_download_failure_emits_error_and_leaves_no_marker(tmp_path, monkeypatch, collected_events):
    events, emit = collected_events
    srv = server_module.Server(weights_cache_root=tmp_path / "models")

    def fails_midway(repo_id, destination, on_progress, cancel_event):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "partial.safetensors").write_bytes(b"x")
        raise downloader.DownloadError(f"{repo_id} unreachable")

    monkeypatch.setattr(downloader, "download", fails_midway)

    srv.handle(protocol.DownloadCommand(id="dl-1", model="z-image-turbo-4bit"), emit)

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "download_failed"
    assert not (srv.weights_dir("z-image-turbo-4bit") / server_module.DOWNLOAD_COMPLETE_MARKER).exists()


def test_partial_download_is_not_treated_as_downloaded(tmp_path, monkeypatch, collected_events):
    events, emit = collected_events
    srv = server_module.Server(weights_cache_root=tmp_path / "models")
    weights_dir = srv.weights_dir("flux1-schnell-4bit")
    weights_dir.mkdir(parents=True)
    (weights_dir / "partial.safetensors").write_bytes(b"x")  # files present, no marker

    cmd = protocol.GenerateCommand(
        id="req-1", model="flux1-schnell-4bit", prompt="x", negative_prompt="",
        steps=1, guidance=1.0, seed=1, width=64, height=64, batch_size=1,
        output_dir=str(tmp_path / "out"),
    )
    srv.handle(cmd, emit)

    assert [e["code"] for e in events] == ["model_not_downloaded"]
