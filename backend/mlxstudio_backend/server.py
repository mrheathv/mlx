"""Dispatches parsed commands to the right engine and emits protocol
events. One `InferenceEngine` is resident at a time (see
`engine_base.InferenceEngine` docstring and ARCHITECTURE.md §2) — selecting
a different model unloads the current one first, which the caller sees as
a `model_loading` event before the first `progress`/`result`.

Memory pressure is *not* re-blocked here: the Swift app already asks the
user to confirm before ever sending a `generate` command when its own
estimate says a model won't fit (`MemoryGuard.swift`), including an
explicit "try anyway" override. Re-blocking on the backend would make that
override a lie. `memory.assess` is still called so a real shortfall is at
least logged to stderr for debugging, not to silently gate the request.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable

from . import downloader, memory, protocol
from .engine_base import GenerationParams, InferenceEngine
from .engines.mflux_engine import MfluxEngine
from .models import UnknownModelError, get_model

Emit = Callable[[str], None]

# Written only after a download finishes, so an interrupted download isn't
# mistaken for a usable model. The Swift side (ModelManager) checks the
# same filename.
DOWNLOAD_COMPLETE_MARKER = ".mlxstudio-download-complete"


class GenerationCancelled(Exception):
    pass


def _engine_for(engine_name: str) -> InferenceEngine:
    if engine_name == "mflux":
        return MfluxEngine()
    raise ValueError(f"Unknown engine: {engine_name}")


class Server:
    def __init__(self, weights_cache_root: Path | None = None):
        self.weights_cache_root = weights_cache_root or (
            Path.home() / "Library" / "Application Support" / "MLXStudio" / "models"
        )
        self._active_engine: InferenceEngine | None = None
        self._active_model_id: str | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()  # guards engine swap; one generation at a time

    def weights_dir(self, model_id: str) -> Path:
        return self.weights_cache_root / model_id

    def handle(self, command: protocol.Command, emit: Emit) -> None:
        if isinstance(command, protocol.GenerateCommand):
            self._handle_generate(command, emit)
        elif isinstance(command, protocol.DownloadCommand):
            self._handle_download(command, emit)
        elif isinstance(command, protocol.CancelCommand):
            self._handle_cancel(command)
        elif isinstance(command, protocol.ListModelsCommand):
            from .models import MODEL_REGISTRY
            emit(protocol.model_list_event(command.id, list(MODEL_REGISTRY.keys())))

    # -- generate ---------------------------------------------------

    def _handle_generate(self, cmd: protocol.GenerateCommand, emit: Emit) -> None:
        try:
            spec = get_model(cmd.model)
        except UnknownModelError as exc:
            emit(protocol.error_event("unknown_model", str(exc), cmd.id))
            return

        weights_dir = self.weights_dir(spec.id)
        if not (weights_dir / DOWNLOAD_COMPLETE_MARKER).exists():
            emit(protocol.error_event(
                "model_not_downloaded",
                f"{spec.display_name} has not been downloaded yet.",
                cmd.id,
            ))
            return

        assessment = memory.assess(spec.approx_runtime_ram_gb)
        if not assessment.is_likely_to_fit:
            print(
                f"[mlxstudio_backend] warning: generating with {spec.id} anyway despite "
                f"estimated {assessment.estimated_available_gb:.1f}GB available < "
                f"{assessment.estimated_required_gb:.1f}GB estimated required",
                file=sys.stderr,
            )

        cancel_event = threading.Event()
        self._cancel_events[cmd.id] = cancel_event

        try:
            with self._lock:
                if self._active_model_id != spec.id:
                    emit(protocol.model_loading_event(cmd.id))
                    if self._active_engine is not None:
                        self._active_engine.unload()
                        self._active_engine = None
                        self._active_model_id = None
                    engine = _engine_for(spec.engine)
                    engine.load(spec, weights_dir)
                    self._active_engine = engine
                    self._active_model_id = spec.id

                assert self._active_engine is not None
                start = time.monotonic()

                def on_step(step: int, total_steps: int) -> None:
                    if cancel_event.is_set():
                        raise GenerationCancelled()
                    emit(protocol.progress_event(cmd.id, step, total_steps, time.monotonic() - start))

                params = GenerationParams(
                    prompt=cmd.prompt,
                    negative_prompt=cmd.negative_prompt,
                    steps=cmd.steps,
                    guidance=cmd.guidance,
                    seed=cmd.seed,
                    width=cmd.width,
                    height=cmd.height,
                    batch_size=cmd.batch_size,
                )

                images = self._active_engine.generate(params, Path(cmd.output_dir), on_step)
        except GenerationCancelled:
            emit(protocol.error_event("cancelled", "Generation was cancelled.", cmd.id))
            return
        except Exception as exc:  # noqa: BLE001 - any engine failure must reach the UI, not crash the process
            emit(protocol.error_event("generation_failed", str(exc), cmd.id))
            return
        finally:
            self._cancel_events.pop(cmd.id, None)

        elapsed = time.monotonic() - start
        emit(protocol.result_event(
            cmd.id,
            [{"path": str(img.path), "seed": img.seed} for img in images],
            elapsed,
        ))

    # -- download -----------------------------------------------------

    def _handle_download(self, cmd: protocol.DownloadCommand, emit: Emit) -> None:
        try:
            spec = get_model(cmd.model)
        except UnknownModelError as exc:
            emit(protocol.error_event("unknown_model", str(exc), cmd.id))
            return

        cancel_event = threading.Event()
        self._cancel_events[cmd.id] = cancel_event
        destination = self.weights_dir(spec.id)

        def on_progress(done: int, total: int) -> None:
            emit(protocol.download_progress_event(cmd.id, done, total))

        try:
            downloader.download(spec.hf_repo, destination, on_progress, cancel_event)
        except downloader.DownloadError as exc:
            emit(protocol.error_event("download_failed", str(exc), cmd.id))
            return
        finally:
            self._cancel_events.pop(cmd.id, None)

        (destination / DOWNLOAD_COMPLETE_MARKER).touch()
        emit(protocol.download_complete_event(cmd.id))

    # -- cancel ---------------------------------------------------------

    def _handle_cancel(self, cmd: protocol.CancelCommand) -> None:
        event = self._cancel_events.get(cmd.id)
        if event is not None:
            event.set()
