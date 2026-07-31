"""Newline-delimited JSON wire protocol shared with
`Sources/MLXStudioApp/Models/BackendMessages.swift`. Field names here must
match that file exactly — there's no schema shared between the two
languages beyond "keep these in sync by hand," which is a known v1
limitation (see ARCHITECTURE.md §6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ProtocolError(Exception):
    """A line from stdin was not a well-formed command."""


@dataclass(frozen=True)
class GenerateCommand:
    id: str
    model: str
    prompt: str
    negative_prompt: str
    steps: int
    guidance: float
    seed: int
    width: int
    height: int
    batch_size: int
    output_dir: str


@dataclass(frozen=True)
class DownloadCommand:
    id: str
    model: str


@dataclass(frozen=True)
class CancelCommand:
    id: str


@dataclass(frozen=True)
class ListModelsCommand:
    id: str


Command = GenerateCommand | DownloadCommand | CancelCommand | ListModelsCommand

_REQUIRED_GENERATE_FIELDS = (
    "id", "model", "prompt", "negative_prompt", "steps", "guidance",
    "seed", "width", "height", "batch_size", "output_dir",
)


def parse_command(line: str) -> Command:
    line = line.strip()
    if not line:
        raise ProtocolError("empty line")
    try:
        payload: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc

    command_type = payload.get("type")
    try:
        if command_type == "generate":
            missing = [f for f in _REQUIRED_GENERATE_FIELDS if f not in payload]
            if missing:
                raise ProtocolError(f"generate command missing fields: {missing}")
            return GenerateCommand(**{f: payload[f] for f in _REQUIRED_GENERATE_FIELDS})
        if command_type == "download":
            return DownloadCommand(id=payload["id"], model=payload["model"])
        if command_type == "cancel":
            return CancelCommand(id=payload["id"])
        if command_type == "list_models":
            return ListModelsCommand(id=payload["id"])
    except KeyError as exc:
        raise ProtocolError(f"missing field: {exc}") from exc

    raise ProtocolError(f"unknown command type: {command_type!r}")


# --- Outgoing events -------------------------------------------------

def ready_event() -> str:
    return _encode({"type": "ready"})


def progress_event(request_id: str, step: int, total_steps: int, elapsed_s: float) -> str:
    return _encode({
        "type": "progress", "id": request_id, "step": step,
        "total_steps": total_steps, "elapsed_s": elapsed_s,
    })


def model_loading_event(request_id: str) -> str:
    return _encode({"type": "model_loading", "id": request_id})


def download_progress_event(request_id: str, bytes_done: int, bytes_total: int) -> str:
    return _encode({
        "type": "download_progress", "id": request_id,
        "bytes_done": bytes_done, "bytes_total": bytes_total,
    })


def download_complete_event(request_id: str) -> str:
    return _encode({"type": "download_complete", "id": request_id})


def result_event(request_id: str, images: list[dict[str, Any]], generation_seconds: float) -> str:
    return _encode({
        "type": "result", "id": request_id, "images": images,
        "generation_seconds": generation_seconds,
    })


def model_list_event(request_id: str, model_ids: list[str]) -> str:
    return _encode({"type": "model_list", "id": request_id, "model_ids": model_ids})


def error_event(code: str, message: str, request_id: str | None = None) -> str:
    payload: dict[str, Any] = {"type": "error", "code": code, "message": message}
    if request_id is not None:
        payload["id"] = request_id
    return _encode(payload)


def _encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))
