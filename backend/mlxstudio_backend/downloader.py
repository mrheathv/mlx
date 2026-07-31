"""Downloads and caches model weights from Hugging Face.

Uses `huggingface_hub.snapshot_download`, which already does the right
thing for re-launch (only fetches what's missing/changed) — we don't
maintain our own "is this downloaded" bookkeeping beyond what's on disk,
matching the Swift side's `ModelManager.detectLocalState`, which also
just looks at the directory rather than trusting a persisted flag.

Byte-level progress is derived by polling the destination directory's
size on disk against the repo's total size from the Hub API, rather than
hooking `huggingface_hub`'s internal tqdm bars — the public progress-bar
hook points have changed across `huggingface_hub` versions, whereas "how
many bytes are on disk vs. how many the repo says it has" is stable and
version-agnostic. The trade-off is coarser granularity (polled, not
per-chunk) and a startup window where the total is unknown before the
first Hub API call resolves.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi, snapshot_download

ProgressCallback = Callable[[int, int], None]  # (bytes_done, bytes_total) -> None

POLL_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True)
class DownloadResult:
    local_path: Path


class DownloadError(Exception):
    pass


def total_repo_size_bytes(repo_id: str, api: HfApi | None = None) -> int:
    api = api or HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    return sum(sibling.size or 0 for sibling in info.siblings)


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def download(
    repo_id: str,
    destination: Path,
    on_progress: ProgressCallback,
    cancel_event: threading.Event | None = None,
) -> DownloadResult:
    """Downloads `repo_id` into `destination`, calling `on_progress`
    roughly every `POLL_INTERVAL_SECONDS` while the download runs.

    Raises `DownloadError` on failure. `cancel_event` is checked between
    polls only — the underlying `snapshot_download` call is not
    interruptible mid-file, so cancellation takes effect at up to one
    poll interval's delay (acceptable for a background model download,
    unlike mid-generation cancellation which needs to be tighter).
    """
    destination.mkdir(parents=True, exist_ok=True)

    try:
        total_bytes = total_repo_size_bytes(repo_id)
    except Exception:
        total_bytes = 0  # Unknown total; progress will report done=growing, total=0.

    error_holder: list[BaseException] = []
    done_event = threading.Event()

    def run_download() -> None:
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(destination),
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller, not swallowed
            error_holder.append(exc)
        finally:
            done_event.set()

    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()

    while not done_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadError("cancelled")
        on_progress(_directory_size_bytes(destination), total_bytes)
        time.sleep(POLL_INTERVAL_SECONDS)

    thread.join()
    if error_holder:
        raise DownloadError(str(error_holder[0])) from error_holder[0]

    on_progress(_directory_size_bytes(destination), total_bytes or _directory_size_bytes(destination))
    return DownloadResult(local_path=destination)
