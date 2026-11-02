"""Entry point: `python3 -m mlxstudio_backend`.

Reads one JSON command per line from stdin, dispatches each to a worker
thread so a long-running `generate` doesn't block a subsequent `cancel`
(or a `download` for a different model) from being read and handled, and
writes JSON events to stdout as they happen. stdout writes are
lock-guarded so two threads' events can never interleave mid-line.

The real stdout is reserved for protocol events: `sys.stdout` is pointed
at stderr before anything else runs, so `print()` calls inside mflux and
its dependencies end up in the backend log instead of on the protocol
channel.
"""
from __future__ import annotations

import sys
import threading
from typing import TextIO

from . import protocol
from .server import Server

_stdout_lock = threading.Lock()


def make_emit(out: TextIO):
    def emit(line: str) -> None:
        with _stdout_lock:
            out.write(line)
            out.write("\n")
            out.flush()
    return emit


def main() -> None:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    emit = make_emit(protocol_out)
    server = Server()
    emit(protocol.ready_event())

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        try:
            command = protocol.parse_command(raw_line)
        except protocol.ProtocolError as exc:
            emit(protocol.error_event("bad_command", str(exc)))
            continue

        thread = threading.Thread(target=server.handle, args=(command, emit), daemon=True)
        thread.start()


if __name__ == "__main__":
    main()
