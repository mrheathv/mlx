"""Server-side memory sanity check.

The Swift app already warns the user before it ever sends a `generate`
command (see `MemoryGuard.swift`) — this is a second, defense-in-depth
check on the backend side so a request sent while the UI's estimate is
stale (or a future non-Swift client) still gets a clean error instead of
an OS-level kill or Metal OOM abort partway through a denoising loop.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import psutil
except ImportError:  # pragma: no cover - exercised via the fallback path in tests
    psutil = None

RESERVED_SYSTEM_GB = 4.0


@dataclass(frozen=True)
class MemoryAssessment:
    is_likely_to_fit: bool
    estimated_required_gb: float
    estimated_available_gb: float


def available_memory_gb() -> float:
    if psutil is None:
        # No way to introspect memory pressure — assume it fits and let
        # the OS/MLX be the backstop, rather than block generation on a
        # machine where this optional dependency isn't installed.
        return float("inf")
    vm = psutil.virtual_memory()
    return (vm.available) / (1024 ** 3)


def assess(approx_runtime_ram_gb: float) -> MemoryAssessment:
    available = available_memory_gb()
    required = approx_runtime_ram_gb + RESERVED_SYSTEM_GB
    return MemoryAssessment(
        is_likely_to_fit=available >= required,
        estimated_required_gb=required,
        estimated_available_gb=available,
    )
