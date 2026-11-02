"""Reports whether MLX can use the M5 GPU's Neural Accelerators, so the
backend log shows it rather than us assuming it.

Mirrors MLX's own rule (`is_nax_available` in mlx/backend/metal/device.cpp,
v0.30+): macOS 26.2 or later, and a GPU architecture generation of at least
17 (18 for phone-class GPUs). The generation is the two digits before the
architecture name's final letter, e.g. "applegpu_g17s" -> 17, class "s".
Only MLX's macOS 26 build contains those kernels, so the installed
mlx-metal wheel's platform tag is checked too.
"""
from __future__ import annotations

import importlib.metadata
import platform


def architecture_generation(architecture: str) -> tuple[int, str] | None:
    if len(architecture) < 3 or not architecture[-3:-1].isdigit():
        return None
    return int(architecture[-3:-1]), architecture[-1]


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def neural_accelerator_status(
    mlx_version: str,
    macos_version: str,
    architecture: str | None,
    metal_wheel_tag: str | None,
) -> tuple[bool, str]:
    """Returns (active, explanation)."""
    reasons = []
    if _version_tuple(mlx_version) < (0, 30):
        reasons.append(f"MLX {mlx_version} predates Neural Accelerator support (0.30)")
    if not macos_version or _version_tuple(macos_version) < (26, 2):
        reasons.append(f"macOS {macos_version or 'unknown'} is older than 26.2")
    generation = architecture_generation(architecture) if architecture else None
    if generation is None:
        reasons.append(f"unrecognized GPU architecture {architecture!r}")
    else:
        gen, gpu_class = generation
        required = 18 if gpu_class == "p" else 17
        if gen < required:
            reasons.append(f"GPU generation {gen} has no Neural Accelerators (needs {required}+, i.e. M5)")
    if metal_wheel_tag is not None and "macosx_26" not in metal_wheel_tag:
        reasons.append(f"installed MLX build ({metal_wheel_tag}) is not the macOS 26 build")
    if reasons:
        return False, "; ".join(reasons)
    return True, "macOS, GPU and MLX build all qualify"


def _metal_wheel_tag() -> str | None:
    try:
        wheel = importlib.metadata.distribution("mlx-metal").read_text("WHEEL") or ""
    except importlib.metadata.PackageNotFoundError:
        return None
    tags = [line.split(":", 1)[1].strip() for line in wheel.splitlines() if line.startswith("Tag:")]
    return tags[0] if tags else None


def runtime_report() -> str:
    try:
        import mlx.core as mx
    except ImportError:
        return "MLX is not installed"

    info_fn = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
    try:
        info = info_fn() if info_fn else {}
    except Exception:  # noqa: BLE001 - diagnostics must never stop the backend
        info = {}
    architecture = info.get("architecture")
    macos = platform.mac_ver()[0]
    active, why = neural_accelerator_status(mx.__version__, macos, architecture, _metal_wheel_tag())
    return (
        f"MLX {mx.__version__}, macOS {macos or 'unknown'}, "
        f"GPU {info.get('device_name', '?')} ({architecture or '?'}); "
        f"M5 Neural Accelerators: {'ACTIVE' if active else 'inactive'} ({why})"
    )
