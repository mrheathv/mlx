"""Load-time guard against checkpoints that would silently generate noise.

mflux applies stored weights with `strict=False`: tensors whose names don't
match a model parameter are skipped without an error, and the unmatched
parameters keep their random initialization. A checkpoint saved with a
different module layout (another mflux version, another tool) therefore
"loads" fine and produces garbage. This compares the parameter names the
model expects with the tensor names in the checkpoint files, reading only
each safetensors file's JSON header (no weights).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path


def tensor_names(component_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(component_dir.glob("*.safetensors")):
        if path.name.startswith("._"):
            continue
        with path.open("rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        names.update(key for key in header if key != "__metadata__")
    return names


def missing_tensors(weights_dir: Path, expected: dict[str, set[str]]) -> dict[str, list[str]]:
    """`expected` maps a component subdirectory (e.g. "transformer") to the
    parameter names the model has for it. Returns, per component, the
    expected names absent from the checkpoint (empty dict if all present).
    Extra checkpoint tensors are fine; mflux ignores them."""
    missing: dict[str, list[str]] = {}
    for component, names in expected.items():
        absent = sorted(names - tensor_names(weights_dir / component))
        if absent:
            missing[component] = absent
    return missing


def describe(missing: dict[str, list[str]], expected: dict[str, set[str]]) -> str:
    parts = [
        f"{component}: {len(names)} of {len(expected[component])} parameters missing (e.g. {names[0]})"
        for component, names in missing.items()
    ]
    return "; ".join(parts)
