"""Interface every inference engine implements, so `server.py` only
routes by `ModelSpec.engine` and owns load/unload/cancellation. Today the
only engine is `mflux` (engines/mflux_engine.py); the seam also lets the
server tests run with fake engines, since mflux needs Apple Silicon.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import ModelSpec

# (step, total_steps) -> None. Engines call this from inside their
# denoising loop when they can; it's best-effort, not guaranteed per
# ARCHITECTURE.md §6.
StepCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class GeneratedImage:
    path: Path
    seed: int


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    negative_prompt: str
    steps: int
    guidance: float
    seed: int
    width: int
    height: int
    batch_size: int


class InferenceEngine(ABC):
    """One loaded model at a time. `load`/`unload` bracket residency in
    unified memory; `generate` assumes `load` has already happened.
    """

    @abstractmethod
    def load(self, spec: ModelSpec, weights_dir: Path) -> None:
        ...

    @abstractmethod
    def generate(
        self,
        params: GenerationParams,
        output_dir: Path,
        on_step: StepCallback,
    ) -> list[GeneratedImage]:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...
