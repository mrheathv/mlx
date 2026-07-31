"""Stable Diffusion 3.5 via `diffusionkit`
(https://github.com/argmaxinc/DiffusionKit — Python/MLX side only; the
Swift side of that project was never finished and the whole repo was
archived in March 2026, see ARCHITECTURE.md §1/§6). Pinned to a specific
released version in `backend/pyproject.toml` for exactly that reason —
it won't get further upstream fixes.

IMPORTANT — unverified against a running install, same caveat as
`flux_engine.py`: this container has no Apple Silicon/Metal, so this
couldn't be exercised here. `DiffusionPipeline`'s constructor/`generate_image`
signature below matches DiffusionKit's documented SD3 usage (its README
doesn't show a distinct "SD3.5" example, only SD3 — `model_version` is the
only thing that should need to change for 3.5, since it's the same
pipeline class across SD3/SD3.5 per the model card, but **verify this on
first real run**).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..engine_base import GeneratedImage, GenerationParams, InferenceEngine, StepCallback
from ..models import ModelSpec


class SD35EngineError(Exception):
    pass


class SD35Engine(InferenceEngine):
    def __init__(self) -> None:
        self._pipeline: Any = None
        self._spec: ModelSpec | None = None

    def load(self, spec: ModelSpec, weights_dir: Path) -> None:
        try:
            from diffusionkit.mlx import DiffusionPipeline
        except ImportError as exc:
            raise SD35EngineError(
                "diffusionkit is not installed. Install it in the backend's "
                "Python environment (see backend/pyproject.toml) before "
                "generating with the Stable Diffusion 3.5 model."
            ) from exc

        self._pipeline = DiffusionPipeline(
            shift=3.0,
            use_t5=True,
            model_version=str(weights_dir),
            low_memory_mode=True,
            a16=True,
            w16=True,
        )
        self._spec = spec

    def generate(
        self,
        params: GenerationParams,
        output_dir: Path,
        on_step: StepCallback,
    ) -> list[GeneratedImage]:
        if self._pipeline is None or self._spec is None:
            raise SD35EngineError("generate() called before load()")

        # DiffusionKit's `generate_image` doesn't expose a step callback in
        # its documented public API, so per-step progress isn't available
        # for this engine — the UI falls back to an indeterminate spinner
        # + elapsed time (see ARCHITECTURE.md §6). We still emit a single
        # "step 0" tick so the UI knows generation has actually started.
        on_step(0, params.steps)

        output_dir.mkdir(parents=True, exist_ok=True)
        images: list[GeneratedImage] = []

        for i in range(params.batch_size):
            frame_seed = params.seed if i == 0 else params.seed + i

            image, _ = self._pipeline.generate_image(
                params.prompt,
                cfg_weight=params.guidance,
                num_steps=params.steps,
                latent_size=(params.height // 16, params.width // 16),
                seed=frame_seed,
                negative_text=params.negative_prompt or None,
            )

            frame_path = output_dir / f"{int(time.time() * 1000)}-{frame_seed}.png"
            image.save(str(frame_path))
            images.append(GeneratedImage(path=frame_path, seed=frame_seed))

        return images

    def unload(self) -> None:
        self._pipeline = None
        self._spec = None
