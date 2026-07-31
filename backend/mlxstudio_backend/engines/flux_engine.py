"""FLUX.1 (schnell/dev) via `mflux` (https://github.com/filipstrand/mflux).

IMPORTANT — unverified against a running install: this container has no
Apple Silicon / Metal, so `mflux` cannot actually be installed or executed
here (see ARCHITECTURE.md §6). The class/method names below
(`mflux.Flux1`, `mflux.config.Config`, `mflux.callbacks.callback.InLoopCallback`)
match `mflux`'s documented public API and its `callbacks/` module layout as
of the July 2026 research pass, but `mflux` is a fast-moving project —
**verify these imports against the installed version on first real run on
the target Mac**, and adjust if they've moved. The rest of this file
(seed bookkeeping, output paths, error wrapping) doesn't depend on that
and needs no changes if only the import paths drift.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..engine_base import GeneratedImage, GenerationParams, InferenceEngine, StepCallback
from ..models import ModelSpec


class FluxEngineError(Exception):
    pass


class FluxEngine(InferenceEngine):
    def __init__(self) -> None:
        self._flux: Any = None
        self._spec: ModelSpec | None = None

    def load(self, spec: ModelSpec, weights_dir: Path) -> None:
        try:
            from mflux import Flux1, ModelConfig
        except ImportError as exc:
            raise FluxEngineError(
                "mflux is not installed. Install it in the backend's Python "
                "environment (see backend/pyproject.toml) before generating "
                "with a FLUX model."
            ) from exc

        model_config = (
            ModelConfig.dev() if spec.family == "flux1" and "dev" in spec.id else ModelConfig.schnell()
        )
        self._flux = Flux1(
            model_config=model_config,
            local_path=str(weights_dir),
        )
        self._spec = spec

    def generate(
        self,
        params: GenerationParams,
        output_dir: Path,
        on_step: StepCallback,
    ) -> list[GeneratedImage]:
        if self._flux is None or self._spec is None:
            raise FluxEngineError("generate() called before load()")

        from mflux import Config

        try:
            from mflux.callbacks.callback_manager import CallbackManager
            from mflux.callbacks.callback import InLoopCallback

            class _ProgressCallback(InLoopCallback):
                def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:  # noqa: ANN001
                    on_step(t + 1, config.num_inference_steps)

            step_callback_supported = True
        except ImportError:
            # Falls back to no per-step progress (UI shows an
            # indeterminate spinner + elapsed time instead) — see
            # ARCHITECTURE.md §6.
            step_callback_supported = False

        output_dir.mkdir(parents=True, exist_ok=True)
        images: list[GeneratedImage] = []

        for i in range(params.batch_size):
            frame_seed = params.seed if i == 0 else params.seed + i

            config = Config(
                num_inference_steps=params.steps,
                height=params.height,
                width=params.width,
                guidance=params.guidance if self._spec.supports_guidance_scale else 1.0,
            )

            if step_callback_supported:
                CallbackManager.register_in_loop(_ProgressCallback())

            result = self._flux.generate_image(
                seed=frame_seed,
                prompt=params.prompt,
                config=config,
            )

            frame_path = output_dir / f"{int(time.time() * 1000)}-{frame_seed}.png"
            result.save(path=str(frame_path))
            images.append(GeneratedImage(path=frame_path, seed=frame_seed))

        return images

    def unload(self) -> None:
        self._flux = None
        self._spec = None
