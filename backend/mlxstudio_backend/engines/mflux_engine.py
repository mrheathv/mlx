"""All three models via `mflux` (https://github.com/filipstrand/mflux).

Written against mflux 0.18.0's actual source (read from the published
wheel, not docs): `Flux1` lives at
`mflux.models.flux.variants.txt2img.flux`, `ZImageTurbo` at
`mflux.models.z_image`; both take `model_path=` (a local directory works)
and expose a per-instance `callbacks` registry whose in-loop subscribers
are called once per denoising step. The version is pinned in
`backend/pyproject.toml` so this stays true.

Weights are loaded pre-quantized: `quantize=None` makes mflux use the
bit width stored in the checkpoint's safetensors metadata instead of
re-quantizing on the fly.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..engine_base import GeneratedImage, GenerationParams, InferenceEngine, StepCallback
from ..models import ModelSpec


class MfluxEngineError(Exception):
    pass


class _ProgressRelay:
    """Registered once per loaded model; `on_step` is swapped per
    generation. mflux discovers subscribers by duck typing
    (`hasattr(cb, "call_in_loop")`), so no base class is needed.
    Exceptions raised from here (cancellation) propagate out of
    `generate_image`, which only catches KeyboardInterrupt.
    """

    def __init__(self) -> None:
        self.on_step: StepCallback | None = None

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:  # noqa: ANN001
        if self.on_step is not None:
            self.on_step(t + 1, config.num_inference_steps)


def _build_model(spec: ModelSpec, weights_dir: Path) -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    if spec.family == "flux1":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        config = ModelConfig.dev() if spec.mflux_base == "dev" else ModelConfig.schnell()
        return Flux1(model_config=config, model_path=str(weights_dir))

    if spec.family == "z_image":
        from mflux.models.z_image import ZImageTurbo

        return ZImageTurbo(model_config=ModelConfig.z_image_turbo(), model_path=str(weights_dir))

    raise MfluxEngineError(f"No mflux model class for family {spec.family!r}")


class MfluxEngine(InferenceEngine):
    def __init__(self) -> None:
        self._model: Any = None
        self._spec: ModelSpec | None = None
        self._progress = _ProgressRelay()

    def load(self, spec: ModelSpec, weights_dir: Path) -> None:
        try:
            import mflux  # noqa: F401
        except ImportError as exc:
            raise MfluxEngineError(
                "mflux is not installed in the backend's Python environment."
            ) from exc

        model = _build_model(spec, weights_dir)
        model.callbacks.register(self._progress)
        self._model = model
        self._spec = spec

    def generate(
        self,
        params: GenerationParams,
        output_dir: Path,
        on_step: StepCallback,
    ) -> list[GeneratedImage]:
        if self._model is None or self._spec is None:
            raise MfluxEngineError("generate() called before load()")

        output_dir.mkdir(parents=True, exist_ok=True)
        guidance = params.guidance if self._spec.supports_guidance_scale else None
        images: list[GeneratedImage] = []

        self._progress.on_step = on_step
        try:
            for i in range(params.batch_size):
                frame_seed = (params.seed + i) % (2**32)
                kwargs: dict[str, Any] = {
                    "seed": frame_seed,
                    "prompt": params.prompt,
                    "num_inference_steps": params.steps,
                    "width": params.width,
                    "height": params.height,
                    "negative_prompt": params.negative_prompt or None,
                }
                if guidance is not None:
                    kwargs["guidance"] = guidance

                result = self._model.generate_image(**kwargs)

                frame_path = output_dir / f"{int(time.time() * 1000)}-{frame_seed}.png"
                result.save(path=str(frame_path), overwrite=True)
                images.append(GeneratedImage(path=frame_path, seed=frame_seed))
        finally:
            self._progress.on_step = None

        return images

    def unload(self) -> None:
        self._model = None
        self._spec = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001 - best effort; freeing happens on GC regardless
            pass
