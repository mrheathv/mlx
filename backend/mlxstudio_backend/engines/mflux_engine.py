"""All models run through `mflux` (https://github.com/mflux-community/mflux).

Written against mflux 0.24.0's actual source (read from the published
wheel), and pinned to it in `backend/pyproject.toml`:
- `Flux2Klein` (mflux.models.flux2) and `ZImageTurbo` (mflux.models.z_image)
  take `model_config=` and `model_path=` (a local directory works).
- `generate_image` keyword arguments differ per family: FLUX.2 klein has
  no `negative_prompt` parameter at all, so only what a model's registry
  entry says it supports is passed.
- Each model has a `callbacks` registry; subscribers with `call_in_loop`
  are called once per denoising step.

Weights are loaded pre-quantized: `quantize=None` makes mflux use the bit
width stored in the checkpoint's safetensors metadata.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from .. import checkpoint_check
from ..engine_base import GeneratedImage, GenerationParams, InferenceEngine, StepCallback
from ..models import ModelSpec

# Component subdirectories of a saved mflux checkpoint, which are also the
# attribute names of the matching modules on both model classes.
_COMPONENTS = ("vae", "transformer", "text_encoder")


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

    config = getattr(ModelConfig, spec.mflux_config)()
    if spec.family == "flux2_klein":
        from mflux.models.flux2 import Flux2Klein

        return Flux2Klein(model_config=config, model_path=str(weights_dir))
    if spec.family == "z_image":
        from mflux.models.z_image import ZImageTurbo

        return ZImageTurbo(model_config=config, model_path=str(weights_dir))
    raise MfluxEngineError(f"No mflux model class for family {spec.family!r}")


def _verify_checkpoint(model: Any, spec: ModelSpec, weights_dir: Path) -> None:
    from mlx.utils import tree_flatten

    expected = {
        name: {key for key, _ in tree_flatten(getattr(model, name).parameters())}
        for name in _COMPONENTS
    }
    missing = checkpoint_check.missing_tensors(weights_dir, expected)
    if missing:
        raise MfluxEngineError(
            f"The downloaded {spec.display_name} checkpoint ({spec.hf_repo}) doesn't match the "
            f"model layout of the installed mflux, so it would generate noise. "
            + checkpoint_check.describe(missing, expected)
        )


class MfluxEngine(InferenceEngine):
    def __init__(self) -> None:
        self._model: Any = None
        self._spec: ModelSpec | None = None
        self._progress = _ProgressRelay()

    def load(self, spec: ModelSpec, weights_dir: Path) -> None:
        try:
            import mflux  # noqa: F401
        except ImportError as exc:
            raise MfluxEngineError("mflux is not installed in the backend's Python environment.") from exc

        model = _build_model(spec, weights_dir)
        _verify_checkpoint(model, spec, weights_dir)
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
                }
                if self._spec.supports_guidance_scale:
                    kwargs["guidance"] = params.guidance
                if self._spec.supports_negative_prompt and params.negative_prompt:
                    kwargs["negative_prompt"] = params.negative_prompt

                result = self._model.generate_image(**kwargs)

                frame_path = output_dir / f"{int(time.time() * 1000)}-{frame_seed}.png"
                result.save(path=str(frame_path), overwrite=True)
                images.append(GeneratedImage(path=frame_path, seed=frame_seed))
        finally:
            self._progress.on_step = None

        return images

    def unload(self) -> None:
        # mflux's own guidance for switching models: drop every reference,
        # collect, then clear MLX's buffer cache, so two models' weights are
        # never resident at once.
        self._model = None
        self._spec = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001 - best effort; memory is freed on GC regardless
            pass
