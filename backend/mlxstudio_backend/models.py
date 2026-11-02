"""The model registry: authoritative source of which checkpoints exist and
how to run them. Swift's `Resources/models.json` mirrors it for the UI
(`tests/test_models_registry.py` fails if the two drift apart).

Every checkpoint is a pre-quantized mflux-format repo. mflux applies stored
weights with `strict=False`, so a checkpoint whose parameter names don't
match the installed mflux would load without error and generate noise.
Two defenses (ARCHITECTURE.md §2): the module layouts of every mflux
version that could have saved these checkpoints were diffed against the
pinned 0.24.0 and found identical, and `checkpoint_check` re-verifies the
actual files at load time.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    tier: str  # "fast" | "quality" | "alternate"
    engine: str  # "mflux"
    family: str  # "flux2_klein" | "z_image"
    # Name of the mflux ModelConfig factory (`ModelConfig.<name>()`). Saved
    # checkpoints don't record which variant they are, so it's declared here.
    mflux_config: str
    hf_repo: str
    approx_download_gb: float
    approx_runtime_ram_gb: float
    supports_guidance_scale: bool
    supports_negative_prompt: bool
    default_steps: int
    max_recommended_steps: int
    default_guidance: float
    default_width: int
    default_height: int


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id="flux2-klein-4b-4bit",
            display_name="FLUX.2 [klein] 4B — Fast",
            tier="fast",
            engine="mflux",
            family="flux2_klein",
            mflux_config="flux2_klein_4b",
            hf_repo="Runpod/FLUX.2-klein-4B-mflux-4bit",
            approx_download_gb=4.8,
            approx_runtime_ram_gb=8.0,
            supports_guidance_scale=False,
            supports_negative_prompt=False,
            default_steps=4,
            max_recommended_steps=8,
            default_guidance=1.0,
            default_width=1024,
            default_height=1024,
        ),
        ModelSpec(
            id="flux2-klein-9b-4bit",
            display_name="FLUX.2 [klein] 9B — Quality",
            tier="quality",
            engine="mflux",
            family="flux2_klein",
            mflux_config="flux2_klein_9b",
            hf_repo="mlx-community/flux2-klein-9b-4bit",
            approx_download_gb=10.0,
            approx_runtime_ram_gb=13.0,
            supports_guidance_scale=False,
            supports_negative_prompt=False,
            default_steps=4,
            max_recommended_steps=8,
            default_guidance=1.0,
            default_width=1024,
            default_height=1024,
        ),
        ModelSpec(
            id="z-image-turbo-4bit",
            display_name="Z-Image Turbo — Alt",
            tier="alternate",
            engine="mflux",
            family="z_image",
            mflux_config="z_image_turbo",
            hf_repo="filipstrand/Z-Image-Turbo-mflux-4bit",
            approx_download_gb=5.9,
            approx_runtime_ram_gb=9.0,
            supports_guidance_scale=False,
            supports_negative_prompt=False,
            default_steps=9,
            max_recommended_steps=16,
            default_guidance=0.0,
            default_width=1024,
            default_height=1024,
        ),
    ]
}


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise UnknownModelError(model_id) from exc


class UnknownModelError(Exception):
    def __init__(self, model_id: str):
        super().__init__(f"Unknown model id: {model_id!r}")
        self.model_id = model_id
