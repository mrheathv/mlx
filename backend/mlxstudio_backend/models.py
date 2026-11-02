"""The model registry: authoritative source of which checkpoints exist and
how to run them. Swift's `Resources/models.json` is a hand-synced mirror of
this for UI display before the backend is even running (see
ARCHITECTURE.md §2/§6 — auto-generating that file from this one is flagged
as tech debt, not done yet).

Every checkpoint here is a pre-quantized mflux-format repo whose parameter
layout was verified against the pinned mflux 0.18.0 (ARCHITECTURE.md §2
describes how). That check matters because mflux applies stored weights
with `strict=False`: a checkpoint saved by an mflux version with a
different module layout would load without error and generate noise.
Don't swap in a different repo without re-running that check.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    tier: str  # "fast" | "quality" | "alternate"
    engine: str  # "mflux"
    family: str  # "flux1" | "z_image"
    hf_repo: str
    approx_download_gb: float
    approx_runtime_ram_gb: float
    supports_guidance_scale: bool
    default_steps: int
    max_recommended_steps: int
    default_guidance: float
    default_width: int
    default_height: int
    # mflux ModelConfig to pair with the weights ("schnell"/"dev"); the
    # saved checkpoint doesn't record which FLUX.1 variant it is.
    mflux_base: str | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id="flux1-schnell-4bit",
            display_name="FLUX.1 [schnell] — Fast",
            tier="fast",
            engine="mflux",
            family="flux1",
            hf_repo="dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit",
            approx_download_gb=9.5,
            approx_runtime_ram_gb=13.0,
            supports_guidance_scale=False,
            default_steps=4,
            max_recommended_steps=8,
            default_guidance=1.0,
            default_width=1024,
            default_height=1024,
            mflux_base="schnell",
        ),
        ModelSpec(
            id="flux1-dev-4bit",
            display_name="FLUX.1 [dev] — Quality",
            tier="quality",
            engine="mflux",
            family="flux1",
            hf_repo="dhairyashil/FLUX.1-dev-mflux-4bit",
            approx_download_gb=9.5,
            approx_runtime_ram_gb=13.5,
            supports_guidance_scale=True,
            default_steps=25,
            max_recommended_steps=40,
            default_guidance=3.5,
            default_width=1024,
            default_height=1024,
            mflux_base="dev",
        ),
        ModelSpec(
            id="z-image-turbo-4bit",
            display_name="Z-Image Turbo — Alt",
            tier="alternate",
            engine="mflux",
            family="z_image",
            hf_repo="filipstrand/Z-Image-Turbo-mflux-4bit",
            approx_download_gb=5.9,
            approx_runtime_ram_gb=9.0,
            supports_guidance_scale=False,
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
