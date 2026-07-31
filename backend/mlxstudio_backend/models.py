"""The model registry: authoritative source of which checkpoints exist and
how to run them. Swift's `Resources/models.json` is a hand-synced mirror of
this for UI display before the backend is even running (see
ARCHITECTURE.md §2/§6 — auto-generating that file from this one is flagged
as v1 tech debt, not done yet).

Every checkpoint listed here was verified to exist on Hugging Face as of
the July 2026 research pass described in ARCHITECTURE.md §2 — this is a
fast-moving space, so treat this file, not prior assumptions, as the
source of truth, and expect to update repo ids as better-vetted mirrors
appear.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    tier: str  # "fast" | "quality" | "alternate"
    engine: str  # "mflux" | "diffusionkit"
    family: str  # "flux1" | "sd35"
    hf_repo: str
    fallback_hf_repo: str | None
    approx_download_gb: float
    approx_runtime_ram_gb: float
    supports_guidance_scale: bool
    default_steps: int
    max_recommended_steps: int
    default_guidance: float
    default_width: int
    default_height: int


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id="flux1-schnell-4bit",
            display_name="FLUX.1 [schnell] — Fast",
            tier="fast",
            engine="mflux",
            family="flux1",
            hf_repo="madroid/flux.1-schnell-mflux-4bit",
            fallback_hf_repo="dhairyashil/FLUX.1-schnell-mflux-4bit",
            approx_download_gb=7.0,
            approx_runtime_ram_gb=11.0,
            supports_guidance_scale=False,
            default_steps=4,
            max_recommended_steps=8,
            default_guidance=1.0,
            default_width=1024,
            default_height=1024,
        ),
        ModelSpec(
            id="flux1-dev-4bit",
            display_name="FLUX.1 [dev] — Quality",
            tier="quality",
            engine="mflux",
            family="flux1",
            hf_repo="madroid/flux.1-dev-mflux-4bit",
            fallback_hf_repo="AITRADER/FLUX1-dev-mlx-4bit",
            approx_download_gb=7.0,
            approx_runtime_ram_gb=11.5,
            supports_guidance_scale=True,
            default_steps=25,
            max_recommended_steps=40,
            default_guidance=3.5,
            default_width=1024,
            default_height=1024,
        ),
        ModelSpec(
            id="sd35-large-4bit",
            display_name="Stable Diffusion 3.5 Large — Alt",
            tier="alternate",
            engine="diffusionkit",
            family="sd35",
            hf_repo="argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized",
            fallback_hf_repo=None,
            approx_download_gb=5.0,
            approx_runtime_ram_gb=9.0,
            supports_guidance_scale=True,
            default_steps=28,
            max_recommended_steps=50,
            default_guidance=4.5,
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
