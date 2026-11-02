import json
from pathlib import Path

import pytest

from mlxstudio_backend.models import MODEL_REGISTRY, UnknownModelError, get_model

SWIFT_CATALOG = Path(__file__).resolve().parents[2] / "Sources" / "MLXStudioApp" / "Resources" / "models.json"


def test_registry_has_the_three_models():
    assert set(MODEL_REGISTRY.keys()) == {
        "flux1-schnell-4bit", "flux1-dev-4bit", "z-image-turbo-4bit",
    }


def test_tiers_are_distinct_and_expected():
    tiers = {spec_id: spec.tier for spec_id, spec in MODEL_REGISTRY.items()}
    assert tiers["flux1-schnell-4bit"] == "fast"
    assert tiers["flux1-dev-4bit"] == "quality"
    assert tiers["z-image-turbo-4bit"] == "alternate"


def test_all_models_run_on_mflux():
    assert {spec.engine for spec in MODEL_REGISTRY.values()} == {"mflux"}


def test_flux_models_declare_their_mflux_base_config():
    assert MODEL_REGISTRY["flux1-schnell-4bit"].mflux_base == "schnell"
    assert MODEL_REGISTRY["flux1-dev-4bit"].mflux_base == "dev"


def test_every_model_declares_a_positive_memory_footprint():
    for spec in MODEL_REGISTRY.values():
        assert spec.approx_download_gb > 0
        # Runtime residency (activations, VAE, text encoder, MLX buffers)
        # should always be estimated at or above the on-disk weight size.
        assert spec.approx_runtime_ram_gb >= spec.approx_download_gb


def test_distilled_models_have_no_guidance_scale():
    assert MODEL_REGISTRY["flux1-schnell-4bit"].supports_guidance_scale is False
    assert MODEL_REGISTRY["z-image-turbo-4bit"].supports_guidance_scale is False


def test_get_model_returns_spec():
    assert get_model("z-image-turbo-4bit").hf_repo == "filipstrand/Z-Image-Turbo-mflux-4bit"


def test_get_model_raises_for_unknown_id():
    with pytest.raises(UnknownModelError):
        get_model("does-not-exist")


@pytest.mark.parametrize(
    "py_field,swift_field",
    [
        ("display_name", "displayName"),
        ("tier", "tier"),
        ("engine", "engine"),
        ("family", "family"),
        ("hf_repo", "hfRepo"),
        ("approx_download_gb", "approxDownloadGB"),
        ("approx_runtime_ram_gb", "approxRuntimeRAMGB"),
        ("supports_guidance_scale", "supportsGuidanceScale"),
        ("default_steps", "defaultSteps"),
        ("max_recommended_steps", "maxRecommendedSteps"),
        ("default_guidance", "defaultGuidance"),
        ("default_width", "defaultWidth"),
        ("default_height", "defaultHeight"),
    ],
)
def test_swift_catalog_mirrors_python_registry(py_field, swift_field):
    swift_models = {m["id"]: m for m in json.loads(SWIFT_CATALOG.read_text())["models"]}
    assert set(swift_models) == set(MODEL_REGISTRY)
    for model_id, spec in MODEL_REGISTRY.items():
        assert swift_models[model_id][swift_field] == getattr(spec, py_field), model_id
