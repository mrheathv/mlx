import pytest

from mlxstudio_backend.models import MODEL_REGISTRY, UnknownModelError, get_model


def test_registry_has_the_three_v1_models():
    assert set(MODEL_REGISTRY.keys()) == {
        "flux1-schnell-4bit", "flux1-dev-4bit", "sd35-large-4bit",
    }


def test_tiers_are_distinct_and_expected():
    tiers = {spec_id: spec.tier for spec_id, spec in MODEL_REGISTRY.items()}
    assert tiers["flux1-schnell-4bit"] == "fast"
    assert tiers["flux1-dev-4bit"] == "quality"
    assert tiers["sd35-large-4bit"] == "alternate"


def test_engine_routing_matches_family():
    assert MODEL_REGISTRY["flux1-schnell-4bit"].engine == "mflux"
    assert MODEL_REGISTRY["flux1-dev-4bit"].engine == "mflux"
    assert MODEL_REGISTRY["sd35-large-4bit"].engine == "diffusionkit"


def test_every_model_declares_a_positive_memory_footprint():
    for spec in MODEL_REGISTRY.values():
        assert spec.approx_download_gb > 0
        assert spec.approx_runtime_ram_gb > 0
        # Runtime residency (activations, VAE, text encoder, MLX buffers)
        # should always be estimated at or above the on-disk weight size.
        assert spec.approx_runtime_ram_gb >= spec.approx_download_gb


def test_schnell_has_no_guidance_scale():
    assert MODEL_REGISTRY["flux1-schnell-4bit"].supports_guidance_scale is False


def test_get_model_returns_spec():
    assert get_model("sd35-large-4bit").hf_repo == "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized"


def test_get_model_raises_for_unknown_id():
    with pytest.raises(UnknownModelError):
        get_model("does-not-exist")
