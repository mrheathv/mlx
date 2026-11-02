import pytest

from mlxstudio_backend.diagnostics import architecture_generation, neural_accelerator_status

M5 = "applegpu_g17g"
M4 = "applegpu_g16g"
MACOS26_WHEEL = "py3-none-macosx_26_0_arm64"


@pytest.mark.parametrize(
    "arch,expected",
    [("applegpu_g17s", (17, "s")), ("applegpu_g16g", (16, "g")), ("applegpu_g18p", (18, "p")), ("x", None), ("applegpu_gXXs", None)],
)
def test_architecture_generation_matches_mlx_parsing(arch, expected):
    assert architecture_generation(arch) == expected


def test_active_on_m5_with_new_macos_and_mlx():
    active, why = neural_accelerator_status("0.32.4", "26.2", M5, MACOS26_WHEEL)
    assert active, why


@pytest.mark.parametrize(
    "mlx,macos,arch,wheel,reason",
    [
        ("0.29.3", "26.2", M5, MACOS26_WHEEL, "predates"),
        ("0.32.4", "26.1", M5, MACOS26_WHEEL, "older than 26.2"),
        ("0.32.4", "26.2", M4, MACOS26_WHEEL, "M5"),
        ("0.32.4", "26.2", M5, "py3-none-macosx_15_0_arm64", "not the macOS 26 build"),
        ("0.32.4", "26.2", "applegpu_g17p", MACOS26_WHEEL, "needs 18+"),
        ("0.32.4", "", M5, MACOS26_WHEEL, "unknown"),
    ],
)
def test_inactive_explains_why(mlx, macos, arch, wheel, reason):
    active, why = neural_accelerator_status(mlx, macos, arch, wheel)
    assert not active
    assert reason in why


def test_version_comparison_is_numeric_not_lexical():
    # "0.100" > "0.30" numerically even though "1" < "3" lexically.
    active, _ = neural_accelerator_status("0.100.0", "26.10", M5, MACOS26_WHEEL)
    assert active
