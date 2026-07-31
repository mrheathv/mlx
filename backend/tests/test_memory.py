from mlxstudio_backend import memory


def test_assess_fits_when_plenty_available(monkeypatch):
    monkeypatch.setattr(memory, "available_memory_gb", lambda: 20.0)
    result = memory.assess(approx_runtime_ram_gb=9.0)
    assert result.is_likely_to_fit is True
    assert result.estimated_required_gb == 9.0 + memory.RESERVED_SYSTEM_GB


def test_assess_does_not_fit_when_scarce(monkeypatch):
    monkeypatch.setattr(memory, "available_memory_gb", lambda: 8.0)
    result = memory.assess(approx_runtime_ram_gb=11.5)
    assert result.is_likely_to_fit is False
    assert result.estimated_available_gb == 8.0


def test_assess_boundary_is_inclusive(monkeypatch):
    monkeypatch.setattr(memory, "available_memory_gb", lambda: 15.0)
    result = memory.assess(approx_runtime_ram_gb=11.0)
    assert result.estimated_required_gb == 15.0
    assert result.is_likely_to_fit is True


def test_available_memory_gb_falls_back_without_psutil(monkeypatch):
    monkeypatch.setattr(memory, "psutil", None)
    assert memory.available_memory_gb() == float("inf")
