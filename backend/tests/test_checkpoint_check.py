import json
import struct

from mlxstudio_backend import checkpoint_check


def write_safetensors_header(path, names, metadata=None):
    """Writes a file with a valid safetensors header (tensor data is not
    needed: the checker only reads the header)."""
    header = {name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for name in names}
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00\x00")


def test_tensor_names_reads_all_shards_and_skips_metadata(tmp_path):
    write_safetensors_header(tmp_path / "transformer" / "0.safetensors", ["a.weight"], {"quantization_level": "4"})
    write_safetensors_header(tmp_path / "transformer" / "1.safetensors", ["b.weight", "b.scales"])
    write_safetensors_header(tmp_path / "transformer" / "._0.safetensors", ["macos.resource.fork"])

    assert checkpoint_check.tensor_names(tmp_path / "transformer") == {"a.weight", "b.weight", "b.scales"}


def test_matching_checkpoint_reports_nothing_missing(tmp_path):
    write_safetensors_header(tmp_path / "vae" / "0.safetensors", ["x", "y", "extra.ignored"])
    assert checkpoint_check.missing_tensors(tmp_path, {"vae": {"x", "y"}}) == {}


def test_mismatched_layout_is_reported_per_component(tmp_path):
    write_safetensors_header(tmp_path / "vae" / "0.safetensors", ["x"])
    write_safetensors_header(tmp_path / "transformer" / "0.safetensors", ["old_name.weight"])
    expected = {"vae": {"x"}, "transformer": {"new_name.weight", "other.weight"}}

    missing = checkpoint_check.missing_tensors(tmp_path, expected)

    assert missing == {"transformer": ["new_name.weight", "other.weight"]}
    assert "transformer: 2 of 2 parameters missing" in checkpoint_check.describe(missing, expected)


def test_absent_component_directory_counts_as_missing(tmp_path):
    missing = checkpoint_check.missing_tensors(tmp_path, {"text_encoder": {"layer.0.weight"}})
    assert missing == {"text_encoder": ["layer.0.weight"]}
