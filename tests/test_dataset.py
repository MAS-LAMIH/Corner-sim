import json

from PIL import Image
import pytest

from cornersim.dataset import atomic_write_json, validate_dataset


def make_dataset(root):
    for folder in ("rgb", "semantic_segmentation", "instance_segmentation", "annotations"):
        (root / folder).mkdir()
    for folder in ("rgb", "semantic_segmentation", "instance_segmentation"):
        Image.new("RGB", (20, 10)).save(root / folder / "image_000001.png")
    (root / "annotations" / "image_000001.json").write_text(
        json.dumps([{"label": "car", "min_x": 1, "min_y": 2, "max_x": 12, "max_y": 9}])
    )


def test_valid_dataset(tmp_path):
    make_dataset(tmp_path)
    report = validate_dataset(tmp_path)
    assert report.valid
    assert report.frames_checked == 1


def test_missing_pair_and_out_of_bounds_box_are_reported(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "semantic_segmentation" / "image_000001.png").unlink()
    (tmp_path / "annotations" / "image_000001.json").write_text(
        json.dumps([{"label": "car", "min_x": -1, "min_y": 0, "max_x": 30, "max_y": 8}])
    )
    report = validate_dataset(tmp_path)
    assert not report.valid
    assert any("missing semantic" in error for error in report.errors)
    assert any("outside" in error for error in report.errors)


def test_atomic_json_refuses_accidental_overwrite(tmp_path):
    target = atomic_write_json(tmp_path / "frame.json", {"frame": 1})
    with pytest.raises(FileExistsError):
        atomic_write_json(target, {"frame": 2})
    assert json.loads(target.read_text()) == {"frame": 1}


def test_atomic_json_rejects_nan_without_partial_file(tmp_path):
    target = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        atomic_write_json(target, {"value": float("nan")})
    assert not target.exists()
