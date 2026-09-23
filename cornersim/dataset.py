"""Atomic annotation export and offline CornerSim dataset validation."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image, UnidentifiedImageError


class DatasetWriteError(ValueError):
    pass


@dataclass
class ValidationReport:
    frames_checked: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "frames_checked": self.frames_checked,
                "errors": self.errors, "warnings": self.warnings}


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> Path:
    """Durably replace a JSON file without exposing a partially-written document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except Exception:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise
    return destination


def _validate_annotations(path: Path, width: int, height: int, report: ValidationReport) -> None:
    try:
        annotations = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        report.errors.append(f"{path}: unreadable annotation JSON ({error})")
        return
    if not isinstance(annotations, list):
        report.errors.append(f"{path}: annotation root must be a list")
        return
    for index, annotation in enumerate(annotations):
        prefix = f"{path}: annotation {index}"
        if not isinstance(annotation, dict):
            report.errors.append(f"{prefix} must be an object")
            continue
        if not isinstance(annotation.get("label", annotation.get("base_label")), str):
            report.errors.append(f"{prefix} has no string label")
        try:
            x1, y1, x2, y2 = (float(annotation[key]) for key in ("min_x", "min_y", "max_x", "max_y"))
        except (KeyError, TypeError, ValueError):
            report.errors.append(f"{prefix} has invalid bounding-box coordinates")
            continue
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            report.errors.append(f"{prefix} box [{x1}, {y1}, {x2}, {y2}] is outside {width}x{height}")


def validate_dataset(root: str | Path) -> ValidationReport:
    """Validate matching RGB, semantic, instance and annotation files by stem."""
    root = Path(root)
    report = ValidationReport()
    folders = {name: root / name for name in ("rgb", "semantic_segmentation", "instance_segmentation")}
    annotation_dir = root / "annotations"
    for name, folder in (*folders.items(), ("annotations", annotation_dir)):
        if not folder.is_dir():
            report.errors.append(f"missing {name} directory: {folder}")
    if report.errors:
        return report
    rgb_files = {path.stem: path for path in folders["rgb"].glob("*.png")}
    if not rgb_files:
        report.errors.append(f"no PNG frames found in {folders['rgb']}")
        return report
    for stem, rgb_path in sorted(rgb_files.items()):
        report.frames_checked += 1
        related = {name: folder / f"{stem}.png" for name, folder in folders.items() if name != "rgb"}
        annotation = annotation_dir / f"{stem}.json"
        for name, path in (*related.items(), ("annotation", annotation)):
            if not path.is_file():
                report.errors.append(f"frame {stem}: missing {name} file {path}")
        try:
            with Image.open(rgb_path) as image:
                image.verify()
            with Image.open(rgb_path) as image:
                size = image.size
            for name, path in related.items():
                if path.is_file():
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        if image.size != size:
                            report.errors.append(f"frame {stem}: {name} dimensions {image.size} != RGB {size}")
        except (OSError, UnidentifiedImageError) as error:
            report.errors.append(f"frame {stem}: corrupt image ({error})")
            continue
        if annotation.is_file():
            _validate_annotations(annotation, *size, report)
    return report
