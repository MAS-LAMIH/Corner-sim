"""Atomic annotation export and offline CornerSim dataset validation."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
import shutil
import uuid
from typing import Any

from PIL import Image, UnidentifiedImageError


class DatasetWriteError(ValueError):
    pass


class SampleWriter:
    """Stage a sample and publish its completion marker last.

    A validator only accepts frames with a metadata marker, so interruption cannot
    make a partially published frame appear complete.
    """

    STREAMS = ("rgb", "semantic_segmentation", "instance_segmentation")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        for folder in (*self.STREAMS, "simulation_objects", "metadata"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def write(self, frame: int, measurements: dict[str, Any], simulation_objects: dict[str, Any]) -> None:
        if set(measurements) != set(self.STREAMS):
            raise DatasetWriteError(f"frame {frame} has streams {sorted(measurements)}, expected {list(self.STREAMS)}")
        stem = f"image_{frame}"
        destinations = [self.root / stream / f"{stem}.png" for stream in self.STREAMS]
        destinations.extend((self.root / "simulation_objects" / f"{stem}.json",
                             self.root / "metadata" / f"{stem}.json"))
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing sample {frame}: {existing}")
        staging = self.root / f".sample-{frame}-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            for stream in self.STREAMS:
                image = measurements[stream]
                if getattr(image, "frame", frame) != frame:
                    raise DatasetWriteError(f"{stream} frame {getattr(image, 'frame', None)} != {frame}")
                image.save_to_disk(str(staging / f"{stream}.png"))
            atomic_write_json(staging / "objects.json", simulation_objects)
            for stream in self.STREAMS:
                os.replace(staging / f"{stream}.png", self.root / stream / f"{stem}.png")
            os.replace(staging / "objects.json", self.root / "simulation_objects" / f"{stem}.json")
            atomic_write_json(self.root / "metadata" / f"{stem}.json", {
                "frame": frame, "streams": list(self.STREAMS), "complete": True
            })
        finally:
            shutil.rmtree(staging, ignore_errors=True)


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
    legacy_labels = root / "labels"
    if not annotation_dir.is_dir() and legacy_labels.is_dir():
        annotation_dir = legacy_labels
    for name, folder in folders.items():
        if not folder.is_dir():
            report.errors.append(f"missing {name} directory: {folder}")
    if not annotation_dir.is_dir():
        report.errors.append(f"missing annotations directory: expected {root / 'annotations'} or {legacy_labels}")
    if report.errors:
        return report
    rgb_files = {path.stem: path for path in folders["rgb"].glob("*.png")}
    if not rgb_files:
        report.errors.append(f"no PNG frames found in {folders['rgb']}")
        return report
    for stream, folder in folders.items():
        if stream == "rgb":
            continue
        extra = {path.stem for path in folder.glob("*.png")} - set(rgb_files)
        for stem in sorted(extra):
            report.errors.append(f"{stream}: orphan frame {stem}")
    metadata_dir = root / "metadata"
    for stem, rgb_path in sorted(rgb_files.items()):
        report.frames_checked += 1
        related = {name: folder / f"{stem}.png" for name, folder in folders.items() if name != "rgb"}
        frame_number = stem.removeprefix("image_")
        annotation = annotation_dir / f"{stem}.json"
        if annotation_dir == legacy_labels:
            annotation = annotation_dir / f"refined_output_{frame_number}.json"
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
        if metadata_dir.is_dir():
            metadata_path = metadata_dir / f"{stem}.json"
            if not metadata_path.is_file():
                report.errors.append(f"frame {stem}: missing completion metadata {metadata_path}")
            else:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("frame") != int(frame_number) or metadata.get("complete") is not True:
                        report.errors.append(f"frame {stem}: invalid completion metadata")
                    if set(metadata.get("streams", [])) != set(folders):
                        report.errors.append(f"frame {stem}: metadata stream list is inconsistent")
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    report.errors.append(f"frame {stem}: unreadable completion metadata ({error})")
    return report
