"""Strict, dependency-light scenario loading and reproducibility support."""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
SUPPORTED_ACTIONS = {"spectator", "spawn_vehicle", "spawn_pedestrian", "change_direction", "pedestrian_jump"}


class ScenarioValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Scenario:
    actions: tuple[dict[str, Any], ...]
    seed: int | None = None


def _vector(value: Any, field: str, index: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 3 or any(not isinstance(v, (int, float)) for v in value):
        raise ScenarioValidationError(f"action {index}: {field} must be a three-number list")
    return [float(v) for v in value]


def validate_scenario(data: Any) -> Scenario:
    seed = None
    if isinstance(data, dict):
        seed, data = data.get("seed"), data.get("actions")
    if not isinstance(data, list) or not data:
        raise ScenarioValidationError("scenario must contain a non-empty action list")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ScenarioValidationError("seed must be an integer")
    ids: set[str] = set()
    actions = []
    for index, original in enumerate(data):
        if not isinstance(original, dict):
            raise ScenarioValidationError(f"action {index}: expected a mapping")
        action = dict(original)
        kind = action.get("type")
        if kind not in SUPPORTED_ACTIONS:
            raise ScenarioValidationError(f"action {index}: unsupported type {kind!r}")
        if kind in {"spectator", "spawn_vehicle", "spawn_pedestrian", "pedestrian_jump"}:
            action["location"] = _vector(action.get("location"), "location", index)
            action["orientation"] = _vector(action.get("orientation"), "orientation", index)
        actor_field = "vehicle_id" if "vehicle" in kind or kind == "change_direction" else "pedestrian_id"
        if kind.startswith("spawn_"):
            actor_id = action.get(actor_field)
            if not isinstance(actor_id, str) or not actor_id.strip():
                raise ScenarioValidationError(f"action {index}: {actor_field} must be a non-empty string")
            if actor_id in ids:
                raise ScenarioValidationError(f"action {index}: duplicate actor id {actor_id!r}")
            ids.add(actor_id)
        if "speed" in action and not isinstance(action["speed"], (int, float)):
            raise ScenarioValidationError(f"action {index}: speed must be numeric")
        actions.append(action)
    return Scenario(tuple(actions), seed)


def load_scenario(path: str | Path) -> Scenario:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ScenarioValidationError(f"scenario file does not exist: {candidate}")
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ScenarioValidationError("PyYAML is required to load scenario files; install the project dependencies") from error
    try:
        return validate_scenario(yaml.safe_load(candidate.read_text(encoding="utf-8")))
    except yaml.YAMLError as error:
        raise ScenarioValidationError(f"invalid YAML in {candidate}: {error}") from error


def seed_everything(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
