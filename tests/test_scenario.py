import random

import numpy as np
import pytest

from cornersim.scenario import ScenarioValidationError, load_scenario, seed_everything, validate_scenario


def test_legacy_scenario_loads():
    pytest.importorskip("yaml")
    scenario = load_scenario("Python/Example.yaml")
    assert len(scenario.actions) == 3
    assert scenario.actions[1]["vehicle_id"] == "v1"


def test_invalid_action_reports_index():
    with pytest.raises(ScenarioValidationError, match="action 0"):
        validate_scenario([{"type": "spawn_vehicle", "location": [1, 2], "orientation": [0, 0, 0]}])


def test_duplicate_actor_ids_are_rejected():
    action = {"type": "spawn_vehicle", "vehicle_id": "v1", "location": [0, 0, 0], "orientation": [0, 0, 0]}
    with pytest.raises(ScenarioValidationError, match="duplicate"):
        validate_scenario([action, action])


def test_seed_everything_is_repeatable():
    seed_everything(7)
    first = random.random(), np.random.random()
    seed_everything(7)
    assert (random.random(), np.random.random()) == first
