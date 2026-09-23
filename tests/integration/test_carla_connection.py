"""Opt-in smoke tests; these never claim to launch or validate the renderer."""

import os

import pytest


pytestmark = pytest.mark.carla


def test_carla_server_reports_a_world():
    if os.environ.get("CORNERSIM_RUN_CARLA_TESTS") != "1":
        pytest.skip("set CORNERSIM_RUN_CARLA_TESTS=1 with a running CARLA server")
    carla = pytest.importorskip("carla")
    client = carla.Client(os.environ.get("CARLA_HOST", "127.0.0.1"), int(os.environ.get("CARLA_PORT", "2000")))
    client.set_timeout(10.0)
    world = client.get_world()
    assert world.get_map().name
