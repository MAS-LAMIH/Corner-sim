# CornerSim

CornerSim is a research prototype for creating CARLA corner-case scenarios and
capturing RGB, semantic-segmentation, instance-segmentation, and object annotation
data. The current GUI implements two scenario-level examples. Other taxonomy entries
describe intended research directions and are **not yet implemented**.

## Compatibility and installation

- Linux or Windows with a CARLA server compatible with 0.9.14 Python APIs
- Python 3.10 or 3.11 for the maintained utilities (the historical GUI was developed
  on Python 3.7, which is end-of-life and is not supported by the new package)
- A dedicated GPU and CARLA/Unreal requirements for interactive simulation

Create an environment and install the project plus GUI dependencies:

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pip install -r requirements.txt
```

Install CARLA itself using its official instructions. Set `path` under `[Carla]` in
`Python/config.ini` to a local installation only if CornerSim should launch the
server. Leaving it empty is appropriate when CARLA is launched separately. Do not
commit a machine-specific path.

## Launching the GUI

Start CARLA, then from the repository root run:

```bash
python Python/new_ui.py
```

Launching from inside `Python/` remains supported. Sensor callbacks copy preview bytes
and deliver them through queued Qt signals; all `QImage`, `QPixmap`, progress-bar, and
widget updates happen on the GUI thread. Stop is cooperative: controls remain in a
stopping state until sensors/actors are cleaned up and the original CARLA world and
Traffic Manager settings have been restored. Cancelled runs are not post-processed.

The GUI connects to localhost port 2000, lets users select taxonomy labels and an
implemented YAML scenario, and displays/captures synchronized sensor views. Treat
generated output as experimental until it passes the validator below.

## Scenario format and reproducible example

Legacy examples are YAML lists. New files may use a root mapping with an integer seed:

```yaml
seed: 42
actions:
  - type: spectator
    location: [-53, 55, 1]
    orientation: [0, 90, 0]
  - type: spawn_vehicle
    vehicle_id: ego
    location: [20, 5, 0]
    orientation: [0, 0, 0]
    speed: 0.5
```

Validate before simulation:

```bash
python -m cornersim.cli validate-scenario Python/Example.yaml
```

A seed controls the new Python/NumPy utility layer. Full repeatability also requires
synchronous CARLA stepping, a fixed delta, deterministic Traffic Manager seeds, the
same CARLA/map/assets/GPU, and stable actor spawn order.

## Dataset structure and validation

The maintained validator expects matching stems:

```text
dataset/
  rgb/image_000001.png
  semantic_segmentation/image_000001.png
  instance_segmentation/image_000001.png
  simulation_objects/image_000001.json
  metadata/image_000001.json
  labels/refined_output_000001.json
```

Completion metadata is published only after all synchronized sensor files and object metadata are staged. Post-processing writes legacy-compatible label JSON containing a string `label` (or `base_label`) and `min_x`, `min_y`, `max_x`, `max_y` bounded by the image. Validate an export with:

```bash
python -m cornersim.cli validate-dataset /path/to/dataset
```

The command exits nonzero for missing/corrupt/mismatched files or invalid boxes.
Existing historical exports may require directory renaming; their JSON schema is not
silently modified.

## Tests

```bash
python -m pytest -q                 # headless unit tests
CORNERSIM_RUN_CARLA_TESTS=1 python -m pytest -q tests/integration
```

The default suite includes a mocked run through the actual capture function, but this is not a live simulator result. CARLA tests must not be interpreted as run unless a compatible server and renderer were actually available. See [the audit progress report](docs/AUDIT_PROGRESS.md) for
resolved findings, limitations, and prioritized next work. See
[`install_Carla.md`](install_Carla.md) for additional simulator setup context.

## Troubleshooting

- **Connection refused:** launch CARLA and check ports 2000/2001 and client/server
  version compatibility.
- **Empty CARLA path:** launch the server manually or configure `Python/config.ini`.
- **Spawn collision:** adjust the scenario transform; do not retry indefinitely.
- **Missing frame:** do not pair measurements from different frames; discard/report
  incomplete frames.
- **Invalid dataset:** retain the run, inspect validator errors, and regenerate rather
  than silently deleting annotations.

CornerSim is not production-ready. Passing unit tests establishes utility-level
correctness only; it does not establish simulation realism or ML performance.
