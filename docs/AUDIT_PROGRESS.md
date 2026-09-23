# CornerSim technical audit and progress

Date: 2026-09-23

## Architecture and verified capability

CornerSim is a PyQt GUI around CARLA 0.9.14-era scripts. `Python/new_ui.py` owns the
main workflow; `Python/tools.py` combines scenario actions, camera callbacks, 3-D
projection, rendering, recording, and CARLA process control. `Python/image_tools.py`
post-processes instance/semantic PNGs and associates visible masks with projected
CARLA objects. Three YAML examples implement two scenario-level corner cases. The
taxonomy lists scene, object, domain, and pixel categories, but marks none of those
examples as implemented. This distinction is important: the taxonomy is not an
implementation inventory.

The new `cornersim` package is deliberately simulator-independent. It provides
strict scenario validation, deterministic Python/NumPy seeding, safe camera
projection, same-frame measurement aggregation, atomic JSON output, and an offline
dataset integrity validator. It does not claim to validate CARLA rendering fidelity.

## Prioritized findings

| Severity | Root cause and impact | Status |
| --- | --- | --- |
| Critical | Projection divided by non-positive depth, allowing behind-camera or near-plane geometry into annotations. | Fixed in the tested core; legacy rendering still needs migration. |
| Critical | `image_tools.match` referenced undefined areas and crashed whenever called. | Fixed in both duplicate modules. |
| Critical | Bounding-box Y clipping used image width, corrupting boxes for non-square images. | Fixed in both duplicate modules. |
| High | Sensor callback data used an unbounded global queue without a reusable same-frame contract. | Added bounded `FrameSynchronizer`; GUI migration remains. |
| High | YAML accepted malformed actions, duplicate IDs, and unclear failures. | Added strict validation while supporting legacy list-root YAML. |
| High | Randomness was spread across scripts and no reusable seed facility existed. | Added deterministic Python/NumPy seeding; CARLA Traffic Manager must also be seeded during integration. |
| High | JSON writes could be partial or overwrite results. | Added atomic, non-overwriting JSON writer. Existing GUI writes need migration. |
| High | Dataset exports had no consistency check. | Added CLI validation of pairs, PNG integrity/dimensions, JSON shape, labels, and box bounds. |
| High | World settings/actors are not always restored in `finally` blocks. | Outstanding; requires live-CARLA regression testing before changing lifecycle behavior. |
| Medium | Hard-coded Windows CARLA path was committed. | Cleared; users configure it locally. Environment-variable configuration is recommended next. |
| Medium | Core scripts import CARLA/PyQt eagerly, blocking headless tests. | New core is independent; legacy decomposition remains. |
| Medium | Dependencies were incomplete (`Pillow`, `PyYAML`) and unconstrained. | Added compatible ranges and package metadata. |
| Medium | Duplicate experimental scripts have diverged. | Documented; not deleted to preserve research behavior. |
| Low | Debug output, spelling, broad exception handlers, and dead imports remain. | Outstanding. |

## Validation and scientific limitations

Unit tests use analytically known camera geometry and synthetic images. They verify
software invariants, not the scientific effectiveness of generated corner cases.
No CARLA server, Unreal renderer, GPU, published experiment dataset, or model weights
were available in this audit environment. Consequently, end-to-end capture,
occlusion fidelity, performance claims, and reproduction of paper results remain
unverified. Similar consecutive frames can leak across ML splits; split datasets by
scenario/seed/run rather than randomly by frame.

## Next actions

1. Migrate the GUI to `FrameSynchronizer` and atomic per-frame export with a manifest.
2. Introduce one lifecycle context manager that snapshots/restores world settings and
   stops/destroys sensors before actors, tested against CARLA 0.9.14.
3. Replace duplicate post-processors with the tested projection API and record explicit
   exclusion reasons (`behind_camera`, `near_plane`, `outside_image`, `too_small`).
4. Add CI for headless tests and a separately triggered CARLA integration job.
5. Add scenario/run IDs and split manifests to prevent data leakage, plus checksums to
   support interruption/resume.
6. Measure throughput and memory only after the correctness migration; no performance
   improvement is claimed by this change.
