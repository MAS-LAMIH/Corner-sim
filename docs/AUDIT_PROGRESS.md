# CornerSim technical audit and validation status

Updated: 2026-09-26

## Live Windows follow-up

A Windows run with Python 3.12 and CARLA 0.9.16 proved that the previous mocked
pipeline test did not cover Qt thread affinity or settings-proxy aliasing. Sensor
callbacks were constructing `QPixmap` and modifying `QLabel` objects outside the GUI
thread; Stop reset GUI/client state before CARLA cleanup completed; and retaining a
CARLA settings object was not sufficient evidence that its original values were
independent. The live run captured a complete sample and removed actors, but crashed
the GUI on Stop and left the world synchronous at a 0.1 second fixed delta.

This pass removes Qt from `Synchro3` and `gui_workers`: callbacks copy raw BGRA bytes
into immutable `PreviewFrame` values and emit queued signals. Only the GUI-thread
`MainWindow.on_preview_ready` creates `QImage`/`QPixmap` and updates labels. Progress,
results, and errors also cross threads through Qt signals. Stop is cooperative and
does not reset state or start post-processing; finalization happens from the
`QThread.finished` handler after the runtime context has restored CARLA. Cancellation
explicitly skips post-processing. Window closure is asynchronous: it ignores the
close event while cleanup is running and closes after the worker finishes.

`CarlaRuntime` now snapshots primitive `synchronous_mode` and
`fixed_delta_seconds` values before mutation and restores them onto a fresh settings
object. CARLA 0.9.16 may expose only `set_synchronous_mode`; CornerSim therefore
mutates a Traffic Manager only under an explicit exclusive-ownership contract and
uses dedicated port 8050 rather than CARLA's commonly shared default manager. It
leaves that owned manager asynchronous during cleanup and does not claim to recover
an unknowable previous manager state. Cleanup failures are reported even when another
exception is active.

The merged lifecycle work also regressed two compatibility points: public
`MainWindow` lifecycle signals were absent, and the runtime rejected CARLA 0.9.16
Traffic Manager bindings that expose a setter but no getter. The signals are again
declared with their typed `dict`/`str` signatures and connected during initialization. Runtime tests
now cover setter-only owned managers, non-owned managers (never mutated), enable and
disable failures, cancellation, and fresh world-settings restoration.

A subsequent Windows run exposed a native Qt abort before completion handling, at both
natural completion and Stop: `QObject::~QObject: Timers cannot be stopped from another
thread`. This disproved the earlier assertion that the QThread completion chain itself
was the root cause. The complete object audit found two remaining ownership violations:
Qt-bound signal emitters were passed into the CARLA runner and retained by native sensor
callback closures, and a legacy GUI method created a `QTimer`, started it, then
overwrote its only Python reference with a native timer ID. The latter method was dead
in the maintained path but was unsafe and has been removed.

The maintained worker path now creates **no QObject at all**. Simulation and
post-processing use plain Python threads and publish plain values through
`SimpleQueue`. CARLA callbacks retain only `functools.partial` objects containing that
queue—never a QObject or bound Qt signal. One `QTimer(self)` is created, polled,
stopped, and eventually destroyed exclusively on the GUI thread. It drains preview,
progress, completion, and error state and emits public Qt signals from the GUI thread.
This structurally prevents worker/native garbage collection from destroying Qt
objects. The exact Windows offender cannot be named with post-fix certainty until the
reported host reruns the patch; the code no longer depends on the previously unproven
QThread-destruction diagnosis.

## Execution path verified in this pass

The GUI path is `new_ui.MainWindow.start_scenario` → plain `SimulationWorker` thread →
`Synchro3.run_carla_simulation` → `CarlaRuntime` → per-run sensor queue →
`FrameSynchronizer`/`collect_frame` → `SampleWriter` → `image_tools.post_process`.
This is now the same path that uses the maintained lifecycle, synchronization,
scenario-validation, geometry, annotation, and atomic-output code. The historical
`image_tools_.py` file is only a compatibility re-export and no longer contains a
second annotation implementation.

Corner-case selection now passes the selected YAML file into the runner. The validated
actions relocate the spectator/ego and spawn or control the requested actors. Normal
mode retains the prototype's seeded random-prop workflow.

## Findings and disposition

| Severity | Issue/root cause | Production source/function | Regression test | Status |
| --- | --- | --- | --- | --- |
| Critical | Non-positive camera depth produced invalid/infinite projections. | `cornersim.geometry.project_point/project_bbox`; delegated by `Python/image_tools.py`, `Python/tools.py`, and experimental `Editor_UI.py` | `tests/test_geometry.py`, `tests/test_legacy_annotation_integration.py` | Resolved for non-archived application paths. `Python/backup` and the standalone diagnostic `Python/test.py` remain historical. |
| Critical | Bounding-box matcher referenced undefined areas. | `Python/image_tools.py::match` | Covered indirectly by annotation suite; direct geometry/intersection behavior remains deterministic. | Resolved. |
| Critical | Y clipping used width on non-square images. | `Python/image_tools.py::refine_bbs_worker` now delegates clipping to `project_bbox` | `test_bbox_is_clipped_to_image_boundaries` | Resolved. |
| Critical | Actual capture combined arbitrary queue entries from different frames and wrote them immediately. | `Python/Synchro3.py::run_carla_simulation`; `cornersim.synchronization.collect_frame` | `tests/test_capture_pipeline.py`, `tests/test_synchronization.py` | Resolved in actual GUI runner; incomplete frames raise and are not committed. |
| Critical | Instance-mask annotation reversed X/Y, merged disconnected regions, reused IDs, and used inclusive maxima. | `cornersim.annotations.annotations_from_masks`; called by `image_tools.process_instance_semantic_segmentation_` | `tests/test_annotations.py`, `tests/test_legacy_annotation_integration.py` | Resolved. |
| High | World synchronous settings, fixed delta, Traffic Manager sync, sensors, and actors leaked on exceptions. | `cornersim.runtime.CarlaRuntime`; used by `Synchro3.run_carla_simulation` | `tests/test_runtime.py`, `tests/test_capture_pipeline.py` | Resolved under normal Python exception/GUI-stop paths. Process termination/UE crash cannot run cleanup. |
| High | Fixed timestep was incorrectly hard-coded to one second rather than `1/fps`. | `CarlaRuntime.__enter__` | `test_runtime_restores_settings_and_destroys_in_reverse_after_exception` | Resolved. |
| High | Global sensor queue retained stale callbacks across runs. | Per-run queue in `Synchro3.run_carla_simulation`; sensor ownership in `CarlaRuntime` | `tests/test_capture_pipeline.py` | Resolved. |
| High | Dataset files were non-atomic, overwrite-prone, and had no completion contract. | `cornersim.dataset.SampleWriter/atomic_write_json` | `tests/test_dataset.py`, `tests/test_capture_pipeline.py` | Resolved for capture metadata/object data. CARLA PNGs are staged and completion metadata is published last. Post-processing remains a distinct phase and validator rejects missing labels. |
| High | GUI Stop did not stop its worker; Pause did not pause it; repeated Start added state/callbacks; close lacked cleanup. | `gui_workers.SimulationWorker`; `new_ui.start_scenario/stop_scenario/pause_scenario/closeEvent` | `tests/test_gui_workers.py`, `tests/test_gui_resource_paths.py`, `tests/test_capture_pipeline.py` | Resolved in QtCore event-loop and mocked-CARLA tests; full QWidget/live-CARLA confirmation remains unavailable. |
| High | Scenario validation/seeding existed but production ignored it. | `Synchro3._apply_scenario/run_carla_simulation`; `tools.load_scenario_from_yaml` | `tests/test_scenario.py`; live behavior unavailable | Resolved by code and unit tests; live actor placement remains unverified. |
| High | Semantic and instance correspondence lacked dimension checks and deterministic canonical labels. | `annotations_from_masks` | `tests/test_annotations.py`, `tests/test_legacy_annotation_integration.py` | Resolved; `car` wins over its legacy `vehicle` color alias. |
| High | CARLA end-to-end behavior was unverified. | Integration test and manual connection probe | `tests/integration/test_carla_connection.py` | Unresolved: Python client imports, but localhost:2000 timed out and no server executable was found. |
| Critical | CARLA callbacks directly updated Qt widgets and built pixmaps off the GUI thread, crashing Stop on Windows. | `Python/Synchro3.py::sensor_callback`, `Python/gui_workers.py`, `new_ui.MainWindow.on_preview_ready` | `tests/test_gui_workers.py`; Windows observation | Resolved by code/unit tests; the repaired GUI has not been rerun against live CARLA in this environment. |
| Critical | Stop reset state immediately and close synchronously joined a worker, racing cleanup and Qt object destruction. | `new_ui.MainWindow.stop_scenario/_simulation_thread_finished/closeEvent` | `tests/test_gui_resource_paths.py`, `tests/test_gui_workers.py` | Resolved by cooperative cancellation and asynchronous finalization; Qt widget integration is not executable on this headless host. |
| Critical | Original CARLA settings could be represented by an aliased mutable proxy and were not reliably restored. | `cornersim.runtime.CarlaRuntime` | `tests/test_runtime.py::test_aliased_settings_values_are_restored_after_cancellation` | Resolved using immutable value snapshots and fresh restore settings; awaits repeat live validation. |
| Medium | Machine-specific CARLA path and working-directory-sensitive config. | `Python/tools.py` configuration | Import/compile checks | Resolved with `CARLA_ROOT`, file-relative config, and actionable launch error. |
| Medium | Dataset split leakage from consecutive frames/repeated configurations. | No splitting feature exists. | None | Unresolved. Split by run/scenario/seed, never random frame, until group-aware manifests are implemented. |
| Medium | OpenCV GUI wheel requires `libGL.so.1` in this headless environment. | `requirements.txt`/legacy rendering | Unit annotation core avoids OpenCV | Partially resolved. GUI hosts still need system OpenGL; a future headless extra may use `opencv-python-headless`. |
| Low | Archived prototypes, debug output, broad exception handlers, and dead imports remain. | `Python/backup`, `Editor_UI.py`, older `Synchro*.py` | Compile check only | Unresolved; retained to avoid silently deleting research artifacts. |

## Dataset integrity and interruption behavior

No example dataset is committed to the repository, so there was no real export to run
through the validator. `SampleWriter` checks every measurement's CARLA frame, stages
all three PNGs and object metadata, refuses overwrites, and writes `metadata/image_N.json`
last. A crash may leave orphan files, but never a completion marker; the validator
reports missing metadata, missing pairs, orphan streams, dimension mismatches, corrupt
images, invalid labels, and out-of-image boxes. Post-processing writes legacy-compatible
`labels/refined_output_N.json`; the validator recognizes this layout without changing
its schema.

## Validation categories

1. **Statically verified:** GUI routing, cleanup ownership, validated scenario routing,
   production imports/delegation, absence of duplicate maintained annotation code, and
   compilation of all Python sources.
2. **Unit-tested:** projection/near plane/clipping, mask annotations and class mapping,
   schema validation/seeding, same-frame collection diagnostics, atomic publishing,
   validator checks, and lifecycle restoration under failures.
3. **Mocked actual pipeline:** the real `Synchro3.run_carla_simulation` ran for two
   ticks against CARLA-compatible fakes. It captured equal frame IDs, wrote completion
   metadata/object snapshots, stopped sensors, destroyed every actor, restored world
   settings, and restored Traffic Manager asynchronous mode.
4. **Live CARLA:** the user-provided Windows observation verified capture and exposed
   the defects above, but it predates this fix. **No post-fix live CARLA run was
   performed in this environment.** `carla` imported successfully, but a three-second
   connection attempt to `127.0.0.1:2000` timed out; no CARLA server executable was on
   `PATH`. Renderer, spawn collision behavior, semantic palette byte order, real sensor
   timing, repeated live runs, occlusion fidelity, and scientific output remain unverified.

## Paper capability comparison

| Paper capability | Implementation | Tested? | Evidence | Remaining limitation |
| --- | --- | --- | --- | --- |
| Scenario management | Validated YAML list/mapping, actor references, selected GUI scenario routing | Unit + static | `scenario.py`, `_apply_scenario`, scenario tests | No live placement/run test; editing remains basic. |
| Scenario-level corner cases | Person-on-street and car-accident YAML files | Static only | taxonomy and YAML examples | Scientific realism not evaluated. |
| RGB acquisition | Attached RGB camera, exact-frame aggregation, staged PNG | Mocked actual path | capture-pipeline test | Live CARLA color/encoding not checked. |
| Semantic segmentation | CARLA semantic camera converted to CityScapes palette and frame-paired | Mocked pairing; synthetic palette tests | capture and annotation tests | Real palette byte order/rendering not checked. |
| Automatic bounding boxes | Instance-mask boxes plus projected CARLA metadata refinement | Unit + synthetic legacy integration | geometry/annotation tests | Live simulator ground-truth comparison absent. |
| Visibility filtering | Positive depth, full near-plane rejection, image clipping, minimum pixels/area, semantic majority and 100 m/forward filtering | Unit | geometry and annotation tests | Occlusion threshold remains heuristic; partial near-plane objects are conservatively excluded. |
| Dataset export | RGB/semantic/instance/object metadata, completion markers, post-processed labels | Unit + mocked path | dataset/capture tests | No committed/live dataset; no standard COCO/KITTI exporter. |
| Object-level corner cases | Random CARLA static props; taxonomy examples marked unimplemented | Mocked spawning only | runner and taxonomy | Paper-level configurable object anomalies are not implemented. |
| Scene/domain/pixel corner cases | Taxonomy descriptions only | No | `CC_terminology.json` flags | Not implemented. |
| Dataset generation | Seeded runs, synchronized samples, validator | Unit + mocked path | runner, writer, validator tests | CARLA determinism, throughput, resume, and split manifests unverified/unimplemented. |

## Exact validation performed in this pass

- `python -m pytest -q`: 53 passed, 1 skipped. The skipped test is the offscreen
  QWidget smoke test because this host lacks the required Qt/OpenGL runtime; scenario
  validation and the QtCore worker tests passed. Skips are not counted as successful
  tests.
- `python -m compileall -q cornersim Python`: passed.
- `git diff --check`: passed.
- `QT_QPA_PLATFORM=offscreen python Python/new_ui.py`: could not start on this
  headless Linux host because the installed PyQt5/OpenCV stack requires missing
  `libGL.so.1`. Resource-path behavior is covered by source-level regression tests,
  while the repaired widget flow still requires Windows/live-GUI confirmation.
- CARLA probe using `carla.Client('127.0.0.1', 2000)` with a 3-second timeout: failed
  with a simulator timeout. Therefore the opt-in live integration test was not run as
  a successful test.

CornerSim is **not fully or scientifically validated**. The highest priority next step
is two repeated short runs against the supported CARLA version with inspection of frame metadata,
semantic colors, annotations, cleanup, and deterministic seeds, followed by grouped
train/validation/test manifest support.
