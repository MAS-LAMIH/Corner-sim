# Native Qt/CARLA isolation harness

These programs are diagnostic-only. They do not alter the production runner, GUI,
CARLA lifecycle, dataset schema, or Qt warning handling. Run every command from
`C:\Corner-sim` in PowerShell with the existing virtual environment activated and
the CARLA server started only where noted.

## One-command Windows matrix

```powershell
Set-Location C:\Corner-sim
New-Item -ItemType Directory -Force diagnostic-logs | Out-Null

# No CARLA server required: dependency probes and fake-runner GUI tests.
powershell -ExecutionPolicy Bypass -File diagnostics\run_windows_isolation.ps1

# With the CARLA 0.9.16 server ready on localhost:2000: adds A and C.
powershell -ExecutionPolicy Bypass -File diagnostics\run_windows_isolation.ps1 -RunLive

# Optional and deliberately separate: actually initialize OpenCV HighGUI.
powershell -ExecutionPolicy Bypass -File diagnostics\run_windows_isolation.ps1 -ExerciseOpenCvGui
```

Every child is launched as `python -u -X faulthandler` in a **fresh process**.
Each run gets separate `stdout`, `stderr`, and JSON-lines checkpoint files plus an
`exit-codes.tsv`, under `diagnostic-logs\YYYYMMDD-HHMMSS`. Native Qt diagnostics and
faulthandler output remain in the corresponding `*.stderr.log`; nothing installs a
Qt message handler or filters warnings. Copy the whole timestamped directory when
reporting results.

## Individual experiments

The individual commands below are useful when bisecting. `Ctrl+C` requests
cooperative cancellation in the standalone runner. GUI harnesses have watchdogs;
their Stop action uses the same cooperative production path. CARLA RPC and sensor
waits retain their production timeouts, so cleanup is bounded by outstanding CARLA
calls rather than unsafe thread termination.

### A — production simulation without Qt

```powershell
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_standalone.py `
  --ticks 5 --capture-interval 1 `
  --output diagnostic-output\standalone `
  --log diagnostic-logs\A-standalone.jsonl `
  1> diagnostic-logs\A-standalone.stdout.log `
  2> diagnostic-logs\A-standalone.stderr.log
```

This calls `Synchro3.run_carla_simulation` directly with its real `CarlaRuntime`,
sensors, synchronization, writer, and cleanup. It checks that PyQt never appears in
`sys.modules`. The CARLA proxy only logs calls and forwards them; actor destruction
and world/Traffic Manager restoration remain owned by `CarlaRuntime`.

### B — real MainWindow with a delayed fake runner

```powershell
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_gui_fake.py --mode natural --duration 2 --log diagnostic-logs\B-natural.jsonl
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_gui_fake.py --mode stop --duration 5 --stop-after 0.5 --log diagnostic-logs\B-stop.jsonl
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_gui_fake.py --mode repeat --duration 1 --log diagnostic-logs\B-repeat.jsonl
```

These invoke the actual `MainWindow.start_scenario()` path, GUI event delivery,
completion handling, Stop behavior, post-processing transition, and window cleanup.
They do not contact a CARLA server. Importing the real `new_ui` module also imports
the CARLA Python binding through production dependencies, so B isolates live CARLA
execution—not the mere presence of the binding. D's PyQt-only fresh process is the
strict no-CARLA comparison. The fake stays alive long enough for normal GUI event
processing rather than completing synchronously.

### C — real MainWindow with live CARLA

```powershell
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_gui_live.py `
  --ticks 5 --capture-interval 1 --output diagnostic-output\gui-live `
  --log diagnostic-logs\C-live.jsonl `
  1> diagnostic-logs\C-live.stdout.log `
  2> diagnostic-logs\C-live.stderr.log
```

Add `--stop-after 3 --ticks 100` for a cancellation run. The harness wraps the
CARLA client/world/actors to log connection, Traffic Manager creation, settings,
spawns, callback registration, first callbacks, first tick, sensor stop, actor
destruction, and settings restoration. It also logs immediately before runner entry
and immediately before/after the exact production output-path resolution line via a
temporary Python line tracer. The proxy does not change production cleanup ordering.

### D — fresh-process native dependency probes

```powershell
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\import_probe.py pyqt --log diagnostic-logs\D-pyqt.jsonl
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\import_probe.py opencv --log diagnostic-logs\D-opencv.jsonl
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\import_probe.py carla --log diagnostic-logs\D-carla.jsonl
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\import_probe.py combined --log diagnostic-logs\D-combined.jsonl
```

The OpenCV probe performs a native image operation and reports the build's `GUI:`
line, but does not equate importing OpenCV with initializing HighGUI. To exercise
that backend explicitly in its own process, append `--exercise-opencv-gui`. Append
`--connect-carla` to a CARLA-containing probe only when the server is running.

### E — post-processing isolation

The supplied Windows results narrow the failure to the transition after successful
natural simulation completion. Run these against the standalone dataset; `--copy-to`
ensures every processor receives an independent copy:

```powershell
$source = "diagnostic-logs\20260926-214245\standalone-dataset"
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_postprocess.py standalone-main --dataset $source --copy-to diagnostic-output\post-main --log diagnostic-logs\E1.jsonl
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_postprocess.py standalone-thread --dataset $source --copy-to diagnostic-output\post-thread --log diagnostic-logs\E2.jsonl
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_postprocess.py qt-import --log diagnostic-logs\E3.jsonl
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_postprocess.py qt-thread --dataset $source --copy-to diagnostic-output\post-qt-thread --log diagnostic-logs\E4.jsonl
```

The updated `-RunLive` matrix performs these automatically, and also adds C0: the
full live GUI natural-completion flow with a logged no-op post-processor. C0 and C1
differ only in whether the real `image_tools.post_process` implementation is used.

To run only that paired control on the affected host:

```powershell
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_gui_live.py --ticks 5 --capture-interval 1 --postprocessing skip --output diagnostic-output\C0 --log diagnostic-logs\C0.jsonl 1> diagnostic-logs\C0.stdout.log 2> diagnostic-logs\C0.stderr.log
.\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_gui_live.py --ticks 5 --capture-interval 1 --postprocessing real --output diagnostic-output\C1 --log diagnostic-logs\C1.jsonl 1> diagnostic-logs\C1.stdout.log 2> diagnostic-logs\C1.stderr.log
```

## Interpretation matrix

| Result | What it supports | What it does **not** establish |
| --- | --- | --- |
| A crashes with the same Qt message while `pyqt_loaded=false` | The trigger is in CARLA, another native dependency, or process-wide native interaction outside `MainWindow`. | Which native QObject exists or which library created it. |
| A passes repeatedly; B crashes | A Qt/MainWindow lifecycle defect is likely independent of live CARLA. | That CARLA cannot amplify the defect. |
| A and B pass; C crashes | The failure requires the GUI/live-CARLA interaction or a dependency initialized only in that combination. | Whether PyQt, CARLA, or OpenCV owns the timer. |
| D combined crashes but individual D probes pass | Import/initialization interaction is sufficient without capture. | Which later capture operation is responsible. |
| D OpenCV compute passes but explicit HighGUI crashes | OpenCV's GUI backend is implicated; plain `cv2` import/compute is insufficient. | Whether CornerSim actually calls HighGUI in production. |
| C0 passes and C1 crashes | Real post-processing, rather than capture cleanup or natural-completion GUI delivery, is required. | Whether import, OpenCV compute, multiprocessing, or their thread interaction is the cause. |
| E1 and E2 pass; E3 passes; E4 crashes | Real post-processing specifically conflicts with a concurrent Qt event loop. | Which native library owns the QObject without a native stack. |
| E1 passes and E2 crashes | Running post-processing from a non-main Python thread is sufficient, without PyQt. | That Qt is involved in the failing process. |
| E3 crashes | Importing `image_tools`/OpenCV beside an active Qt application is sufficient. | That image processing or multiprocessing is involved. |
| All probes pass, including repeated C | The reported sequence was not reproduced under that run's conditions. | A proof that the race is fixed; repeat and retain logs. |

Absence of `runner.output_resolve.return` after `runner.output_resolve.begin` narrows
the interval but does not prove `Path.resolve` caused a native abort. Likewise, Python thread stacks show
where the interpreter happened to be, not the native QObject destructor.

## Capturing the native C++ stack on Windows

Python faulthandler cannot identify the QObject. Use WinDbg Preview or ProcDump and
retain the dump with the text logs.

1. Start CARLA, then open WinDbg Preview and launch the virtual-environment Python:

   ```text
   C:\Corner-sim\.venv\Scripts\python.exe -u -X faulthandler C:\Corner-sim\diagnostics\run_gui_live.py --ticks 5 --capture-interval 1 --log C:\Corner-sim\diagnostic-logs\windbg-live.jsonl
   ```

2. In the WinDbg command window, capture a transcript and break at the C runtime
   abort (works even without Qt private symbols):

   ```text
   .logopen /t C:\Corner-sim\diagnostic-logs\windbg-native.txt
   .symfix
   .reload
   sxe av
   bu ucrtbase!abort
   g
   ```

3. When it breaks, do not continue. Record all native threads and loaded Qt/OpenCV
   modules:

   ```text
   ~* kp
   lm m Qt5Core
   lm m opencv*
   .ecxr
   kp
   .dump /ma C:\Corner-sim\diagnostic-logs\qt-abort.dmp
   .logclose
   ```

   If Qt symbols are available, `bm Qt5Core!*fatal*` before `g` can stop closer to
   the diagnostic. Keep the `ucrtbase!abort` breakpoint as the reliable fallback.

4. Alternatively, launch under Sysinternals ProcDump:

   ```powershell
   procdump.exe -accepteula -ma -e 1 -t -x diagnostic-logs\dumps `
     .\.venv\Scripts\python.exe -u -X faulthandler diagnostics\run_gui_live.py --ticks 5
   ```

   Then run experiment C normally and open the resulting dump in WinDbg. The key
   evidence is the native thread containing `QObject::~QObject` and the module/frame
   that invoked that destructor—not the contemporaneous Python line.

## Current evidence and open questions

**Observed:** the user-provided Windows run creates the `QSocketNotifier` on the GUI
thread, then aborts with a cross-thread QObject timer diagnostic while the Python GUI
thread remains in `app.exec()` and a worker is near runner return.

**Hypotheses to test:** a QObject is created by a native dependency; an OpenCV Qt
backend is initialized in an unexpected thread; or a Qt object remains reachable from
a native/CARLA callback. The existing traceback alone supports none of these as a
root-cause conclusion.

**Unresolved:** the identity, creator thread, owner thread, and destroying thread of
the QObject; the complete B/D exit-code results; and whether the failure requires
OpenCV computation, multiprocessing, or merely importing `cv2` beside PyQt. Wait for
the C0/E matrix and native stack before changing production ownership again.

## Evidence received from Windows run 20260926-214245

This evidence changes the working boundary, but does not yet identify the native
QObject:

- **A passed:** five synchronized live frames were captured without PyQt or OpenCV;
  every owned actor was destroyed, Traffic Manager was set asynchronous, and world
  settings were restored to asynchronous mode with no fixed delta.
- **C1 reached `runner.return`:** live GUI natural completion captured five frames,
  completed actor/settings cleanup, completed the exact `Path.resolve` line, and
  delivered `qt.simulation_finished` on the main Qt thread. The supplied tail has no
  later idle/notifier/window/process-exit checkpoint.
- **C2 passed:** Stop completed the same cleanup, delivered completion on the main Qt
  thread, destroyed the notifier and window on that thread, returned event-loop code
  zero, and exited with `cv2_loaded=false`.

Therefore `Path.resolve`, capture cleanup, and notification of simulation completion
are not the failing operation in that run. The first production-only action after the
last C1 checkpoint is `_start_postprocessing`; C2 deliberately skips that action. This
supports post-processing as the next isolation boundary, **not** a conclusion that
OpenCV, `ProcessPoolExecutor`, or a specific QObject is the root cause. The new
`postprocess.transition.*`, `postprocess.import.*`, and `postprocess.worker.*`
checkpoints plus experiments C0/E distinguish those cases without changing production
behavior.
