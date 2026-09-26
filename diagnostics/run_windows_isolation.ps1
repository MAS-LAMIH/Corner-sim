param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$LogRoot = ".\diagnostic-logs",
    [switch]$RunLive,
    [switch]$ExerciseOpenCvGui
)

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDirectory = New-Item -ItemType Directory -Force -Path (Join-Path $LogRoot $stamp)
$pythonPath = (Resolve-Path $Python).Path

function Run-Probe {
    param([string]$Name, [string[]]$Arguments)
    $stdout = Join-Path $runDirectory "$Name.stdout.log"
    $stderr = Join-Path $runDirectory "$Name.stderr.log"
    $jsonl = Join-Path $runDirectory "$Name.checkpoints.jsonl"
    $allArguments = @("-u", "-X", "faulthandler") + $Arguments + @("--log", $jsonl)
    Write-Host "=== $Name ==="
    Write-Host "$pythonPath $($allArguments -join ' ')"
    $process = Start-Process -FilePath $pythonPath -ArgumentList $allArguments `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    "$Name`t$($process.ExitCode)" | Add-Content (Join-Path $runDirectory "exit-codes.tsv")
    Write-Host "exit=$($process.ExitCode); stdout=$stdout; stderr=$stderr; checkpoints=$jsonl"
}

# D: Every invocation is a fresh process. The OpenCV native GUI backend is an
# explicit separate probe because importing/using cvtColor does not initialize it.
Run-Probe "D1-pyqt" @("diagnostics\import_probe.py", "pyqt")
Run-Probe "D2-opencv-compute" @("diagnostics\import_probe.py", "opencv")
Run-Probe "D3-carla-client" @("diagnostics\import_probe.py", "carla")
Run-Probe "D4-combined" @("diagnostics\import_probe.py", "combined")
if ($ExerciseOpenCvGui) {
    Run-Probe "D5-opencv-native-gui" @(
        "diagnostics\import_probe.py", "opencv", "--exercise-opencv-gui")
}

# B: Actual MainWindow.start_scenario(), but its runner and CARLA connection are fakes.
Run-Probe "B1-gui-fake-natural" @(
    "diagnostics\run_gui_fake.py", "--mode", "natural", "--duration", "2")
Run-Probe "B2-gui-fake-stop" @(
    "diagnostics\run_gui_fake.py", "--mode", "stop", "--duration", "5", "--stop-after", "0.5")
Run-Probe "B3-gui-fake-repeat" @(
    "diagnostics\run_gui_fake.py", "--mode", "repeat", "--duration", "1")

if ($RunLive) {
    Run-Probe "D6-carla-connected" @(
        "diagnostics\import_probe.py", "carla", "--connect-carla")
    Run-Probe "D7-combined-connected" @(
        "diagnostics\import_probe.py", "combined", "--connect-carla")

    # A: Production runner and lifecycle, no QApplication/new_ui/PyQt import.
    Run-Probe "A-standalone-live" @(
        "diagnostics\run_standalone.py", "--ticks", "5", "--capture-interval", "1",
        "--output", (Join-Path $runDirectory "standalone-dataset"))

    # E: use identical copies of A's completed dataset to isolate import,
    # post-processing, Python-thread, and Qt-interaction effects.
    Run-Probe "E1-post-standalone-main" @(
        "diagnostics\run_postprocess.py", "standalone-main",
        "--dataset", (Join-Path $runDirectory "standalone-dataset"),
        "--copy-to", (Join-Path $runDirectory "post-standalone-main-dataset"))
    Run-Probe "E2-post-standalone-thread" @(
        "diagnostics\run_postprocess.py", "standalone-thread",
        "--dataset", (Join-Path $runDirectory "standalone-dataset"),
        "--copy-to", (Join-Path $runDirectory "post-standalone-thread-dataset"))
    Run-Probe "E3-post-qt-import" @(
        "diagnostics\run_postprocess.py", "qt-import")
    Run-Probe "E4-post-qt-thread" @(
        "diagnostics\run_postprocess.py", "qt-thread",
        "--dataset", (Join-Path $runDirectory "standalone-dataset"),
        "--copy-to", (Join-Path $runDirectory "post-qt-thread-dataset"))

    # C0 controls for natural completion without real post-processing. C1 then
    # changes only post-processing back to the production implementation.
    Run-Probe "C0-gui-live-natural-no-post" @(
        "diagnostics\run_gui_live.py", "--ticks", "5", "--capture-interval", "1",
        "--postprocessing", "skip",
        "--output", (Join-Path $runDirectory "gui-live-natural-no-post-dataset"))
    Run-Probe "C1-gui-live-natural" @(
        "diagnostics\run_gui_live.py", "--ticks", "5", "--capture-interval", "1",
        "--output", (Join-Path $runDirectory "gui-live-natural-dataset"))
    Run-Probe "C2-gui-live-stop" @(
        "diagnostics\run_gui_live.py", "--ticks", "100", "--capture-interval", "10",
        "--stop-after", "3", "--output", (Join-Path $runDirectory "gui-live-stop-dataset"))
}

Write-Host "All logs are isolated under $($runDirectory.FullName)"
