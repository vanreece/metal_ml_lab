# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 008: IOReport vs powermetrics GPU power cross-check.

Same staircase workload as exp 007, but instead of comparing
ioreg utilization to powermetrics active residency, this experiment
compares notes/ioreport.py's IOReport-derived GPU power (no sudo)
against powermetrics' GPU power (sudo, from gpu_telemetry.py).

The user runs gpu_telemetry.py in a separate terminal recording at
1000 ms cadence. This script launches notes/ioreport.py as a
subprocess at the same cadence, runs the staircase, then terminates
the subprocess. analysis.py joins all three CSVs by monotonic_ns.

Usage:
  # Terminal A (user, sudo):
  sudo uv run notes/gpu_telemetry.py \\
    --csv experiments/008-ioreport-vs-powermetrics-power/raw/PMTELEM.csv \\
    --interval-ms 1000 --quiet

  # Terminal B (no sudo):
  uv run experiments/008-ioreport-vs-powermetrics-power/run.py \\
    --powermetrics-csv experiments/008-ioreport-vs-powermetrics-power/raw/PMTELEM.csv

Per CLAUDE.md / 003+004+005 conventions: no warmup, no retries, no
averaging in live output.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import Metal
import objc


KERNEL_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

constant int FMA_ITERS = {iters};

kernel void fma_loop(device float *out [[buffer(0)]],
                     uint tid [[thread_position_in_grid]]) {{
    float x = float(tid) * 0.0001f + 1.0f;
    float y = 1.0f;
    for (int i = 0; i < FMA_ITERS; i++) {{
        y = fma(y, x, x);
    }}
    out[tid] = y;
}}
"""

# Same as exp 007 -- known long-enough kernel from exp 004 M4 Max
WORKLOAD_FMA_ITERS = 65536
WORKLOAD_THREADS = 32
WORKLOAD_THREADGROUP = 32

STAIRCASE_STEPS = [0.00, 0.25, 0.50, 0.75, 1.00]
STEP_DURATION_S = 8.0
BASELINE_S = 10.0
TAIL_S = 10.0

TELEMETRY_INTERVAL_MS = 1000   # both ioreport.py and powermetrics

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
IOREPORT_SCRIPT = PROJECT_ROOT / "notes" / "ioreport.py"


def set_user_interactive_qos() -> str:
    QOS_CLASS_USER_INTERACTIVE = 0x21
    libsystem = ctypes.CDLL("/usr/lib/libSystem.dylib")
    rc = libsystem.pthread_set_qos_class_self_np(
        ctypes.c_int(QOS_CLASS_USER_INTERACTIVE), ctypes.c_int(0)
    )
    return f"pthread_set_qos_class_self_np -> {rc}"


def power_source() -> str:
    out = subprocess.run(
        ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip().splitlines()[0] if out.stdout else "<unavailable>"


def build_pipeline(device):
    options = Metal.MTLCompileOptions.alloc().init()
    src = KERNEL_TEMPLATE.format(iters=WORKLOAD_FMA_ITERS)
    library, err = device.newLibraryWithSource_options_error_(src, options, None)
    if library is None:
        raise RuntimeError(f"compile failed: {err}")
    fn = library.newFunctionWithName_("fma_loop")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"pipeline create failed: {err}")
    return pipeline


def dispatch_one(queue, pipeline, out_buffer):
    cb = queue.commandBuffer()
    encoder = cb.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(WORKLOAD_THREADS, 1, 1),
        Metal.MTLSizeMake(WORKLOAD_THREADGROUP, 1, 1),
    )
    encoder.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()


def run_step(queue, pipeline, out_buffer, target_busy_fraction: float,
             duration_s: float, dispatch_writer, step_label: str) -> int:
    end_t = time.monotonic() + duration_s
    issued = 0
    if target_busy_fraction <= 0:
        time.sleep(duration_s)
        return 0
    expected_dispatch_s = 0.001  # adjusted from 832us per exp 007 observed
    if target_busy_fraction >= 1.0:
        inter_sleep_s = 0.0
    else:
        dispatch_period_s = expected_dispatch_s / target_busy_fraction
        inter_sleep_s = max(0.0, dispatch_period_s - expected_dispatch_s)
    while time.monotonic() < end_t:
        dispatch_writer.writerow([
            time.monotonic_ns(), step_label, f"{target_busy_fraction:.2f}", issued
        ])
        dispatch_one(queue, pipeline, out_buffer)
        issued += 1
        if inter_sleep_s > 0:
            time.sleep(inter_sleep_s)
    return issued


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--powermetrics-csv", type=str, required=True,
                    help="path to a powermetrics CSV that the user is "
                         "ALREADY recording in another terminal")
    args = ap.parse_args()

    pm_path = Path(args.powermetrics_csv)
    if not pm_path.exists():
        print(f"ERROR: powermetrics CSV not found at {pm_path}", file=sys.stderr)
        print("Start it in another terminal first:", file=sys.stderr)
        print(f"  sudo uv run notes/gpu_telemetry.py --csv {pm_path} "
              f"--interval-ms {TELEMETRY_INTERVAL_MS} --quiet", file=sys.stderr)
        return 2
    with open(pm_path) as f:
        rows = sum(1 for _ in f)
    if rows < 2:
        print(f"ERROR: powermetrics CSV {pm_path} has only {rows} row(s); "
              "give it a few seconds to start writing samples", file=sys.stderr)
        return 2

    if not IOREPORT_SCRIPT.exists():
        print(f"ERROR: notes/ioreport.py not found at {IOREPORT_SCRIPT}",
              file=sys.stderr)
        return 2

    qos_result = set_user_interactive_qos()
    pwr = power_source()
    device = Metal.MTLCreateSystemDefaultDevice()
    arch = device.architecture().name() if (
        hasattr(device, "architecture") and device.architecture()
    ) else "<unavailable>"
    print(f"device: {device.name()}  arch: {arch}")
    print(f"OS: {platform.platform()}")
    print(f"QoS: {qos_result}")
    print(f"power: {pwr}")
    print(f"powermetrics CSV: {pm_path} ({rows} rows visible at start)")

    pipeline = build_pipeline(device)
    queue = device.newCommandQueue()
    out_buffer = device.newBufferWithLength_options_(
        4 * WORKLOAD_THREADS, Metal.MTLResourceStorageModeShared
    )

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    ioreport_csv = RAW_DIR / f"{ts}-ioreport.csv"
    phases_path = RAW_DIR / f"{ts}-phases.csv"
    dispatches_path = RAW_DIR / f"{ts}-dispatches.csv"
    meta_path = RAW_DIR / f"{ts}-meta.txt"
    ioreport_log = RAW_DIR / f"{ts}-ioreport-stdout.log"

    # Start ioreport.py subprocess
    ioreport_proc = subprocess.Popen(
        ["uv", "run", str(IOREPORT_SCRIPT),
         "--interval-ms", str(TELEMETRY_INTERVAL_MS),
         "--csv", str(ioreport_csv)],
        stdout=open(ioreport_log, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"ioreport.py launched pid={ioreport_proc.pid} -> {ioreport_csv.name}")
    # Give it a moment to start sampling
    time.sleep(1.5)
    if ioreport_proc.poll() is not None:
        print(f"ERROR: ioreport.py exited rc={ioreport_proc.returncode} immediately;"
              f" see {ioreport_log}", file=sys.stderr)
        return 3
    if not ioreport_csv.exists():
        print(f"ERROR: ioreport.py did not create CSV {ioreport_csv}",
              file=sys.stderr)
        ioreport_proc.terminate()
        return 3

    phases_f = open(phases_path, "w", newline="")
    phases_writer = csv.writer(phases_f)
    phases_writer.writerow(["monotonic_ns", "phase", "target_busy_fraction"])
    dispatches_f = open(dispatches_path, "w", newline="")
    dispatches_writer = csv.writer(dispatches_f)
    dispatches_writer.writerow([
        "monotonic_ns", "step_label", "target_busy_fraction", "dispatch_idx_within_step"
    ])

    run_start_ns = time.monotonic_ns()
    print()
    print(f"=== phase 0: baseline (idle) {BASELINE_S:.0f}s ===")
    phases_writer.writerow([time.monotonic_ns(), "baseline", "0.00"])
    phases_f.flush()
    time.sleep(BASELINE_S)

    print(f"\n=== phase 1: utilization staircase ({len(STAIRCASE_STEPS)} steps "
          f"x {STEP_DURATION_S:.0f}s) ===")
    for fraction in STAIRCASE_STEPS:
        label = f"step_{int(fraction * 100):03d}pct"
        print(f"  {label}  target_busy_fraction={fraction:.2f}")
        phases_writer.writerow([time.monotonic_ns(), label, f"{fraction:.2f}"])
        phases_f.flush()
        n = run_step(queue, pipeline, out_buffer, fraction, STEP_DURATION_S,
                     dispatches_writer, label)
        dispatches_f.flush()
        print(f"    -> {n} dispatches")

    print(f"\n=== phase 2: tail (idle) {TAIL_S:.0f}s ===")
    phases_writer.writerow([time.monotonic_ns(), "tail", "0.00"])
    phases_f.flush()
    time.sleep(TAIL_S)

    phases_writer.writerow([time.monotonic_ns(), "stopped", "0.00"])
    phases_f.flush()
    phases_f.close()
    dispatches_f.close()

    # Stop ioreport.py cleanly so it flushes its CSV
    print()
    print(f"stopping ioreport.py (SIGINT) ...")
    ioreport_proc.send_signal(signal.SIGINT)
    try:
        ioreport_proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        print("  ioreport.py did not exit in 5s, sending SIGTERM")
        ioreport_proc.terminate()
        ioreport_proc.wait(timeout=3.0)
    run_end_ns = time.monotonic_ns()
    print(f"ioreport.py exited rc={ioreport_proc.returncode}")

    # Metadata
    meta_lines = [
        f"experiment: 008-ioreport-vs-powermetrics-power",
        f"timestamp: {ts}",
        f"device: {device.name()}",
        f"architecture: {arch}",
        f"chip: {platform.processor()}  machine: {platform.machine()}",
        f"os: {platform.platform()}",
        f"python: {sys.version.splitlines()[0]}",
        f"pyobjc: {objc.__version__}",
        f"qos: {qos_result}",
        f"power: {pwr}",
        f"powermetrics_csv: {pm_path}",
        f"ioreport_csv: {ioreport_csv.name}",
        f"phases_csv: {phases_path.name}",
        f"dispatches_csv: {dispatches_path.name}",
        f"workload_kernel: fma_loop iters={WORKLOAD_FMA_ITERS} threads={WORKLOAD_THREADS}",
        f"staircase_steps: {STAIRCASE_STEPS}",
        f"step_duration_s: {STEP_DURATION_S}",
        f"baseline_s: {BASELINE_S}",
        f"tail_s: {TAIL_S}",
        f"telemetry_interval_ms: {TELEMETRY_INTERVAL_MS}",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {(run_end_ns - run_start_ns) / 1e9:.2f}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")

    print()
    print("Done.")
    print(f"NOW: stop the powermetrics collector (Ctrl-C the other terminal)")
    print(f"THEN: uv run experiments/008-.../analysis.py "
          f"--powermetrics-csv {pm_path} --prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
