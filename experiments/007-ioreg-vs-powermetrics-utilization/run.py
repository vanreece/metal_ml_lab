# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 007: ioreg utilization vs powermetrics active residency.

Runs a known GPU workload (utilization staircase from idle through
100% busy and back) while sampling ioreg's
AGXAccelerator.PerformanceStatistics every 100 ms in a separate
thread. The user runs powermetrics in a separate terminal (sudo)
recording at the same 100 ms cadence to a CSV; analysis.py joins the
two by monotonic_ns and reports per-sample agreement.

Pre-registration: experiments/007-ioreg-vs-powermetrics-utilization/README.md.

Usage:
  # In terminal A (user, sudo):
  sudo uv run notes/gpu_telemetry.py \\
    --csv experiments/007-ioreg-vs-powermetrics-utilization/raw/PMTELEM.csv \\
    --interval-ms 100 --quiet

  # In terminal B (no sudo):
  uv run experiments/007-ioreg-vs-powermetrics-utilization/run.py \\
    --powermetrics-csv experiments/007-ioreg-vs-powermetrics-utilization/raw/PMTELEM.csv

Per CLAUDE.md / 003+004+005 conventions: no warmup, no retries, no
averaging. Raw values to CSV. If something fails, it fails loudly.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import Metal
import objc


# ----------------------------------------------------------------------
# Workload kernel — same fma_loop pattern as exp 004 / 005.

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

# 65536 iters at 32 threads -> p50 ~832 us per dispatch (per exp 004
# M4 Max addendum). Long enough that the GPU is "obviously busy"
# during a dispatch, short enough that 100% busy is achievable with
# back-to-back dispatches plus modest idle gaps for pacing.
WORKLOAD_FMA_ITERS = 65536
WORKLOAD_THREADS = 32
WORKLOAD_THREADGROUP = 32

# Staircase: each step is 8 s of dispatches with the target busy
# fraction, paced by Python sleep between dispatches. Total Phase 1 =
# 5 steps * 8 s = 40 s.
STAIRCASE_STEPS = [0.00, 0.25, 0.50, 0.75, 1.00]
STEP_DURATION_S = 8.0
BASELINE_S = 10.0
TAIL_S = 10.0

# ioreg sampler cadence (matches powermetrics interval expected from
# the user-side gpu_telemetry.py invocation).
IOREG_SAMPLE_INTERVAL_S = 0.100

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"


# ----------------------------------------------------------------------
# Environment

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


# ----------------------------------------------------------------------
# ioreg sampler — runs in a separate thread.

# Match the 'PerformanceStatistics' property to filter to the AGX node
# regardless of chip-family-specific class name (G13X, G16X, etc.).
IOREG_CMD = ["ioreg", "-rl", "-k", "PerformanceStatistics", "-d", "5"]

DEVICE_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
RENDERER_UTIL_RE = re.compile(r'"Renderer Utilization %"\s*=\s*(\d+)')
TILER_UTIL_RE = re.compile(r'"Tiler Utilization %"\s*=\s*(\d+)')
INUSE_MEM_RE = re.compile(r'"In use system memory"\s*=\s*(\d+)')
RECOVERY_COUNT_RE = re.compile(r'"recoveryCount"\s*=\s*(\d+)')


def parse_ioreg(text: str) -> dict:
    """Pull the four PerformanceStatistics fields we care about. Returns
    empty values if a field is missing rather than raising."""
    out = {}
    for key, regex in [
        ("device_util_pct", DEVICE_UTIL_RE),
        ("renderer_util_pct", RENDERER_UTIL_RE),
        ("tiler_util_pct", TILER_UTIL_RE),
        ("in_use_system_memory_bytes", INUSE_MEM_RE),
        ("recovery_count", RECOVERY_COUNT_RE),
    ]:
        m = regex.search(text)
        out[key] = m.group(1) if m else ""
    return out


def ioreg_sampler_thread(out_path: Path, stop_event: threading.Event) -> None:
    """Sample ioreg every IOREG_SAMPLE_INTERVAL_S until stop_event is set.
    Append rows to out_path with monotonic_ns + parsed fields."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iso_ts", "monotonic_ns",
            "device_util_pct", "renderer_util_pct", "tiler_util_pct",
            "in_use_system_memory_bytes", "recovery_count",
            "ioreg_subprocess_ms",
        ])
        next_due = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_due:
                # Sleep up to the next due time, but wake up promptly
                # if the stop_event is set.
                if stop_event.wait(timeout=next_due - now):
                    break
            t0 = time.monotonic_ns()
            try:
                proc = subprocess.run(IOREG_CMD, capture_output=True, text=True,
                                      check=False, timeout=2.0)
                t1 = time.monotonic_ns()
                fields = parse_ioreg(proc.stdout)
            except subprocess.TimeoutExpired:
                t1 = time.monotonic_ns()
                fields = {k: "" for k in [
                    "device_util_pct", "renderer_util_pct", "tiler_util_pct",
                    "in_use_system_memory_bytes", "recovery_count",
                ]}
            w.writerow([
                dt.datetime.now().isoformat(timespec="microseconds"),
                t0,
                fields.get("device_util_pct", ""),
                fields.get("renderer_util_pct", ""),
                fields.get("tiler_util_pct", ""),
                fields.get("in_use_system_memory_bytes", ""),
                fields.get("recovery_count", ""),
                round((t1 - t0) / 1e6, 3),
            ])
            f.flush()
            next_due += IOREG_SAMPLE_INTERVAL_S


# ----------------------------------------------------------------------
# Workload

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
    """Issue dispatches at a rate that targets `target_busy_fraction` of the
    duration window busy, by adjusting inter-dispatch sleep. Returns count
    of dispatches issued."""
    end_t = time.monotonic() + duration_s
    issued = 0
    if target_busy_fraction <= 0:
        # Just sleep for duration_s with no GPU work.
        time.sleep(duration_s)
        return 0
    # Empirically the dispatch p50 is ~832 us; assume that's the per-
    # dispatch GPU time and pace inter-dispatch sleep accordingly. If
    # target=1.0, sleep=0 (back-to-back).
    expected_dispatch_s = 0.000832
    if target_busy_fraction >= 1.0:
        inter_sleep_s = 0.0
    else:
        # busy_time + idle_time = dispatch_period
        # busy_time = dispatch_period * target_busy_fraction
        # so dispatch_period = expected_dispatch_s / target_busy_fraction
        # and idle_time = dispatch_period - expected_dispatch_s
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


# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--powermetrics-csv", type=str, required=True,
                    help="path to a powermetrics CSV that the user is "
                         "ALREADY recording in another terminal via "
                         "`sudo uv run notes/gpu_telemetry.py --csv ...`. "
                         "This script verifies the file exists with >= 2 "
                         "rows before starting the workload.")
    args = ap.parse_args()

    pm_path = Path(args.powermetrics_csv)
    if not pm_path.exists():
        print(f"ERROR: powermetrics CSV not found at {pm_path}", file=sys.stderr)
        print("Start it in another terminal first:", file=sys.stderr)
        print(f"  sudo uv run notes/gpu_telemetry.py --csv {pm_path} "
              "--interval-ms 100 --quiet", file=sys.stderr)
        return 2
    with open(pm_path) as f:
        rows = sum(1 for _ in f)
    if rows < 2:
        print(f"ERROR: powermetrics CSV {pm_path} has only {rows} row(s); "
              "give it a few seconds to start writing samples", file=sys.stderr)
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
    ioreg_path = RAW_DIR / f"{ts}-ioreg.csv"
    phases_path = RAW_DIR / f"{ts}-phases.csv"
    dispatches_path = RAW_DIR / f"{ts}-dispatches.csv"
    meta_path = RAW_DIR / f"{ts}-meta.txt"

    # Start ioreg sampler
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=ioreg_sampler_thread, args=(ioreg_path, stop_event), daemon=True
    )
    sampler.start()
    print(f"ioreg sampler started -> {ioreg_path.name}")

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

    # Stop ioreg sampler
    stop_event.set()
    sampler.join(timeout=5.0)
    run_end_ns = time.monotonic_ns()
    print(f"\nioreg sampler stopped after {(run_end_ns - run_start_ns) / 1e9:.1f}s")

    # Metadata
    meta_lines = [
        f"experiment: 007-ioreg-vs-powermetrics-utilization",
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
        f"workload_kernel: fma_loop iters={WORKLOAD_FMA_ITERS} threads={WORKLOAD_THREADS}",
        f"staircase_steps: {STAIRCASE_STEPS}",
        f"step_duration_s: {STEP_DURATION_S}",
        f"baseline_s: {BASELINE_S}",
        f"tail_s: {TAIL_S}",
        f"ioreg_sample_interval_s: {IOREG_SAMPLE_INTERVAL_S}",
        f"ioreg_csv: {ioreg_path.name}",
        f"phases_csv: {phases_path.name}",
        f"dispatches_csv: {dispatches_path.name}",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {(run_end_ns - run_start_ns) / 1e9:.2f}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")

    print()
    print("Done.")
    print(f"NOW: stop the powermetrics collector (Ctrl-C the other terminal)")
    print(f"THEN: uv run experiments/007-.../analysis.py "
          f"--powermetrics-csv {pm_path} --prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
