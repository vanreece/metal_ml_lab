# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 012: GPUPH vs powermetrics per-MHz residency cross-validation.

Same staircase shape as exp 008/010. Two data paths run in parallel:

  - User starts powermetrics raw capture in another terminal:
      sudo powermetrics --samplers gpu_power -i 250 \
          > experiments/012-.../raw/PMRAW.txt 2>&1
    (stay running for the duration; Ctrl-C after run.py reports done)

  - This script launches notes/ioreport.py --include-states at 250 ms
    cadence in the same window, capturing GPUPH per-P-state residency.

analysis.py joins them by monotonic_ns and matches per-phase residency
distributions to build a P-state -> MHz mapping.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
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

# Same as exp 008/010
WORKLOAD_FMA_ITERS = 65536
WORKLOAD_THREADS = 32

STAIRCASE_STEPS = [0.00, 0.25, 0.50, 0.75, 1.00]
STEP_DURATION_S = 8.0
BASELINE_S = 10.0
TAIL_S = 10.0

TELEMETRY_INTERVAL_MS = 250

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
        Metal.MTLSizeMake(WORKLOAD_THREADS, 1, 1),
    )
    encoder.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()


def run_step(queue, pipeline, out_buffer, target_busy: float, duration_s: float):
    end_t = time.monotonic() + duration_s
    issued = 0
    if target_busy <= 0:
        time.sleep(duration_s)
        return 0
    expected_dispatch_s = 0.001
    if target_busy >= 1.0:
        inter_sleep_s = 0.0
    else:
        period_s = expected_dispatch_s / target_busy
        inter_sleep_s = max(0.0, period_s - expected_dispatch_s)
    while time.monotonic() < end_t:
        dispatch_one(queue, pipeline, out_buffer)
        issued += 1
        if inter_sleep_s > 0:
            time.sleep(inter_sleep_s)
    return issued


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmraw", type=Path,
                    default=RAW_DIR / "PMRAW.txt",
                    help="path to powermetrics raw text capture (the user "
                         "starts this in another terminal before running)")
    args = ap.parse_args()

    if not args.pmraw.exists():
        print(f"ERROR: powermetrics raw capture not found at {args.pmraw}",
              file=sys.stderr)
        print("Start it in another terminal first:", file=sys.stderr)
        print(f"  sudo powermetrics --samplers gpu_power -i {TELEMETRY_INTERVAL_MS} "
              f"> {args.pmraw} 2>&1", file=sys.stderr)
        return 2

    initial_size = args.pmraw.stat().st_size
    if initial_size < 100:
        print(f"WARN: powermetrics file is only {initial_size} bytes; "
              "make sure powermetrics is actively writing", file=sys.stderr)

    if not IOREPORT_SCRIPT.exists():
        print(f"ERROR: ioreport.py not found at {IOREPORT_SCRIPT}", file=sys.stderr)
        return 2

    RAW_DIR.mkdir(exist_ok=True)
    qos_result = set_user_interactive_qos()
    pwr = power_source()

    device = Metal.MTLCreateSystemDefaultDevice()
    arch = device.architecture().name() if (
        hasattr(device, "architecture") and device.architecture()
    ) else "<unavailable>"

    print("=" * 78)
    print("Experiment 012: GPUPH vs powermetrics MHz cross-validation")
    print("=" * 78)
    print(f"device: {device.name()}  arch: {arch}")
    print(f"OS:     {platform.platform()}")
    print(f"QoS:    {qos_result}")
    print(f"power:  {pwr}")
    print(f"powermetrics raw: {args.pmraw} ({initial_size} bytes)")

    pipeline = build_pipeline(device)
    queue = device.newCommandQueue()
    out_buffer = device.newBufferWithLength_options_(
        4 * WORKLOAD_THREADS, Metal.MTLResourceStorageModeShared
    )

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    energy_csv = RAW_DIR / f"{ts}-ioreport.csv"
    states_csv = RAW_DIR / f"{ts}-ioreport-states.csv"
    phases_csv = RAW_DIR / f"{ts}-phases.csv"
    iolog = RAW_DIR / f"{ts}-ioreport-stdout.log"
    meta_path = RAW_DIR / f"{ts}-meta.txt"

    ioreport_proc = subprocess.Popen(
        ["uv", "run", str(IOREPORT_SCRIPT),
         "--include-states",
         "--interval-ms", str(TELEMETRY_INTERVAL_MS),
         "--csv", str(energy_csv)],
        stdout=open(iolog, "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)
    if ioreport_proc.poll() is not None:
        print(f"ERROR: ioreport.py exited rc={ioreport_proc.returncode}",
              file=sys.stderr)
        return 3
    print(f"ioreport.py: pid={ioreport_proc.pid}")

    phases_f = open(phases_csv, "w", newline="")
    phases_w = csv.writer(phases_f)
    phases_w.writerow(["monotonic_ns", "phase", "target_busy_fraction"])

    run_start_ns = time.monotonic_ns()
    try:
        print()
        print(f"=== phase 0: baseline (idle) {BASELINE_S:.0f}s ===")
        phases_w.writerow([time.monotonic_ns(), "baseline", "0.00"])
        phases_f.flush()
        time.sleep(BASELINE_S)

        print(f"\n=== phase 1: staircase ===")
        for fraction in STAIRCASE_STEPS:
            label = f"step_{int(fraction * 100):03d}pct"
            print(f"  {label}  target_busy={fraction:.2f}")
            phases_w.writerow([time.monotonic_ns(), label, f"{fraction:.2f}"])
            phases_f.flush()
            n = run_step(queue, pipeline, out_buffer, fraction, STEP_DURATION_S)
            print(f"    -> {n} dispatches")

        print(f"\n=== phase 2: tail (idle) {TAIL_S:.0f}s ===")
        phases_w.writerow([time.monotonic_ns(), "tail", "0.00"])
        phases_f.flush()
        time.sleep(TAIL_S)

        phases_w.writerow([time.monotonic_ns(), "stopped", "0.00"])
        phases_f.flush()
    finally:
        phases_f.close()
        run_end_ns = time.monotonic_ns()
        print()
        print("stopping ioreport.py (SIGINT)")
        ioreport_proc.send_signal(signal.SIGINT)
        try:
            ioreport_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            ioreport_proc.terminate()
            ioreport_proc.wait(timeout=3.0)
        print(f"ioreport.py rc={ioreport_proc.returncode}")

    final_pmraw_size = args.pmraw.stat().st_size
    elapsed_s = (run_end_ns - run_start_ns) / 1e9

    meta_lines = [
        "experiment: 012-gpuph-vs-powermetrics-mhz",
        f"timestamp: {ts}",
        f"device: {device.name()}",
        f"architecture: {arch}",
        f"os: {platform.platform()}",
        f"python: {sys.version.splitlines()[0]}",
        f"pyobjc: {objc.__version__}",
        f"qos: {qos_result}",
        f"power: {pwr}",
        f"pmraw: {args.pmraw}  initial_size={initial_size}  final_size={final_pmraw_size}",
        f"energy_csv: {energy_csv.name}",
        f"states_csv: {states_csv.name}",
        f"phases_csv: {phases_csv.name}",
        f"telemetry_interval_ms: {TELEMETRY_INTERVAL_MS}",
        f"baseline_s: {BASELINE_S}",
        f"staircase_steps: {STAIRCASE_STEPS}",
        f"step_duration_s: {STEP_DURATION_S}",
        f"tail_s: {TAIL_S}",
        f"workload_kernel: fma_loop iters={WORKLOAD_FMA_ITERS} threads={WORKLOAD_THREADS}",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {elapsed_s:.2f}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")
    print()
    print("Done.")
    print(f"NOW: stop powermetrics (Ctrl-C the other terminal)")
    print(f"THEN: uv run experiments/012-gpuph-vs-powermetrics-mhz/analysis.py "
          f"--prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
