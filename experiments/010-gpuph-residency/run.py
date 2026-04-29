# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 010: GPUPH state residency face-validity test.

Drives the GPU through known phases (idle baseline, 5-step staircase,
tail, 009-style sub-floor recipe) while notes/ioreport.py runs as a
subprocess writing per-state residency to a sibling -states.csv. The
analysis (separate script) checks whether GPUPH residency monotonically
shifts to higher state indices with workload intensity, and whether the
sub-floor phase pins residency in the top state.

Per CLAUDE.md / 003-009 conventions: no warmup beyond explicit setup,
no retries, no averaging in live output.
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

WRITE_TID_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void write_tid(device uint *out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
"""

# Same staircase kernel as exp 008
STAIRCASE_FMA_ITERS = 65536
STAIRCASE_THREADS = 32

# Sub-floor recipe matches exp 009 (FMA_ITERS=1024, K=20)
SUBFLOOR_FMA_ITERS = 1024
SUBFLOOR_K = 20

STAIRCASE_STEPS = [0.00, 0.25, 0.50, 0.75, 1.00]
STEP_DURATION_S = 8.0
BASELINE_S = 10.0
TAIL_S = 10.0
SUBFLOOR_DURATION_S = 50.0

TELEMETRY_INTERVAL_MS = 1000

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


def build_pipeline(device, source: str, fn_name: str):
    options = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None:
        raise RuntimeError(f"compile failed for {fn_name}: {err}")
    fn = library.newFunctionWithName_(fn_name)
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"pipeline create failed for {fn_name}: {err}")
    return pipeline


def dispatch_one(queue, pipeline, out_buffer, threads, group):
    cb = queue.commandBuffer()
    encoder = cb.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(threads, 1, 1),
        Metal.MTLSizeMake(group, 1, 1),
    )
    encoder.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()


def run_staircase_step(queue, pipeline, out_buffer, target_busy: float,
                       duration_s: float):
    """Same duty-cycle approach as exp 008."""
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
        dispatch_one(queue, pipeline, out_buffer,
                     STAIRCASE_THREADS, STAIRCASE_THREADS)
        issued += 1
        if inter_sleep_s > 0:
            time.sleep(inter_sleep_s)
    return issued


def run_subfloor(queue, fma_pipe, write_pipe, out_buffer, duration_s: float):
    """Loop the exp 009 recipe (20 fma_loop K=1024 warmup, 1 write_tid 32t
    measured) for `duration_s`. We don't need timestamps here -- ioreport.py
    is the data path. Just keep the GPU saturated under the same kernel
    pattern that pushes it into peak DVFS per exp 009."""
    end_t = time.monotonic() + duration_s
    issued = 0
    while time.monotonic() < end_t:
        for _ in range(SUBFLOOR_K):
            dispatch_one(queue, fma_pipe, out_buffer,
                         STAIRCASE_THREADS, STAIRCASE_THREADS)
        dispatch_one(queue, write_pipe, out_buffer,
                     STAIRCASE_THREADS, STAIRCASE_THREADS)
        issued += 1
    return issued


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    if not IOREPORT_SCRIPT.exists():
        print(f"ERROR: notes/ioreport.py not found at {IOREPORT_SCRIPT}",
              file=sys.stderr)
        return 2

    RAW_DIR.mkdir(exist_ok=True)
    qos_result = set_user_interactive_qos()
    pwr = power_source()

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("MTLCreateSystemDefaultDevice returned nil")
    arch = device.architecture().name() if (
        hasattr(device, "architecture") and device.architecture()
    ) else "<unavailable>"

    print("=" * 78)
    print("Experiment 010: GPUPH state residency face validity")
    print("=" * 78)
    print(f"device: {device.name()}  arch: {arch}")
    print(f"OS:     {platform.platform()}")
    print(f"QoS:    {qos_result}")
    print(f"power:  {pwr}")

    staircase_pipe = build_pipeline(
        device,
        KERNEL_TEMPLATE.format(iters=STAIRCASE_FMA_ITERS),
        "fma_loop",
    )
    subfloor_fma_pipe = build_pipeline(
        device,
        KERNEL_TEMPLATE.format(iters=SUBFLOOR_FMA_ITERS),
        "fma_loop",
    )
    subfloor_write_pipe = build_pipeline(device, WRITE_TID_SOURCE, "write_tid")
    queue = device.newCommandQueue()
    out_buffer = device.newBufferWithLength_options_(
        4 * 1024, Metal.MTLResourceStorageModeShared
    )

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    energy_csv = RAW_DIR / f"{ts}.csv"
    states_csv = RAW_DIR / f"{ts}-states.csv"
    phases_csv = RAW_DIR / f"{ts}-phases.csv"
    meta_path = RAW_DIR / f"{ts}-meta.txt"
    iolog = RAW_DIR / f"{ts}-ioreport-stdout.log"

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
        print(f"ERROR: ioreport.py exited rc={ioreport_proc.returncode}; "
              f"see {iolog}", file=sys.stderr)
        return 3
    if not energy_csv.exists():
        print(f"ERROR: ioreport.py did not create {energy_csv}", file=sys.stderr)
        ioreport_proc.terminate()
        return 3
    print(f"ioreport.py: pid={ioreport_proc.pid} -> {energy_csv.name} "
          f"+ {states_csv.name}")

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

        print(f"\n=== phase 1: staircase ({len(STAIRCASE_STEPS)} steps "
              f"x {STEP_DURATION_S:.0f}s) ===")
        for fraction in STAIRCASE_STEPS:
            label = f"step_{int(fraction * 100):03d}pct"
            print(f"  {label}  target_busy={fraction:.2f}")
            phases_w.writerow([time.monotonic_ns(), label, f"{fraction:.2f}"])
            phases_f.flush()
            n = run_staircase_step(queue, staircase_pipe, out_buffer,
                                   fraction, STEP_DURATION_S)
            print(f"    -> {n} dispatches")

        print(f"\n=== phase 2: tail (idle) {TAIL_S:.0f}s ===")
        phases_w.writerow([time.monotonic_ns(), "tail", "0.00"])
        phases_f.flush()
        time.sleep(TAIL_S)

        print(f"\n=== phase 3: sub-floor recipe {SUBFLOOR_DURATION_S:.0f}s "
              f"(fma_loop K={SUBFLOOR_K} sleep_0, exp 009 pattern) ===")
        phases_w.writerow([time.monotonic_ns(), "subfloor", "1.00"])
        phases_f.flush()
        n_sub = run_subfloor(queue, subfloor_fma_pipe, subfloor_write_pipe,
                             out_buffer, SUBFLOOR_DURATION_S)
        print(f"    -> {n_sub} (warmup x K + measured) trials")

        phases_w.writerow([time.monotonic_ns(), "stopped", "0.00"])
        phases_f.flush()
    finally:
        phases_f.close()
        run_end_ns = time.monotonic_ns()
        print()
        print("stopping ioreport.py (SIGINT) ...")
        ioreport_proc.send_signal(signal.SIGINT)
        try:
            ioreport_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            ioreport_proc.terminate()
            ioreport_proc.wait(timeout=3.0)
        print(f"ioreport.py rc={ioreport_proc.returncode}")

    elapsed_s = (run_end_ns - run_start_ns) / 1e9
    meta_lines = [
        "experiment: 010-gpuph-residency",
        f"timestamp: {ts}",
        f"device: {device.name()}",
        f"architecture: {arch}",
        f"os: {platform.platform()}",
        f"python: {sys.version.splitlines()[0]}",
        f"pyobjc: {objc.__version__}",
        f"qos: {qos_result}",
        f"power: {pwr}",
        f"energy_csv: {energy_csv.name}",
        f"states_csv: {states_csv.name}",
        f"phases_csv: {phases_csv.name}",
        f"telemetry_interval_ms: {TELEMETRY_INTERVAL_MS}",
        f"baseline_s: {BASELINE_S}",
        f"staircase_steps: {STAIRCASE_STEPS}",
        f"step_duration_s: {STEP_DURATION_S}",
        f"tail_s: {TAIL_S}",
        f"subfloor_s: {SUBFLOOR_DURATION_S}",
        f"staircase_kernel: fma_loop iters={STAIRCASE_FMA_ITERS} threads={STAIRCASE_THREADS}",
        f"subfloor_kernel: fma_loop iters={SUBFLOOR_FMA_ITERS} threads={STAIRCASE_THREADS} K={SUBFLOOR_K}",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {elapsed_s:.2f}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")
    print(f"\nNow run: uv run experiments/010-gpuph-residency/analysis.py "
          f"--prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
