# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 013: PWRCTRL recipe boundary sweep.

For each (FMA_ITERS, sleep_us) cell, sustain back-to-back fma_loop
dispatches for 5 s while ioreport.py records GPU Stats per-state
residency. Analysis classifies each cell as DEADLINE / PERF / MIXED
based on PWRCTRL top state.
"""
from __future__ import annotations

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

# 2D sweep
FMA_ITERS_LEVELS = [256, 1024, 4096, 16384, 65536]
SLEEP_US_LEVELS = [0, 50, 200, 1000]
CELL_DURATION_S = 5.0
BASELINE_S = 5.0
TAIL_S = 5.0
THREADS = 32

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


def build_pipeline_for_iters(device, iters: int):
    options = Metal.MTLCompileOptions.alloc().init()
    src = KERNEL_TEMPLATE.format(iters=iters)
    library, err = device.newLibraryWithSource_options_error_(src, options, None)
    if library is None:
        raise RuntimeError(f"compile failed (iters={iters}): {err}")
    fn = library.newFunctionWithName_("fma_loop")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"pipeline create failed (iters={iters}): {err}")
    return pipeline


def dispatch_one(queue, pipeline, out_buffer):
    cb = queue.commandBuffer()
    encoder = cb.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(THREADS, 1, 1),
        Metal.MTLSizeMake(THREADS, 1, 1),
    )
    encoder.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()


def run_cell(queue, pipeline, out_buffer, sleep_us: int, duration_s: float) -> int:
    end_t = time.monotonic() + duration_s
    issued = 0
    sleep_s = sleep_us / 1e6
    while time.monotonic() < end_t:
        dispatch_one(queue, pipeline, out_buffer)
        issued += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
    return issued


def main() -> int:
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
    queue = device.newCommandQueue()
    out_buffer = device.newBufferWithLength_options_(
        4 * 1024, Metal.MTLResourceStorageModeShared
    )

    print("=" * 78)
    print("Experiment 013: PWRCTRL recipe boundary sweep")
    print("=" * 78)
    print(f"device: {device.name()}  arch: {arch}")
    print(f"OS:     {platform.platform()}")
    print(f"power:  {pwr}")

    # Build pipelines for each FMA_ITERS level (compile once)
    pipelines = {}
    for iters in FMA_ITERS_LEVELS:
        pipelines[iters] = build_pipeline_for_iters(device, iters)
    print(f"compiled pipelines: {sorted(pipelines.keys())}")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    energy_csv = RAW_DIR / f"{ts}.csv"
    states_csv = RAW_DIR / f"{ts}-states.csv"
    cells_csv = RAW_DIR / f"{ts}-cells.csv"
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
        print(f"ERROR: ioreport.py exited rc={ioreport_proc.returncode}",
              file=sys.stderr)
        return 3
    print(f"ioreport.py: pid={ioreport_proc.pid}")

    cells_f = open(cells_csv, "w", newline="")
    cells_w = csv.writer(cells_f)
    cells_w.writerow([
        "monotonic_ns_start", "monotonic_ns_end", "cell_idx",
        "fma_iters", "sleep_us", "n_dispatches",
    ])

    run_start_ns = time.monotonic_ns()
    cell_idx = 0
    try:
        print()
        print(f"=== phase 0: baseline (idle) {BASELINE_S:.0f}s ===")
        time.sleep(BASELINE_S)

        print(f"\n=== phase 1: 2D sweep {len(FMA_ITERS_LEVELS)} x "
              f"{len(SLEEP_US_LEVELS)} = {len(FMA_ITERS_LEVELS) * len(SLEEP_US_LEVELS)} cells ===")
        for iters in FMA_ITERS_LEVELS:
            for sleep_us in SLEEP_US_LEVELS:
                start_ns = time.monotonic_ns()
                n = run_cell(queue, pipelines[iters], out_buffer, sleep_us,
                             CELL_DURATION_S)
                end_ns = time.monotonic_ns()
                cells_w.writerow([
                    start_ns, end_ns, cell_idx, iters, sleep_us, n,
                ])
                cells_f.flush()
                print(f"  cell {cell_idx:>2}: FMA_ITERS={iters:>5} "
                      f"sleep_us={sleep_us:>4}  -> {n:>5} dispatches "
                      f"in {(end_ns - start_ns) / 1e9:.2f}s")
                cell_idx += 1

        print(f"\n=== phase 2: tail (idle) {TAIL_S:.0f}s ===")
        time.sleep(TAIL_S)
    finally:
        cells_f.close()
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

    elapsed_s = (run_end_ns - run_start_ns) / 1e9

    meta_lines = [
        "experiment: 013-pwrctrl-recipe-sweep",
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
        f"cells_csv: {cells_csv.name}",
        f"telemetry_interval_ms: {TELEMETRY_INTERVAL_MS}",
        f"baseline_s: {BASELINE_S}",
        f"cell_duration_s: {CELL_DURATION_S}",
        f"tail_s: {TAIL_S}",
        f"fma_iters_levels: {FMA_ITERS_LEVELS}",
        f"sleep_us_levels: {SLEEP_US_LEVELS}",
        f"threads: {THREADS}",
        f"run_start_monotonic_ns: {run_start_ns}",
        f"run_end_monotonic_ns:   {run_end_ns}",
        f"run_wall_clock_s: {elapsed_s:.2f}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"wrote {meta_path.name}")
    print()
    print(f"Now run: uv run experiments/013-pwrctrl-recipe-sweep/analysis.py "
          f"--prefix {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
