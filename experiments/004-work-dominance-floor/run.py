# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 004: Work-dominance floor + 5.4 µs reproducibility.

Pre-registered. See README.md.

Per CLAUDE.md and the experiment README:
- No averaging in live output, no swallowed exceptions, no retries.
- Raw values to CSV under raw/ with a timestamp prefix.
- Order: 5.4 µs protocol FIRST (before any other GPU work),
  then 3 sequential full sweeps (axis A then axis B per sweep),
  with 30s cooldown between sweeps.
- caffeinate sidecar holds display awake throughout.
- powermetrics gated on EXP004_NO_POWERMETRICS env var (default
  off, set to non-empty to skip).
"""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import Metal
import objc


# Kernel sources

WRITE_TID_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void write_tid(device uint *out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
"""


def fma_loop_source(iters: int) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
constant int FMA_ITERS = {iters};
kernel void fma_loop(device float *out [[buffer(0)]],
                     uint tid [[thread_position_in_grid]]) {{
    float x = float(tid) * 0.001f + 0.000001f;
    float y = 1.0f;
    for (int i = 0; i < FMA_ITERS; i++) {{
        y = fma(y, x, x);
    }}
    out[tid] = y;
}}
"""


# Sweep design

WRITE_TID_THREAD_LEVELS = [
    32, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    1536, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 8388608,
]
FMA_ITERS_LEVELS = [
    16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
]

THREADGROUP_SIZE = 32
FMA_THREAD_COUNT = 32          # fixed for axis B
N_PER_COMBO = 300
N_SWEEPS = 3
CALIBRATION_BURST = 10
PER_COMBO_COOLDOWN_S = 2.0
BETWEEN_SWEEP_COOLDOWN_S = 30.0

# 5.4 µs reproduction protocol
REPRO_ATTEMPTS = 5
REPRO_COOLDOWN_S = 5.0
REPRO_CALIBRATION_BURST = 10
REPRO_MEASURED_TRIALS = 40

FLOOR_WINDOW = (8000, 8200)         # back-to-back floor from 001
LOW_FLOOR_WINDOW = (5000, 6000)     # 5.4 µs floor from 003

# Sample buffer needs to hold the largest single burst we ever resolve
# at once. Largest: per-combo calibration + measured = 10 + 300 = 310
# dispatches => 620 slots. 5.4 µs protocol largest: 10 + 40 = 50
# dispatches => 100 slots. Use 620 throughout.
SAMPLE_BUFFER_SLOTS = 2 * (CALIBRATION_BURST + N_PER_COMBO)

# Output buffer needs to hold the largest dispatch:
# write_tid 8388608 threads writing uint = 32 MiB.
# fma_loop 32 threads writing float = 128 bytes.
OUT_BUFFER_BYTES = 4 * max(WRITE_TID_THREAD_LEVELS)

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"


# Environment / sidecars

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


def display_powerstate() -> str:
    out = subprocess.run(
        ["pmset", "-g", "powerstate", "IODisplayWrangler"],
        capture_output=True, text=True, check=False,
    )
    return out.stdout.strip() if out.stdout else "<unavailable>"


def pmset_assertions() -> str:
    out = subprocess.run(
        ["pmset", "-g", "assertions"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.stdout else "<unavailable>"


def start_caffeinate() -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        ["caffeinate", "-d", "-i", "-m"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    if proc.poll() is not None:
        return proc, f"caffeinate exited immediately rc={proc.returncode}"
    return proc, f"started pid={proc.pid}"


def stop_caffeinate(proc: subprocess.Popen) -> str:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    return f"stopped rc={proc.returncode}"


def maybe_start_powermetrics(out_path: Path) -> tuple[subprocess.Popen | None, str]:
    if os.environ.get("EXP004_NO_POWERMETRICS"):
        return None, "skipped (EXP004_NO_POWERMETRICS set)"
    check = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if check.returncode != 0:
        return None, "sudo -n failed; powermetrics skipped"
    f = open(out_path, "w")
    cmd = ["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-i", "200"]
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.5)
    if proc.poll() is not None:
        return None, f"powermetrics exited immediately rc={proc.returncode}"
    return proc, f"started pid={proc.pid} -> {out_path.name}"


def stop_powermetrics(proc: subprocess.Popen | None) -> str:
    if proc is None:
        return "no sidecar"
    subprocess.run(["sudo", "-n", "killall", "-INT", "powermetrics"],
                   capture_output=True)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        subprocess.run(["sudo", "-n", "killall", "-KILL", "powermetrics"],
                       capture_output=True)
        proc.wait(timeout=2)
    return f"stopped rc={proc.returncode}"


# Metal setup

def find_timestamp_counter_set(device):
    sets = device.counterSets()
    if sets is None:
        raise RuntimeError("device.counterSets() returned None")
    for cs in sets:
        if str(cs.name()) == "timestamp":
            return cs
    raise RuntimeError(f"No 'timestamp' counter set; have {[str(s.name()) for s in sets]}")


def build_pipeline(device, source: str, fn_name: str):
    options = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None:
        raise RuntimeError(f"Kernel compile failed for {fn_name}: {err}")
    fn = library.newFunctionWithName_(fn_name)
    if fn is None:
        raise RuntimeError(f"Function {fn_name} not in library")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"Pipeline creation failed for {fn_name}: {err}")
    return pipeline


def make_sample_buffer(device, counter_set, sample_count, label):
    desc = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
    desc.setCounterSet_(counter_set)
    desc.setSampleCount_(sample_count)
    desc.setStorageMode_(Metal.MTLStorageModeShared)
    desc.setLabel_(label)
    buf, err = device.newCounterSampleBufferWithDescriptor_error_(desc, None)
    if buf is None:
        raise RuntimeError(f"Counter sample buffer alloc failed: {err}")
    return buf


# Dispatching

def dispatch_untimed(queue, pipeline, out_buffer, threads, group):
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


def dispatch_timed(queue, pipeline, out_buffer, threads, group,
                    sample_buffer, slot_pair):
    start_idx, end_idx = slot_pair
    pass_desc = Metal.MTLComputePassDescriptor.computePassDescriptor()
    att = pass_desc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
    att.setSampleBuffer_(sample_buffer)
    att.setStartOfEncoderSampleIndex_(start_idx)
    att.setEndOfEncoderSampleIndex_(end_idx)

    cb = queue.commandBuffer()

    wall_clock_ns = time.monotonic_ns()
    cpu_t0 = time.perf_counter_ns()
    encoder = cb.computeCommandEncoderWithDescriptor_(pass_desc)
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out_buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(threads, 1, 1),
        Metal.MTLSizeMake(group, 1, 1),
    )
    encoder.endEncoding()
    cpu_t1 = time.perf_counter_ns()
    cb.commit()
    cpu_t2 = time.perf_counter_ns()
    cb.waitUntilCompleted()
    cpu_t3 = time.perf_counter_ns()

    return {
        "wall_clock_ns": wall_clock_ns,
        "cpu_encode_ns": cpu_t1 - cpu_t0,
        "cpu_commit_ns": cpu_t2 - cpu_t1,
        "cpu_wait_ns": cpu_t3 - cpu_t2,
        "cpu_total_ns": cpu_t3 - cpu_t0,
    }


def resolve_pair(sample_buffer, start_idx):
    data = sample_buffer.resolveCounterRange_((start_idx, 2))
    if data is None or data.length() < 16:
        raise RuntimeError(f"resolveCounterRange({start_idx},2) returned {data}")
    raw = bytes(data)
    return (
        int.from_bytes(raw[0:8], "little", signed=False),
        int.from_bytes(raw[8:16], "little", signed=False),
    )


# Calibration probe (always uses write_tid 32t)

def run_calibration(queue, write_tid_pipeline, out_buffer, sample_buffer,
                     slot_offset):
    rows = []
    for i in range(CALIBRATION_BURST):
        slot_pair = (slot_offset + 2 * i, slot_offset + 2 * i + 1)
        cpu = dispatch_timed(
            queue, write_tid_pipeline, out_buffer,
            32, THREADGROUP_SIZE, sample_buffer, slot_pair,
        )
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot_pair[0])
        rows.append({
            "probe_idx_within_burst": i,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
    return rows


# Combo runner: K=1 untimed warmup of measured kernel, then 1 measured

def run_combo(queue, measured_pipeline, threads, group, out_buffer,
              sample_buffer, measured_slot_offset, n_trials):
    rows = []
    for trial_idx in range(n_trials):
        # K=1 warmup of the same kernel with the same parameters
        dispatch_untimed(queue, measured_pipeline, out_buffer, threads, group)
        slot_pair = (measured_slot_offset + 2 * trial_idx,
                     measured_slot_offset + 2 * trial_idx + 1)
        cpu = dispatch_timed(
            queue, measured_pipeline, out_buffer, threads, group,
            sample_buffer, slot_pair,
        )
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot_pair[0])
        rows.append({
            "trial_idx_within_combo": trial_idx,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
    return rows


# 5.4 µs reproduction protocol: NO warmup, sleep_0 between trials
def run_repro_attempt(queue, write_tid_pipeline, out_buffer, sample_buffer,
                       cal_offset, meas_offset):
    cal_rows = []
    for i in range(REPRO_CALIBRATION_BURST):
        slot_pair = (cal_offset + 2 * i, cal_offset + 2 * i + 1)
        cpu = dispatch_timed(
            queue, write_tid_pipeline, out_buffer,
            32, THREADGROUP_SIZE, sample_buffer, slot_pair,
        )
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot_pair[0])
        cal_rows.append({
            "phase": "calibration",
            "idx_within_phase": i,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
    meas_rows = []
    for i in range(REPRO_MEASURED_TRIALS):
        slot_pair = (meas_offset + 2 * i, meas_offset + 2 * i + 1)
        cpu = dispatch_timed(
            queue, write_tid_pipeline, out_buffer,
            32, THREADGROUP_SIZE, sample_buffer, slot_pair,
        )
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot_pair[0])
        meas_rows.append({
            "phase": "measured",
            "idx_within_phase": i,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
    return cal_rows + meas_rows


# Summary helpers

def percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def stddev(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def summarize(deltas):
    p50 = percentile(deltas, 50)
    return {
        "n": len(deltas),
        "min": min(deltas),
        "p05": percentile(deltas, 5),
        "p50": p50,
        "p95": percentile(deltas, 95),
        "p99": percentile(deltas, 99),
        "max": max(deltas),
        "stddev": stddev(deltas),
        "cv": (stddev(deltas) / p50) if p50 else float("nan"),
        "in_floor": sum(1 for d in deltas if FLOOR_WINDOW[0] <= d <= FLOOR_WINDOW[1]),
        "in_low_floor": sum(1 for d in deltas if LOW_FLOOR_WINDOW[0] <= d <= LOW_FLOOR_WINDOW[1]),
    }


# CSV writers

MEASURED_FIELDS = [
    "axis", "complexity_level", "sweep_idx", "trial_idx_within_combo",
    "wall_clock_ns",
    "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
    "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
]
CALIBRATION_FIELDS = [
    "axis", "complexity_level", "sweep_idx", "probe_idx_within_burst",
    "wall_clock_ns",
    "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
    "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
]
REPRO_FIELDS = [
    "attempt_idx", "phase", "idx_within_phase",
    "wall_clock_ns",
    "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
    "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


# Main

def main():
    RAW_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    print("=" * 78)
    print("Experiment 004: work-dominance floor + 5.4us reproducibility")
    print("=" * 78)

    qos_result = set_user_interactive_qos()
    print(f"QoS:   {qos_result}")
    print(f"Power: {power_source()}")
    display_state_start = display_powerstate()
    print(f"Display powerstate (start):\n  {display_state_start}")
    assertions_start = pmset_assertions()
    print("pmset assertions (start, first 10 lines):")
    for line in assertions_start.splitlines()[:10]:
        print(f"  {line}")

    caffeinate_proc, caffeinate_status = start_caffeinate()
    print(f"caffeinate: {caffeinate_status}")

    pm_path = RAW_DIR / f"{ts}-powermetrics.txt"
    pm_proc, pm_status = maybe_start_powermetrics(pm_path)
    print(f"powermetrics: {pm_status}")
    pm_started_wall_ns = time.monotonic_ns()

    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("MTLCreateSystemDefaultDevice returned nil")
        queue = device.newCommandQueue()
        counter_set = find_timestamp_counter_set(device)
        out_buffer = device.newBufferWithLength_options_(
            OUT_BUFFER_BYTES, Metal.MTLResourceStorageModeShared
        )
        sample_buffer = make_sample_buffer(
            device, counter_set, SAMPLE_BUFFER_SLOTS, "exp004-shared"
        )

        # Build pipelines:
        # 1 write_tid pipeline (single source, dispatched at varying thread counts)
        # 17 fma_loop pipelines, one per FMA_ITERS level
        write_tid_pipe = build_pipeline(device, WRITE_TID_SOURCE, "write_tid")
        fma_pipes = {}
        print(f"\nCompiling {len(FMA_ITERS_LEVELS)} fma_loop variants ...")
        for iters in FMA_ITERS_LEVELS:
            fma_pipes[iters] = build_pipeline(
                device, fma_loop_source(iters), "fma_loop"
            )
        print(f"  done. write_tid TEW={write_tid_pipe.threadExecutionWidth()} "
              f"maxTPG={write_tid_pipe.maxTotalThreadsPerThreadgroup()}")
        print(f"Conditions:")
        print(f"  write_tid_threadcount levels: {WRITE_TID_THREAD_LEVELS}")
        print(f"  fma_loop_iters levels:        {FMA_ITERS_LEVELS}")
        print(f"  N_per_combo={N_PER_COMBO}  N_sweeps={N_SWEEPS}")
        print(f"  per-combo cooldown={PER_COMBO_COOLDOWN_S}s  "
              f"between-sweep cooldown={BETWEEN_SWEEP_COOLDOWN_S}s")
        print(f"  sample_buffer slots={SAMPLE_BUFFER_SLOTS}, "
              f"out_buffer bytes={OUT_BUFFER_BYTES}")

        cpu_ts_start, gpu_ts_start = device.sampleTimestamps_gpuTimestamp_(None, None)
        print(f"timestamp correlation (start): cpu={cpu_ts_start} gpu={gpu_ts_start}")

        run_start_wall = time.monotonic_ns()
        all_measured = []
        all_calibration = []
        all_repro = []

        # PHASE 1: 5.4 µs reproduction protocol (BEFORE any other GPU work)
        print(f"\n{'=' * 78}\nPHASE 1: 5.4us reproducibility protocol "
              f"({REPRO_ATTEMPTS} attempts)\n{'=' * 78}")
        for attempt_idx in range(REPRO_ATTEMPTS):
            print(f"\n  Attempt {attempt_idx+1}/{REPRO_ATTEMPTS}: "
                  f"cooldown {REPRO_COOLDOWN_S}s ...")
            time.sleep(REPRO_COOLDOWN_S)
            cal_offset = 0
            meas_offset = 2 * REPRO_CALIBRATION_BURST
            rows = run_repro_attempt(
                queue, write_tid_pipe, out_buffer, sample_buffer,
                cal_offset, meas_offset,
            )
            for r in rows:
                r["attempt_idx"] = attempt_idx
            all_repro.extend(rows)
            cal_rows = [r for r in rows if r["phase"] == "calibration"]
            meas_rows = [r for r in rows if r["phase"] == "measured"]
            cal_first = cal_rows[0]["gpu_delta_raw"]
            cal_med_rest = percentile(
                [r["gpu_delta_raw"] for r in cal_rows[1:]], 50
            )
            meas_deltas = [r["gpu_delta_raw"] for r in meas_rows]
            s = summarize(meas_deltas)
            in_low = s["in_low_floor"]
            print(f"    cal: first={cal_first} med_of_rest={cal_med_rest:.0f}")
            print(f"    measured (N={s['n']}): min={s['min']} p50={s['p50']:.0f} "
                  f"p95={s['p95']:.0f} max={s['max']}  cv={s['cv']:.4f}  "
                  f"in_low_floor[5000,6000]={in_low}/{s['n']}")

        # PHASE 2: full sweeps
        print(f"\n{'=' * 78}\nPHASE 2: {N_SWEEPS} sequential sweeps\n{'=' * 78}")
        for sweep_idx in range(N_SWEEPS):
            print(f"\n--- Sweep {sweep_idx+1}/{N_SWEEPS} ---")

            # Axis A: write_tid thread count
            print(f"\n  Axis A: write_tid thread count ({len(WRITE_TID_THREAD_LEVELS)} levels)")
            for level in WRITE_TID_THREAD_LEVELS:
                time.sleep(PER_COMBO_COOLDOWN_S)
                # Calibration probe (slots 0..19)
                cal_rows = run_calibration(
                    queue, write_tid_pipe, out_buffer, sample_buffer, 0
                )
                for r in cal_rows:
                    r.update({
                        "axis": "write_tid_threadcount",
                        "complexity_level": level,
                        "sweep_idx": sweep_idx,
                    })
                all_calibration.extend(cal_rows)
                # Measured trials (slots 20..)
                meas_offset = 2 * CALIBRATION_BURST
                rows = run_combo(
                    queue, write_tid_pipe, level, THREADGROUP_SIZE,
                    out_buffer, sample_buffer, meas_offset, N_PER_COMBO,
                )
                for r in rows:
                    r.update({
                        "axis": "write_tid_threadcount",
                        "complexity_level": level,
                        "sweep_idx": sweep_idx,
                    })
                all_measured.extend(rows)
                s = summarize([r["gpu_delta_raw"] for r in rows])
                print(f"    threads={level:>8d}  N={s['n']}  "
                      f"p50={s['p50']:>9.0f}  p95={s['p95']:>9.0f}  "
                      f"p99={s['p99']:>9.0f}  cv={s['cv']:.4f}")

            # Axis B: fma_loop iters
            print(f"\n  Axis B: fma_loop iters ({len(FMA_ITERS_LEVELS)} levels)")
            for level in FMA_ITERS_LEVELS:
                time.sleep(PER_COMBO_COOLDOWN_S)
                cal_rows = run_calibration(
                    queue, write_tid_pipe, out_buffer, sample_buffer, 0
                )
                for r in cal_rows:
                    r.update({
                        "axis": "fma_loop_iters",
                        "complexity_level": level,
                        "sweep_idx": sweep_idx,
                    })
                all_calibration.extend(cal_rows)
                meas_offset = 2 * CALIBRATION_BURST
                rows = run_combo(
                    queue, fma_pipes[level], FMA_THREAD_COUNT, THREADGROUP_SIZE,
                    out_buffer, sample_buffer, meas_offset, N_PER_COMBO,
                )
                for r in rows:
                    r.update({
                        "axis": "fma_loop_iters",
                        "complexity_level": level,
                        "sweep_idx": sweep_idx,
                    })
                all_measured.extend(rows)
                s = summarize([r["gpu_delta_raw"] for r in rows])
                print(f"    fma_iters={level:>5d}  N={s['n']}  "
                      f"p50={s['p50']:>9.0f}  p95={s['p95']:>9.0f}  "
                      f"p99={s['p99']:>9.0f}  cv={s['cv']:.4f}")

            if sweep_idx < N_SWEEPS - 1:
                print(f"\n  Between-sweep cooldown {BETWEEN_SWEEP_COOLDOWN_S}s ...")
                time.sleep(BETWEEN_SWEEP_COOLDOWN_S)

        run_end_wall = time.monotonic_ns()
        cpu_ts_end, gpu_ts_end = device.sampleTimestamps_gpuTimestamp_(None, None)

        # Write CSVs
        measured_csv = RAW_DIR / f"{ts}-measured.csv"
        cal_csv = RAW_DIR / f"{ts}-calibration.csv"
        repro_csv = RAW_DIR / f"{ts}-repro.csv"
        write_csv(measured_csv, MEASURED_FIELDS, all_measured)
        write_csv(cal_csv, CALIBRATION_FIELDS, all_calibration)
        write_csv(repro_csv, REPRO_FIELDS, all_repro)
        print(f"\nwrote {measured_csv.name} ({len(all_measured)} rows)")
        print(f"wrote {cal_csv.name} ({len(all_calibration)} rows)")
        print(f"wrote {repro_csv.name} ({len(all_repro)} rows)")

        cpu_dt = cpu_ts_end - cpu_ts_start
        gpu_dt = gpu_ts_end - gpu_ts_start
        ratio = (gpu_dt / cpu_dt) if cpu_dt else float("nan")
        elapsed_s = (run_end_wall - run_start_wall) / 1e9
        print(f"\ntimestamp correlation (end):   cpu={cpu_ts_end} gpu={gpu_ts_end}")
        print(f"elapsed cpu_ticks={cpu_dt}  gpu_ticks={gpu_dt}  "
              f"ratio_gpu/cpu={ratio:.6f}")
        print(f"experiment wall-clock: {elapsed_s:.1f}s")

        # Per-combo per-sweep summary for the metadata file
        meta_lines = [
            "experiment: 004-work-dominance-floor",
            f"timestamp: {ts}",
            f"device: {device.name()}",
            f"registry_id: {device.registryID()}",
        ]
        if hasattr(device, "architecture") and device.architecture():
            meta_lines.append(f"architecture: {device.architecture().name()}")
        meta_lines.extend([
            f"chip: {platform.processor()}  machine: {platform.machine()}",
            f"os: {platform.platform()}",
            f"python: {sys.version.splitlines()[0]}",
            f"pyobjc: {objc.__version__}",
            f"qos: {qos_result}",
            f"power: {power_source()}",
            f"powermetrics: {pm_status}",
            f"powermetrics_started_wall_ns: {pm_started_wall_ns}",
            f"caffeinate: {caffeinate_status}",
            f"display_powerstate_start: {display_state_start}",
            f"display_powerstate_end:   {display_powerstate()}",
            "pmset_assertions_start: |",
            *[f"  {line}" for line in assertions_start.splitlines()],
            f"experiment_started_wall_ns: {run_start_wall}",
            f"experiment_ended_wall_ns:   {run_end_wall}",
            f"write_tid_thread_levels: {WRITE_TID_THREAD_LEVELS}",
            f"fma_iters_levels: {FMA_ITERS_LEVELS}",
            f"threadgroup_size: {THREADGROUP_SIZE}",
            f"fma_thread_count: {FMA_THREAD_COUNT}",
            f"n_per_combo: {N_PER_COMBO}",
            f"n_sweeps: {N_SWEEPS}",
            f"per_combo_cooldown_s: {PER_COMBO_COOLDOWN_S}",
            f"between_sweep_cooldown_s: {BETWEEN_SWEEP_COOLDOWN_S}",
            f"calibration_burst: {CALIBRATION_BURST}",
            f"repro_attempts: {REPRO_ATTEMPTS}",
            f"repro_cooldown_s: {REPRO_COOLDOWN_S}",
            f"repro_calibration_burst: {REPRO_CALIBRATION_BURST}",
            f"repro_measured_trials: {REPRO_MEASURED_TRIALS}",
            f"floor_window: {FLOOR_WINDOW}",
            f"low_floor_window: {LOW_FLOOR_WINDOW}",
            "kernel_measured_axisA: write_tid (varied thread count)",
            "kernel_measured_axisB: fma_loop (32 threads, varied per-thread iters)",
            "kernel_calibration: write_tid 32 threads",
            "warmup_recipe: K=1 untimed dispatch of measured kernel "
            "(per 003 finding)",
            "sampling_point: MTLCounterSamplingPointAtStageBoundary",
            f"correlation_start: cpu={cpu_ts_start} gpu={gpu_ts_start}",
            f"correlation_end:   cpu={cpu_ts_end} gpu={gpu_ts_end}",
            f"experiment_wall_clock_s: {elapsed_s:.2f}",
            "",
            "per-(axis, level, sweep) summary "
            "(min | p50 | p95 | p99 | max | cv | in_floor):",
        ])
        grouped = defaultdict(list)
        for r in all_measured:
            grouped[(r["axis"], r["complexity_level"], r["sweep_idx"])].append(
                r["gpu_delta_raw"]
            )
        for (axis, level, sweep), deltas in sorted(grouped.items()):
            s = summarize(deltas)
            meta_lines.append(
                f"  {axis:24s} level={level:>8d} sweep={sweep} "
                f"min={s['min']:>7d} p50={s['p50']:>9.0f} p95={s['p95']:>9.0f} "
                f"p99={s['p99']:>10.0f} max={s['max']:>10d} cv={s['cv']:.4f} "
                f"in_floor={s['in_floor']:>3d}/{s['n']}"
            )
        meta_lines.append("")
        meta_lines.append("5.4us reproduction summary:")
        repro_grouped = defaultdict(list)
        for r in all_repro:
            if r["phase"] == "measured":
                repro_grouped[r["attempt_idx"]].append(r["gpu_delta_raw"])
        for attempt_idx, deltas in sorted(repro_grouped.items()):
            s = summarize(deltas)
            meta_lines.append(
                f"  attempt {attempt_idx}: min={s['min']:>5d} p50={s['p50']:>6.0f} "
                f"p95={s['p95']:>6.0f} cv={s['cv']:.4f} "
                f"in_low_floor[5000,6000]={s['in_low_floor']:>2d}/{s['n']}"
            )
        meta_path = RAW_DIR / f"{ts}-meta.txt"
        meta_path.write_text("\n".join(meta_lines) + "\n")
        print(f"wrote {meta_path.name}")
    finally:
        pm_stop_status = stop_powermetrics(pm_proc)
        print(f"\npowermetrics: {pm_stop_status}")
        caffeinate_stop_status = stop_caffeinate(caffeinate_proc)
        print(f"caffeinate: {caffeinate_stop_status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
