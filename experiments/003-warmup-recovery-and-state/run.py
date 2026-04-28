# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 003: Does a warmup prefix recover sleep_0's tight distribution
at noisy cadences? + GPU state observation triangulated three ways.

Pre-registered. See README.md.

Per CLAUDE.md and the experiment README:
- No averaging in live output, no swallowed exceptions, no retries.
- Raw values to CSV under raw/ with a timestamp prefix.
- Outer loop = warmup_kind, middle = cadence (asc), inner = K (asc).
- Calibration probe (10 sleep_0 dispatches of the measured kernel)
  precedes each (warmup_kind, K, cadence) combination.
- powermetrics sidecar runs only if `sudo -n` works (i.e. user has
  pre-authenticated with `sudo -v`); otherwise we skip it gracefully
  and record that fact in the metadata.
"""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import Metal
import objc


WRITE_TID_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void write_tid(device uint *out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
"""

FMA_LOOP_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Tight FMA loop, fully data-dependent so the optimizer cannot elide.
// 32 threads, each does FMA_ITERS fused multiply-adds. Output is
// nominal (we don't read it back); the point is to push the compute
// pipeline.
constant int FMA_ITERS = 1024;

kernel void fma_loop(device float *out [[buffer(0)]],
                     uint tid [[thread_position_in_grid]]) {
    float x = float(tid) * 0.001f + 0.000001f;
    float y = 1.0f;
    for (int i = 0; i < FMA_ITERS; i++) {
        y = fma(y, x, x);
    }
    out[tid] = y;
}
"""

# Measured kernel is always write_tid with these dimensions.
MEASURED_THREADS = 32
MEASURED_GROUP = 32

# Three warmup kinds. Each entry = (kind_label, kernel_name, threads, group).
WARMUP_KINDS = [
    ("same",        "write_tid", 32,   32),   # identical to measured
    ("heavy_write", "write_tid", 1024, 32),   # 32 threadgroups of write_tid
    ("fma_loop",    "fma_loop",  32,   32),   # arithmetic-heavy
]

K_VALUES = [0, 1, 5, 20]
SLEEP_CONDITIONS_S = [0.0, 0.001, 0.01, 0.1, 1.0]
N_PER_COMBO = 40
CALIBRATION_BURST = 10
FLOOR_WINDOW = (8000, 8200)

# Output buffer needs to hold the largest dispatch (heavy_write 1024 uints = 4 KB).
OUT_BUFFER_BYTES = 4 * 1024
# One big sample buffer reused across combinations. Largest combination is
# CALIBRATION_BURST + N_PER_COMBO dispatches => 2*(10 + 40) = 100 slots.
SAMPLE_BUFFER_SLOTS = 2 * (CALIBRATION_BURST + N_PER_COMBO)

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"


# -- environment probes -----------------------------------------------------

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


# -- caffeinate sidecar -----------------------------------------------------

def start_caffeinate() -> tuple[subprocess.Popen, str]:
    """Prevent display sleep and idle sleep for the duration of the run.

    -d = prevent display sleep, -i = prevent idle sleep, -m = prevent disk
    idle. No sudo needed.
    """
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


# -- powermetrics sidecar ---------------------------------------------------

def maybe_start_powermetrics(out_path: Path) -> tuple[subprocess.Popen | None, str]:
    """Start powermetrics if `sudo -n` works; otherwise return (None, reason)."""
    if os.environ.get("EXP003_NO_POWERMETRICS"):
        return None, "skipped (EXP003_NO_POWERMETRICS set)"
    check = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if check.returncode != 0:
        return None, "sudo -n failed; powermetrics skipped"
    f = open(out_path, "w")
    cmd = ["sudo", "-n", "powermetrics",
           "--samplers", "gpu_power",
           "-i", "200"]
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    # Verify it didn't die immediately
    time.sleep(0.5)
    if proc.poll() is not None:
        return None, f"powermetrics exited immediately rc={proc.returncode}"
    return proc, f"started pid={proc.pid} -> {out_path.name}"


def stop_powermetrics(proc: subprocess.Popen | None) -> str:
    if proc is None:
        return "no sidecar"
    # sudo'd processes do not always forward signals; killall is the
    # robust path.
    subprocess.run(["sudo", "-n", "killall", "-INT", "powermetrics"],
                   capture_output=True)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        subprocess.run(["sudo", "-n", "killall", "-KILL", "powermetrics"],
                       capture_output=True)
        proc.wait(timeout=2)
    return f"stopped rc={proc.returncode}"


# -- Metal setup ------------------------------------------------------------

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


# -- dispatching ------------------------------------------------------------

def dispatch_untimed(queue, pipeline, out_buffer, threads, group):
    """A warmup dispatch: no counter sampling, but we still wait for completion."""
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
    """A measured dispatch: counter samples at stage boundaries; CPU walls too."""
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


# -- combination runner -----------------------------------------------------

def run_calibration_probe(queue, measured_pipeline, out_buffer, sample_buffer,
                           slot_offset):
    """10 back-to-back sleep_0 dispatches of the measured kernel."""
    rows = []
    for i in range(CALIBRATION_BURST):
        slot_pair = (slot_offset + 2 * i, slot_offset + 2 * i + 1)
        cpu = dispatch_timed(
            queue, measured_pipeline, out_buffer,
            MEASURED_THREADS, MEASURED_GROUP,
            sample_buffer, slot_pair,
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


def run_combo(K, sleep_s, queue, measured_pipeline, warmup_pipeline,
              warmup_threads, warmup_group, out_buffer, sample_buffer,
              measured_slot_offset):
    """N trials of: sleep cadence_s, K warmup dispatches, 1 measured dispatch."""
    rows = []
    for trial_idx in range(N_PER_COMBO):
        if sleep_s > 0:
            # Sleep AFTER prior trial (i.e. before this trial's warmup),
            # except for the first trial which has no prior. Matches the
            # mental model "cadence between measurements".
            if trial_idx > 0:
                time.sleep(sleep_s)
        for _ in range(K):
            dispatch_untimed(queue, warmup_pipeline, out_buffer,
                             warmup_threads, warmup_group)
        slot_pair = (measured_slot_offset + 2 * trial_idx,
                     measured_slot_offset + 2 * trial_idx + 1)
        cpu = dispatch_timed(
            queue, measured_pipeline, out_buffer,
            MEASURED_THREADS, MEASURED_GROUP,
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


# -- summary ----------------------------------------------------------------

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
    }


# -- IO ---------------------------------------------------------------------

def write_measured_csv(path, all_rows):
    fieldnames = [
        "warmup_kind", "K", "sleep_s", "trial_idx_within_combo",
        "wall_clock_ns",
        "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
        "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fieldnames})


def write_calibration_csv(path, all_rows):
    fieldnames = [
        "warmup_kind", "K", "sleep_s", "probe_idx_within_burst",
        "wall_clock_ns",
        "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
        "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fieldnames})


# -- main -------------------------------------------------------------------

def main():
    RAW_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    print("=" * 78)
    print("Experiment 003: warmup recovery + GPU state observation")
    print("=" * 78)
    qos_result = set_user_interactive_qos()
    print(f"QoS:   {qos_result}")
    print(f"Power: {power_source()}")
    display_state_start = display_powerstate()
    print(f"Display powerstate (start):\n  {display_state_start}")
    assertions_start = pmset_assertions()
    # Print only the first ~10 lines of assertions to avoid stdout flood
    print(f"pmset assertions (start, first 10 lines):")
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
        write_tid_pipe = build_pipeline(device, WRITE_TID_SOURCE, "write_tid")
        fma_loop_pipe = build_pipeline(device, FMA_LOOP_SOURCE, "fma_loop")
        out_buffer = device.newBufferWithLength_options_(
            OUT_BUFFER_BYTES, Metal.MTLResourceStorageModeShared
        )
        sample_buffer = make_sample_buffer(
            device, counter_set, SAMPLE_BUFFER_SLOTS, "exp003-shared"
        )
        print(f"Device: {device.name()}  registryID={device.registryID()}")
        print(f"sample_buffer slots={SAMPLE_BUFFER_SLOTS}, "
              f"out_buffer bytes={OUT_BUFFER_BYTES}")
        print(f"Conditions: warmup_kinds={[k[0] for k in WARMUP_KINDS]}  "
              f"K={K_VALUES}  sleep_s={SLEEP_CONDITIONS_S}  N={N_PER_COMBO}")

        cpu_ts_start, gpu_ts_start = device.sampleTimestamps_gpuTimestamp_(None, None)
        print(f"timestamp correlation (start): cpu={cpu_ts_start} gpu={gpu_ts_start}")

        all_measured = []
        all_calibration = []
        run_start_wall = time.monotonic_ns()

        for kind_label, kernel_name, w_threads, w_group in WARMUP_KINDS:
            warmup_pipe = (
                fma_loop_pipe if kernel_name == "fma_loop" else write_tid_pipe
            )
            print(f"\n=== Warmup kind: {kind_label} "
                  f"(kernel={kernel_name}, threads={w_threads}, group={w_group}) ===")
            for sleep_s in SLEEP_CONDITIONS_S:
                for K in K_VALUES:
                    label = f"{kind_label} K={K} sleep_s={sleep_s}"
                    print(f"\n  Combo: {label}")
                    # Cooldown: ensure the chip has settled before the
                    # calibration probe samples its current state.
                    cooldown = max(sleep_s, 1.0)
                    time.sleep(cooldown)
                    # Calibration probe (slots 0..19)
                    cal_rows = run_calibration_probe(
                        queue, write_tid_pipe, out_buffer, sample_buffer,
                        slot_offset=0,
                    )
                    for r in cal_rows:
                        r.update({"warmup_kind": kind_label, "K": K, "sleep_s": sleep_s})
                    all_calibration.extend(cal_rows)
                    cal_first = cal_rows[0]["gpu_delta_raw"]
                    cal_last_med = percentile(
                        [r["gpu_delta_raw"] for r in cal_rows[1:]], 50
                    )
                    print(f"    cal_probe: first={cal_first} "
                          f"med_of_rest={cal_last_med:.0f}")
                    # Measured trials (slots 20..99)
                    measured_slot_offset = 2 * CALIBRATION_BURST
                    rows = run_combo(
                        K, sleep_s, queue, write_tid_pipe, warmup_pipe,
                        w_threads, w_group, out_buffer, sample_buffer,
                        measured_slot_offset,
                    )
                    for r in rows:
                        r.update({"warmup_kind": kind_label, "K": K, "sleep_s": sleep_s})
                    all_measured.extend(rows)
                    s = summarize([r["gpu_delta_raw"] for r in rows])
                    cpu_s = summarize([r["cpu_wait_ns"] for r in rows])
                    print(f"    measured (N={s['n']}): "
                          f"min={s['min']} p05={s['p05']:.0f} p50={s['p50']:.0f} "
                          f"p95={s['p95']:.0f} p99={s['p99']:.0f} max={s['max']}")
                    print(f"               cv={s['cv']:.4f}  "
                          f"in_floor={s['in_floor']}/{s['n']}  "
                          f"cpu_wait p50={cpu_s['p50']:.0f}")

        run_end_wall = time.monotonic_ns()
        cpu_ts_end, gpu_ts_end = device.sampleTimestamps_gpuTimestamp_(None, None)

        # Write CSVs
        measured_csv = RAW_DIR / f"{ts}-measured.csv"
        cal_csv = RAW_DIR / f"{ts}-calibration.csv"
        write_measured_csv(measured_csv, all_measured)
        write_calibration_csv(cal_csv, all_calibration)
        print(f"\nwrote {measured_csv.name} ({len(all_measured)} rows)")
        print(f"wrote {cal_csv.name} ({len(all_calibration)} rows)")

        cpu_dt = cpu_ts_end - cpu_ts_start
        gpu_dt = gpu_ts_end - gpu_ts_start
        ratio = (gpu_dt / cpu_dt) if cpu_dt else float("nan")
        elapsed_s = (run_end_wall - run_start_wall) / 1e9
        print(f"\ntimestamp correlation (end):   cpu={cpu_ts_end} gpu={gpu_ts_end}")
        print(f"elapsed cpu_ticks={cpu_dt}  gpu_ticks={gpu_dt}  "
              f"ratio_gpu/cpu={ratio:.6f}")
        print(f"experiment wall-clock: {elapsed_s:.1f}s")

        # Per-combination summary table for the metadata file
        meta_lines = [
            "experiment: 003-warmup-recovery-and-state",
            f"timestamp: {ts}",
            f"device: {device.name()}",
            f"registry_id: {device.registryID()}",
            f"architecture: {device.architecture().name() if hasattr(device, 'architecture') and device.architecture() else '<unavailable>'}",
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
            f"warmup_kinds: {WARMUP_KINDS}",
            f"K_values: {K_VALUES}",
            f"sleep_conditions_s: {SLEEP_CONDITIONS_S}",
            f"n_per_combo: {N_PER_COMBO}",
            f"calibration_burst: {CALIBRATION_BURST}",
            f"floor_window: {FLOOR_WINDOW}",
            "kernel_measured: write_tid 32 threads",
            "sampling_point: MTLCounterSamplingPointAtStageBoundary",
            f"correlation_start: cpu={cpu_ts_start} gpu={gpu_ts_start}",
            f"correlation_end:   cpu={cpu_ts_end} gpu={gpu_ts_end}",
            f"experiment_wall_clock_s: {elapsed_s:.2f}",
            "",
            "per-combo summary (warmup_kind | K | sleep_s | "
            "min | p50 | p95 | p99 | max | cv | in_floor):",
        ]
        # Group rows for summary
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in all_measured:
            grouped[(r["warmup_kind"], r["K"], r["sleep_s"])].append(r["gpu_delta_raw"])
        for (kind, K, sleep_s), deltas in sorted(grouped.items()):
            s = summarize(deltas)
            meta_lines.append(
                f"  {kind:11s} K={K:>2d} sleep_s={sleep_s:<6} "
                f"min={s['min']:>7d} p50={s['p50']:>7.0f} p95={s['p95']:>7.0f} "
                f"p99={s['p99']:>8.0f} max={s['max']:>8d} cv={s['cv']:.4f} "
                f"in_floor={s['in_floor']:>2d}/{s['n']}"
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
