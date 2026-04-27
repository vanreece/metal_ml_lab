# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyobjc-framework-Metal>=10.0",
# ]
# ///
"""
Experiment 002: How does the GPU dispatch-timing distribution depend on
inter-dispatch idle?

Pre-registered. See README.md.

Per CLAUDE.md / experiment README:
- No warmup, no retries, no averaging in live output, no swallowed
  exceptions. Means are deliberately absent — the distributions are
  not Gaussian and means hide their shape.
- Raw values to CSV under raw/ with a timestamp prefix.
- Conditions run sleep-ascending so any thermal drift is monotonically
  visible rather than hidden by random ordering.
"""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import Metal
import objc


KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void write_tid(device uint *out [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
"""

THREADS_PER_GRID = 32
THREADS_PER_GROUP = 32
N_PER_CONDITION = 200
SLEEP_CONDITIONS_S = [0.0, 0.001, 0.01, 0.1, 1.0]
FLOOR_WINDOW = (8000, 8200)  # back-to-back floor from 001

EXPERIMENT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXPERIMENT_DIR / "raw"


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


def find_timestamp_counter_set(device):
    sets = device.counterSets()
    if sets is None:
        raise RuntimeError("device.counterSets() returned None")
    for cs in sets:
        if str(cs.name()) == "timestamp":
            return cs
    raise RuntimeError(f"No 'timestamp' counter set; have {[str(s.name()) for s in sets]}")


def build_pipeline(device):
    options = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(
        KERNEL_SOURCE, options, None
    )
    if library is None:
        raise RuntimeError(f"Kernel compile failed: {err}")
    fn = library.newFunctionWithName_("write_tid")
    if fn is None:
        raise RuntimeError("Could not find 'write_tid'")
    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if pipeline is None:
        raise RuntimeError(f"Pipeline creation failed: {err}")
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


def dispatch_once(queue, pipeline, out_buffer, sample_buffer, slot_pair):
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
        Metal.MTLSizeMake(THREADS_PER_GRID, 1, 1),
        Metal.MTLSizeMake(THREADS_PER_GROUP, 1, 1),
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


def run_condition(sleep_s, queue, pipeline, out_buffer, sample_buffer):
    label = condition_label(sleep_s)
    rows = []
    for i in range(N_PER_CONDITION):
        if sleep_s > 0 and i > 0:
            time.sleep(sleep_s)
        slot_pair = (2 * i, 2 * i + 1)
        cpu = dispatch_once(queue, pipeline, out_buffer, sample_buffer, slot_pair)
        gpu_start, gpu_end = resolve_pair(sample_buffer, slot_pair[0])
        rows.append({
            "sample_idx": i,
            "sleep_s": sleep_s,
            "wall_clock_ns": cpu["wall_clock_ns"],
            "gpu_t_start_raw": gpu_start,
            "gpu_t_end_raw": gpu_end,
            "gpu_delta_raw": gpu_end - gpu_start,
            "cpu_encode_ns": cpu["cpu_encode_ns"],
            "cpu_commit_ns": cpu["cpu_commit_ns"],
            "cpu_wait_ns": cpu["cpu_wait_ns"],
            "cpu_total_ns": cpu["cpu_total_ns"],
        })
        # Live progress, sparse
        if i == 0 or i == N_PER_CONDITION - 1 or (i + 1) % 50 == 0:
            print(
                f"  [{label}] {i+1:3d}/{N_PER_CONDITION}  "
                f"gpu_delta_raw={rows[-1]['gpu_delta_raw']:>10d}  "
                f"cpu_wait_ns={rows[-1]['cpu_wait_ns']:>10d}"
            )
    return rows


def condition_label(sleep_s: float) -> str:
    if sleep_s == 0:
        return "sleep_0"
    if sleep_s < 1:
        return f"sleep_{int(sleep_s * 1000)}ms"
    return f"sleep_{int(sleep_s)}s"


def write_csv(path, rows):
    fieldnames = [
        "sample_idx", "sleep_s", "wall_clock_ns",
        "gpu_t_start_raw", "gpu_t_end_raw", "gpu_delta_raw",
        "cpu_encode_ns", "cpu_commit_ns", "cpu_wait_ns", "cpu_total_ns",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


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


def summarize(rows, key):
    xs = [r[key] for r in rows]
    p50 = percentile(xs, 50)
    return {
        "n": len(xs),
        "min": min(xs),
        "p05": percentile(xs, 5),
        "p50": p50,
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": max(xs),
        "stddev": stddev(xs),
        "cv": (stddev(xs) / p50) if p50 else float("nan"),
    }


def floor_count(rows, lo, hi):
    return sum(1 for r in rows if lo <= r["gpu_delta_raw"] <= hi)


def fmt_summary(label, gpu, cpu, in_floor):
    return (
        f"\n  {label} (N={gpu['n']})\n"
        f"    gpu_delta_raw: min={gpu['min']:>8d}  p05={gpu['p05']:>8.0f}  "
        f"p50={gpu['p50']:>8.0f}  p95={gpu['p95']:>8.0f}  "
        f"p99={gpu['p99']:>8.0f}  max={gpu['max']:>9d}\n"
        f"                   cv(stddev/p50)={gpu['cv']:.4f}  "
        f"in_floor[{FLOOR_WINDOW[0]},{FLOOR_WINDOW[1]}]={in_floor}/{gpu['n']}\n"
        f"    cpu_wait_ns:   min={cpu['min']:>8d}  p05={cpu['p05']:>8.0f}  "
        f"p50={cpu['p50']:>8.0f}  p95={cpu['p95']:>8.0f}  "
        f"p99={cpu['p99']:>8.0f}  max={cpu['max']:>9d}"
    )


def main():
    RAW_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    print("=" * 78)
    print("Experiment 002: noise floor vs inter-dispatch idle on M1 Pro")
    print("=" * 78)

    qos_result = set_user_interactive_qos()
    print(f"QoS: {qos_result}")
    print(f"Power: {power_source()}")

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("MTLCreateSystemDefaultDevice returned nil")
    queue = device.newCommandQueue()
    counter_set = find_timestamp_counter_set(device)
    pipeline = build_pipeline(device)
    out_buffer = device.newBufferWithLength_options_(
        4 * THREADS_PER_GRID, Metal.MTLResourceStorageModeShared
    )
    print(f"Device: {device.name()}  registryID={device.registryID()}")
    print(f"pipeline.threadExecutionWidth={pipeline.threadExecutionWidth()}  "
          f"maxTotalThreadsPerThreadgroup={pipeline.maxTotalThreadsPerThreadgroup()}")
    print(f"conditions: sleep_s in {SLEEP_CONDITIONS_S}  N_per_condition={N_PER_CONDITION}")

    cpu_ts_start, gpu_ts_start = device.sampleTimestamps_gpuTimestamp_(None, None)
    print(f"timestamp correlation (start): cpu={cpu_ts_start}  gpu={gpu_ts_start}")

    run_start_wall = time.monotonic_ns()
    summaries = {}
    for sleep_s in SLEEP_CONDITIONS_S:
        label = condition_label(sleep_s)
        est_s = sleep_s * (N_PER_CONDITION - 1)
        print(f"\nCondition {label}: sleep_s={sleep_s} (estimated >={est_s:.1f}s)")
        sample_buffer = make_sample_buffer(
            device, counter_set, 2 * N_PER_CONDITION, f"exp002-{label}"
        )
        rows = run_condition(sleep_s, queue, pipeline, out_buffer, sample_buffer)
        csv_path = RAW_DIR / f"{ts}-{label}.csv"
        write_csv(csv_path, rows)
        gpu = summarize(rows, "gpu_delta_raw")
        cpu = summarize(rows, "cpu_wait_ns")
        in_floor = floor_count(rows, *FLOOR_WINDOW)
        summaries[label] = {"gpu": gpu, "cpu": cpu, "in_floor": in_floor}
        print(f"  wrote {csv_path.name}")
        print(fmt_summary(label, gpu, cpu, in_floor))
    run_end_wall = time.monotonic_ns()

    cpu_ts_end, gpu_ts_end = device.sampleTimestamps_gpuTimestamp_(None, None)
    cpu_dt = cpu_ts_end - cpu_ts_start
    gpu_dt = gpu_ts_end - gpu_ts_start
    ratio = (gpu_dt / cpu_dt) if cpu_dt else float("nan")
    print(f"\ntimestamp correlation (end):   cpu={cpu_ts_end}  gpu={gpu_ts_end}")
    print(f"elapsed cpu_ticks={cpu_dt}  gpu_ticks={gpu_dt}  ratio_gpu/cpu={ratio:.6f}")
    print(f"total wall-clock: {(run_end_wall - run_start_wall) / 1e9:.1f}s")

    meta = RAW_DIR / f"{ts}-meta.txt"
    lines = [
        "experiment: 002-noise-floor-vs-idle",
        f"timestamp: {ts}",
        f"device: {device.name()}",
        f"registry_id: {device.registryID()}",
        f"chip: {platform.processor()}  machine: {platform.machine()}",
        f"os: {platform.platform()}",
        f"python: {sys.version.splitlines()[0]}",
        f"pyobjc: {objc.__version__}",
        f"qos: {qos_result}",
        f"power: {power_source()}",
        f"n_per_condition: {N_PER_CONDITION}",
        f"sleep_conditions_s: {SLEEP_CONDITIONS_S}",
        f"threads_per_grid: {THREADS_PER_GRID}",
        f"threads_per_threadgroup: {THREADS_PER_GROUP}",
        f"thread_execution_width: {pipeline.threadExecutionWidth()}",
        "kernel: write_tid (out[tid] = tid)",
        "sampling_point: MTLCounterSamplingPointAtStageBoundary",
        "counter_set: timestamp",
        f"correlation_start: cpu={cpu_ts_start} gpu={gpu_ts_start}",
        f"correlation_end:   cpu={cpu_ts_end} gpu={gpu_ts_end}",
        f"floor_window: {FLOOR_WINDOW}",
        "",
        "per-condition summary:",
    ]
    for label, s in summaries.items():
        g = s["gpu"]
        lines.append(
            f"  {label}: gpu_delta_raw min={g['min']} p50={g['p50']:.0f} "
            f"p95={g['p95']:.0f} p99={g['p99']:.0f} max={g['max']} "
            f"cv={g['cv']:.4f} in_floor={s['in_floor']}/{g['n']}"
        )
    meta.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {meta.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
